"""
Scale-RAE Architecture Mixin (Object-Centric 확장 포함)

GPU (CUDA) 환경 전용.

이슈 2 수정:
  - prepare_inputs_labels_for_multimodal에서 pred_image_features_batched를
    self._oc_gt_cache에 저장 → scale_rae_qwen2.py의 _forward_oc_train에서 사용

이슈 3 수정:
  - OC slot 추가 전 text 부분을 미리 clip하여 model_max_length 초과 방지
  - 전체 시퀀스 = img + query + slots + text ≤ model_max_length 보장

이슈 4 (multi-image offset) 사전 처리:
  - images_per_batch를 self._oc_images_per_batch에 저장
    → scale_rae_qwen2.py에서 정확한 offset 계산에 사용
"""

from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import logging
logger = logging.getLogger(__name__)

from .multimodal_encoder.builder import build_vision_tower_aux_list
from .multimodal_projector.builder import build_vision_projector

from scale_rae.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from scale_rae.utils import IS_XLA_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# GPU용 커스텀 커널
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_kernel(
    input_embeds:  torch.Tensor,
    img_embeds:    torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """
    이미지 임베딩을 텍스트 임베딩 시퀀스의 지정 위치에 삽입.
    GPU path: scatter/gather 방식.
    """
    if IS_XLA_AVAILABLE:
        from .reference.custom_kernel import _xla_custom_kernel
        return _xla_custom_kernel(input_embeds, img_embeds, token_indices)

    B, L_text, D = input_embeds.shape
    _, L_img,  _ = img_embeds.shape

    combined     = torch.cat([input_embeds, img_embeds], dim=1)
    indices_safe = token_indices.clamp(0, L_text + L_img - 1)
    idx_expanded = indices_safe.unsqueeze(-1).expand(-1, -1, D)
    output       = combined.gather(dim=1, index=idx_expanded)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# ScaleRAEMetaModel
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEMetaModel:

    def __init__(self, config):
        super(ScaleRAEMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):
            self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True)
            config.mm_hidden_size = sum(
                vt.hidden_size for vt in self.vision_tower_aux_list
            )
            self.mm_projector = build_vision_projector(config)

        vision_loss_mode_cfg = getattr(config, 'vision_loss_mode', 'causal')
        if vision_loss_mode_cfg in ('query', 'block', 'half-query', 'query-block'):
            vision_token_len = getattr(config, 'vision_tower_aux_token_len_list', [256])[0]
            embed_std        = 1.0 / (config.hidden_size ** 0.5)
            self.latent_queries = nn.Parameter(
                torch.randn(vision_token_len, config.hidden_size) * embed_std
            )
        else:
            self.latent_queries = None

    def get_vision_tower_aux_list(self):
        return getattr(self, 'vision_tower_aux_list', None)

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower_aux_list            = model_args.vision_tower_aux_list
        vision_tower_aux_token_len_list  = model_args.vision_tower_aux_token_len_list
        mm_vision_select_layer           = model_args.mm_vision_select_layer
        mm_vision_select_feature         = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter          = model_args.pretrain_mm_mlp_adapter
        pretrain_adapter_and_vision_head = getattr(model_args, 'pretrain_adapter_and_vision_head', None)
        connector_only                   = model_args.connector_only

        self.config.mm_vision_tower_aux_list           = vision_tower_aux_list
        self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list
        self.config.connector_only                     = connector_only

        if self.get_vision_tower_aux_list() is None:
            self.vision_tower_aux_list = build_vision_tower_aux_list(model_args)
        else:
            for vt in self.vision_tower_aux_list:
                vt.load_model()

        self.config.use_mm_proj              = True
        self.config.mm_projector_type        = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer   = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.config.mm_hidden_size = sum(
                vt.hidden_size for vt in self.vision_tower_aux_list
            )
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(w, key):
                return {k.split(key + '.')[1]: v for k, v in w.items() if key + '.' in k}
            self.mm_projector.load_state_dict(get_w(mm_weights, 'mm_projector'), strict=True)

        if pretrain_adapter_and_vision_head is not None:
            logger.info(f"adapter+vision_head 가중치 로드: {pretrain_adapter_and_vision_head}")
            if os.path.isdir(pretrain_adapter_and_vision_head):
                # safetensors 형식 디렉토리 (HuggingFace 체크포인트)
                # from_pretrained가 이미 기본 가중치를 로드했으므로
                # model.safetensors.index.json이 있으면 safetensors로 로드
                index_path = os.path.join(pretrain_adapter_and_vision_head, 'model.safetensors.index.json')
                single_path = os.path.join(pretrain_adapter_and_vision_head, 'model.safetensors')
                if os.path.isfile(index_path):
                    import json
                    from safetensors.torch import load_file as st_load
                    with open(index_path) as f:
                        index = json.load(f)
                    shard_files = sorted(set(index['weight_map'].values()))
                    model_weights = {}
                    for shard in shard_files:
                        shard_path = os.path.join(pretrain_adapter_and_vision_head, shard)
                        model_weights.update(st_load(shard_path, device='cpu'))
                elif os.path.isfile(single_path):
                    from safetensors.torch import load_file as st_load
                    model_weights = st_load(single_path, device='cpu')
                else:
                    # pytorch_model.bin fallback
                    bin_path = os.path.join(pretrain_adapter_and_vision_head, 'pytorch_model.bin')
                    if os.path.isfile(bin_path):
                        weights = torch.load(bin_path, map_location='cpu')
                        model_weights = weights.get('model', weights)
                    else:
                        logger.warning(f"pretrain_adapter_and_vision_head 디렉토리에서 가중치 파일을 찾을 수 없음: {pretrain_adapter_and_vision_head}")
                        model_weights = {}
            else:
                weights = torch.load(pretrain_adapter_and_vision_head, map_location='cpu')
                model_weights = weights.get('model', weights)

            def get_w(w, key):
                return {k.split(key + '.')[1]: v for k, v in w.items() if key + '.' in k}

            if hasattr(self, 'mm_projector') and any('mm_projector.' in k for k in model_weights):
                self.mm_projector.load_state_dict(get_w(model_weights, 'mm_projector'), strict=False)

            if (
                hasattr(self, 'latent_queries') and self.latent_queries is not None
                and any('latent_queries' in k for k in model_weights)
            ):
                lq_key = next(k for k in model_weights if 'latent_queries' in k)
                pretrained_lq = model_weights[lq_key]
                if pretrained_lq.shape == self.latent_queries.data.shape:
                    with torch.no_grad():
                        self.latent_queries.data.copy_(pretrained_lq)
                else:
                    logger.warning(
                        f"latent_queries shape 불일치: "
                        f"ckpt={pretrained_lq.shape} vs model={self.latent_queries.data.shape}. 스킵."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# ScaleRAEMetaForCausalLM
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def encode_images(self, image_aux_list):
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        features = []
        for image_aux, vt in zip(image_aux_list, vision_tower_aux_list):
            if len(image_aux.shape) == 3:
                image_aux = image_aux.unsqueeze(0)
            # vision tower는 plain list이므로 Trainer가 GPU로 이동시키지 않음
            # 입력 텐서의 device에 맞춰 lazy하게 이동
            try:
                vt_device = next(vt.parameters()).device
                if vt_device != image_aux.device:
                    vt.to(image_aux.device)
            except StopIteration:
                pass
            features.append(vt(image_aux))
        return features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images               = None,
        vision_token_indices = None,
        answer_img_mask      = None,
        reverse_vti          = None,
        answer_token_mask    = None,
        image_embeds         = None,
        images_gen           = None,
    ):
        """
        멀티모달 입력 전처리.

        [이슈 2 수정]
          pred_image_features_batched (RAE encode 결과)를
          self._oc_gt_cache에 저장.
          → _forward_oc_train에서 꺼내 gt_image_features로 사용.

        [이슈 3 수정]
          OC 모드에서 slot 토큰 추가 전에 text 부분을 미리 clip.
          전체 시퀀스 길이 ≤ tokenizer_model_max_length 보장.

        [이슈 4 사전 처리]
          images_per_batch를 self._oc_images_per_batch에 저장.
          → _forward_oc_train/decode에서 정확한 offset 계산에 사용.
        """
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()

        if (
            vision_tower_aux_list is None
            or input_ids.shape[1] == 1
            or (images is None and image_embeds is None)
        ):
            return (
                input_ids, position_ids, attention_mask,
                past_key_values, None, labels,
                None, None, None, None,
            )

        # ── 이미지 인코딩 ─────────────────────────────────────────
        if image_embeds is None:
            image_aux_list          = [images]
            image_aux_list          = [
                img.flatten(0, 1) if img.dim() == 5 else img
                for img in image_aux_list
            ]
            image_aux_features_list = self.encode_images(image_aux_list)
            image_features          = image_aux_features_list[0]   # (B*M, 256, 1152)
            pred_image_features     = image_features.clone().detach()
        else:
            image_features      = image_embeds
            pred_image_features = image_features.clone().detach()

        # ── mm_projector ─────────────────────────────────────────
        dtype          = image_features.dtype
        image_features = self.get_model().mm_projector(image_features).to(dtype)

        # ── 배치/이미지 차원 복원 ─────────────────────────────────
        batch_size       = input_ids.shape[0]
        total_images     = image_features.shape[0]
        images_per_batch = total_images // batch_size if batch_size > 0 else 1

        tokens_per_image  = image_features.shape[1]       # 256
        feature_dim       = image_features.shape[2]       # D
        img_feature_dim   = pred_image_features.shape[2]  # 1152

        image_features_batched      = image_features.view(
            batch_size, images_per_batch * tokens_per_image, feature_dim
        )
        pred_image_features_batched = pred_image_features.view(
            batch_size, images_per_batch * tokens_per_image, img_feature_dim
        )

        # ── [이슈 2] gt_image_features 캐싱 ──────────────────────
        # OC 모드에서 _forward_oc_train이 꺼내 쓸 수 있도록 저장
        # detach(): gradient가 vision tower로 흐르지 않도록 (GT는 고정)
        use_oc = getattr(self.get_model().config, 'use_object_centric', False)
        if use_oc:
            self._oc_gt_cache = pred_image_features_batched.detach()
            # [이슈 4] images_per_batch 저장
            self._oc_images_per_batch = images_per_batch

        # ── 텍스트 임베딩 ─────────────────────────────────────────
        new_input_ids_for_emb = torch.where(input_ids == IMAGE_TOKEN_INDEX, 0, input_ids)
        input_embeds = self.get_model().embed_tokens(new_input_ids_for_emb)
        if not self.get_model().embed_tokens.weight.requires_grad:
            input_embeds = input_embeds.clone()

        # ── 이미지 임베딩 삽입 ────────────────────────────────────
        input_embeds = apply_custom_kernel(
            input_embeds, image_features_batched, vision_token_indices
        )

        # ── selected_features ────────────────────────────────────
        seq_len = input_ids.shape[1]
        zero_selected = torch.zeros(
            (batch_size, seq_len, img_feature_dim),
            dtype=pred_image_features.dtype,
            device=input_ids.device,
        ).clone()
        selected_features = apply_custom_kernel(
            zero_selected, pred_image_features_batched, vision_token_indices
        )

        # ── latent query 삽입 (query 모드) ───────────────────────
        vision_loss_mode = getattr(self.get_model().config, 'vision_loss_mode', 'causal')
        use_query_mode   = vision_loss_mode in ("query", "half-query", "query-block")

        if use_query_mode and not use_oc:
            latent_queries = self.get_model().latent_queries
            if latent_queries is not None:
                expanded_lq = latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
                expanded_lq = expanded_lq.repeat(1, images_per_batch, 1)

                full_input_embed_mask = (input_ids == IMAGE_TOKEN_INDEX).int()
                full_is_start = (input_ids == getattr(self, 'im_start_id', -1))
                full_is_end   = (input_ids == getattr(self, 'im_end_id',   -1))
                full_is_ans   = (
                    (labels != IGNORE_INDEX) if labels is not None
                    else torch.zeros_like(input_ids, dtype=torch.bool)
                )

                full_start_in_ans = full_is_start & full_is_ans
                full_end_in_ans   = full_is_end   & full_is_ans

                full_region_markers = torch.zeros_like(full_input_embed_mask)
                full_region_markers = torch.where(
                    full_start_in_ans,  torch.ones_like(full_region_markers), full_region_markers
                )
                full_region_markers = torch.where(
                    full_end_in_ans,   -torch.ones_like(full_region_markers), full_region_markers
                )
                full_answer_regions = full_region_markers.cumsum(dim=1) > 0

                answer_image_mask = (
                    full_input_embed_mask.bool() & full_answer_regions
                ).float()

                zero_latent     = torch.zeros_like(input_embeds)
                latent_embedded = apply_custom_kernel(
                    zero_latent, expanded_lq, vision_token_indices
                )
                input_embeds = input_embeds + answer_image_mask.unsqueeze(-1) * (
                    latent_embedded - input_embeds
                )

        # ── [이슈 3 + OC] slot 토큰 추가 ─────────────────────────
        attention_bias = None

        if use_oc:
            oc_max_slots   = getattr(self.get_model().config, 'oc_max_slots', 10)
            slot_token_emb = getattr(self, 'slot_token_emb', None)

            if slot_token_emb is not None:
                # ── OC: latent_query 직접 concat (answer_image_mask 우회) ──
                n_img   = tokens_per_image * images_per_batch
                n_query = 0
                if use_query_mode:
                    lq = self.get_model().latent_queries
                    if lq is not None:
                        lq_expanded = lq.unsqueeze(0).expand(batch_size, -1, -1)
                        lq_expanded = lq_expanded.repeat(1, images_per_batch, 1)
                        input_embeds = torch.cat([input_embeds, lq_expanded], dim=1)
                        n_lq = lq_expanded.shape[1]
                        if attention_mask is not None:
                            lq_attn = torch.ones(
                                batch_size, n_lq,
                                device=input_embeds.device,
                                dtype=attention_mask.dtype,
                            )
                            attention_mask = torch.cat(
                                [attention_mask, lq_attn], dim=1
                            )
                        if labels is not None:
                            lq_labels = torch.full(
                                (batch_size, n_lq), IGNORE_INDEX,
                                dtype=labels.dtype, device=labels.device,
                            )
                            labels = torch.cat([labels, lq_labels], dim=1)
                        n_query = n_lq

                n_prefix = n_img + n_query
                slot_tokens = slot_token_emb.expand(batch_size, oc_max_slots, -1)

                img_query_part = input_embeds[:, :n_prefix, :]
                text_part      = input_embeds[:, n_prefix:, :]

                # ── [이슈 3] text_part를 미리 clip ────────────────
                # 전체 길이 = n_prefix + n_slots + text_len ≤ max_total_len
                max_total_len = getattr(
                    self.get_model().config, 'tokenizer_model_max_length', 2048
                )
                max_text_len = max_total_len - n_prefix - oc_max_slots

                if max_text_len <= 0:
                    # img+query만으로 이미 max_length를 초과하는 극단적 상황
                    # slot만 붙이고 text는 버림
                    logger.warning(
                        f"[OC] n_prefix({n_prefix}) + oc_max_slots({oc_max_slots}) "
                        f">= max_total_len({max_total_len}). text를 전부 제거합니다."
                    )
                    text_part = text_part[:, :0, :]
                    if labels is not None:
                        labels = labels[:, :n_prefix]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :n_prefix]
                elif text_part.shape[1] > max_text_len:
                    # 안전하게 clip
                    text_part = text_part[:, :max_text_len, :]
                    if labels is not None:
                        lbl_prefix = labels[:, :n_prefix]
                        lbl_text   = labels[:, n_prefix: n_prefix + max_text_len]
                        labels     = torch.cat([lbl_prefix, lbl_text], dim=1)
                    if attention_mask is not None:
                        attn_prefix = attention_mask[:, :n_prefix]
                        attn_text   = attention_mask[:, n_prefix: n_prefix + max_text_len]
                        attention_mask = torch.cat([attn_prefix, attn_text], dim=1)

                # slot 삽입: [img+query | slots | text]
                input_embeds = torch.cat([img_query_part, slot_tokens, text_part], dim=1)

                # attention_mask 확장 (slot 위치 = 1)
                if attention_mask is not None:
                    slot_attn      = torch.ones(
                        batch_size, oc_max_slots,
                        device=input_embeds.device,
                        dtype=attention_mask.dtype,
                    )
                    attn_prefix    = attention_mask[:, :n_prefix]
                    attn_text_part = attention_mask[:, n_prefix:]
                    attention_mask = torch.cat(
                        [attn_prefix, slot_attn, attn_text_part], dim=1
                    )

                # labels 확장 (slot 위치 = IGNORE_INDEX)
                if labels is not None:
                    lbl_prefix    = labels[:, :n_prefix]
                    lbl_text_part = labels[:, n_prefix:]
                    slot_labels   = torch.full(
                        (batch_size, oc_max_slots), IGNORE_INDEX,
                        dtype=labels.dtype, device=labels.device,
                    )
                    labels = torch.cat([lbl_prefix, slot_labels, lbl_text_part], dim=1)

                # 완전한 4D attention mask 생성: OC 패턴 + 표준 causal + 패딩
                from .object_centric.slot_generator import build_oc_attention_mask
                oc_bias = build_oc_attention_mask(
                    n_img=n_img,
                    n_query=n_query,
                    n_slots=oc_max_slots,
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )  # (L_oc, L_oc)

                L_total   = input_embeds.shape[1]
                L_oc      = n_img + n_query + oc_max_slots

                min_dtype = torch.finfo(input_embeds.dtype).min
                causal_mask = torch.triu(
                    torch.full(
                        (L_total, L_total), min_dtype,
                        device=input_embeds.device, dtype=input_embeds.dtype,
                    ),
                    diagonal=1,
                )
                causal_mask[:L_oc, :L_oc] = oc_bias

                # Qwen2/transformers는 (batch, 1, seq, seq) 형식을 요구함
                # num_heads로 expand하면 안 됨 - 1로 유지해 broadcast
                attention_bias = (
                    causal_mask
                    .unsqueeze(0).unsqueeze(0)
                    .expand(batch_size, 1, -1, -1)
                )

                if attention_mask is not None:
                    padding_mask = (
                        (1.0 - attention_mask[:, None, None, :].to(
                            dtype=input_embeds.dtype))
                        * min_dtype
                    )
                    attention_bias = attention_bias + padding_mask

        # ── 최종 길이 정렬 ────────────────────────────────────────
        if labels is not None:
            max_len = labels.shape[1]
            if input_embeds.shape[1] > max_len:
                input_embeds   = input_embeds[:, :max_len]
                attention_mask = attention_mask[:, :max_len] if attention_mask is not None else None
                if attention_bias is not None:
                    attention_bias = attention_bias[:, :, :max_len, :max_len]
            elif input_embeds.shape[1] < max_len:
                pad_len      = max_len - input_embeds.shape[1]
                pad_embed    = torch.zeros(
                    batch_size, pad_len, feature_dim,
                    dtype=input_embeds.dtype, device=input_embeds.device,
                )
                input_embeds = torch.cat([input_embeds, pad_embed], dim=1)

        # input_embed_mask
        input_embed_mask = (input_ids == IMAGE_TOKEN_INDEX).int()
        if labels is not None:
            is_start     = (input_ids == getattr(self, 'im_start_id', -1))
            is_end       = (input_ids == getattr(self, 'im_end_id',   -1))
            is_answer    = (
                (labels[:, :input_ids.shape[1]] != IGNORE_INDEX)
                if labels is not None
                else torch.zeros_like(input_ids, dtype=torch.bool)
            )
            start_in_ans = is_start & is_answer
            end_in_ans   = is_end   & is_answer
            region_mrk   = torch.zeros_like(input_embed_mask)
            region_mrk   = torch.where(start_in_ans,  torch.ones_like(region_mrk),  region_mrk)
            region_mrk   = torch.where(end_in_ans,    -torch.ones_like(region_mrk), region_mrk)
            answer_rgn   = region_mrk.cumsum(dim=1) > 0
            input_embed_mask = (input_embed_mask.bool() & answer_rgn).int()

        selected_features = (
            selected_features[:, :labels.shape[1]]
            if labels is not None and selected_features is not None
            else selected_features
        )
        input_embed_mask = (
            input_embed_mask[:, :labels.shape[1]]
            if labels is not None
            else input_embed_mask
        )

        # extra_mm 구성
        extra_mm_outputs = None
        if reverse_vti is not None and answer_img_mask is not None:
            extra_mm_outputs = (
                image_features_batched,
                reverse_vti,
                answer_img_mask,
                pred_image_features_batched,
            )

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            input_embeds,
            labels,
            selected_features,
            input_embed_mask,
            attention_bias,
            extra_mm_outputs,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_emb  = self.get_input_embeddings().weight.data
                output_emb = self.get_output_embeddings().weight.data
                in_avg     = input_emb[:-num_new_tokens].mean(dim=0, keepdim=True)
                out_avg    = output_emb[:-num_new_tokens].mean(dim=0, keepdim=True)
                input_emb[-num_new_tokens:]  = in_avg
                output_emb[-num_new_tokens:] = out_avg

            if model_args.tune_mm_mlp_adapter or getattr(
                model_args, 'tune_adapter_and_vision_head', False
            ):
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False