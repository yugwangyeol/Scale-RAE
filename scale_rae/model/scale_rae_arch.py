"""
Scale-RAE Architecture Mixin (Object-Centric 확장 포함)

GPU (CUDA) 환경 전용.
IS_XLA_AVAILABLE=False로 가정하여 SPMD/XLA 코드 제거.

Object-Centric 변경사항:
  - prepare_inputs_labels_for_multimodal에서
    [img | latent_query | <SLOT>×max_slots] 시퀀스 구성
  - OC attention mask (build_oc_attention_mask) 생성 및 전달
"""

from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ezcolorlog import root_logger as logger

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
# GPU용 커스텀 커널 (SPMD 없이 scatter/gather로 구현)
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_kernel(
    input_embeds:  torch.Tensor,
    img_embeds:    torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """
    이미지 임베딩을 텍스트 임베딩 시퀀스의 지정 위치에 삽입.

    GPU path: scatter/gather 방식 (XLA SPMD 없음).

    Args:
        input_embeds:  (B, L_text, D) - 텍스트 임베딩
        img_embeds:    (B, L_img,  D) - 이미지 패치 임베딩
        token_indices: (B, L_text) - 각 위치의 소스 인덱스
                       L_text 이내이면 텍스트, 이상이면 이미지에서 가져옴

    Returns:
        output: (B, L_text, D)
    """
    if IS_XLA_AVAILABLE:
        # XLA path (원본 SPMD 구현)
        from .reference.custom_kernel import _xla_custom_kernel
        return _xla_custom_kernel(input_embeds, img_embeds, token_indices)

    # GPU path
    B, L_text, D  = input_embeds.shape
    _, L_img,  _  = img_embeds.shape

    # 텍스트 + 이미지를 concat한 뒤 index_select
    combined = torch.cat([input_embeds, img_embeds], dim=1)  # (B, L_text + L_img, D)

    # token_indices가 L_text + L_img 범위를 벗어나면 clamp
    indices_safe = token_indices.clamp(0, L_text + L_img - 1)  # (B, L_text)

    # gather
    idx_expanded = indices_safe.unsqueeze(-1).expand(-1, -1, D)  # (B, L_text, D)
    output = combined.gather(dim=1, index=idx_expanded)           # (B, L_text, D)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# ScaleRAEMetaModel
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEMetaModel:
    """
    Scale-RAE 멀티모달 메타 모델.
    vision_tower, mm_projector, latent_queries 관리.
    """

    def __init__(self, config):
        super(ScaleRAEMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):
            projector_type = getattr(config, 'mm_projector_type', 'linear')
            self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True)
            config.mm_hidden_size      = sum(
                vt.hidden_size for vt in self.vision_tower_aux_list
            )
            self.mm_projector = build_vision_projector(config)

        # ── latent queries (query / query-block 모드) ────────────
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
        """비전 모듈 초기화 (학습 시작 시 1회 호출)."""
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
            vision_tower_aux_list = build_vision_tower_aux_list(model_args)
            self.vision_tower_aux_list = vision_tower_aux_list
        else:
            for vt in self.vision_tower_aux_list:
                vt.load_model()

        self.config.use_mm_proj             = True
        self.config.mm_projector_type       = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer  = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.config.mm_hidden_size = sum(
                vt.hidden_size for vt in self.vision_tower_aux_list
            )
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # pretrained adapter 가중치 로드
        if pretrain_mm_mlp_adapter is not None:
            mm_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(w, key):
                return {k.split(key + '.')[1]: v for k, v in w.items() if key + '.' in k}
            self.mm_projector.load_state_dict(get_w(mm_weights, 'mm_projector'), strict=True)

        if pretrain_adapter_and_vision_head is not None:
            logger.info(f"adapter+vision_head 가중치 로드: {pretrain_adapter_and_vision_head}")
            weights       = torch.load(pretrain_adapter_and_vision_head, map_location='cpu')
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
    """
    Scale-RAE Causal LM 멀티모달 추상 클래스.
    prepare_inputs_labels_for_multimodal: 이미지 임베딩 시퀀스 구성.
    OC 모드: [img | query | slot] 시퀀스 + OC attention mask.
    """

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def encode_images(self, image_aux_list):
        """이미지를 비전 타워로 인코딩."""
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        features = []
        for image_aux, vt in zip(image_aux_list, vision_tower_aux_list):
            if len(image_aux.shape) == 3:
                image_aux = image_aux.unsqueeze(0)
            features.append(vt(image_aux))
        return features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images              = None,
        vision_token_indices = None,
        answer_img_mask     = None,
        reverse_vti         = None,
        answer_token_mask   = None,
        image_embeds        = None,
        images_gen          = None,
    ):
        """
        멀티모달 입력을 MLLM이 처리할 수 있는 형태로 변환.

        Object-Centric 모드 (use_object_centric=True):
          입력 시퀀스: [img_embed×256 | latent_query×256 | slot_token×max_slots]
          OC attention mask 생성 후 attention_bias로 반환.

        표준 query 모드:
          입력 시퀀스: [img_embed×256 | latent_query×256]
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
            image_aux_list = [images]
            image_aux_list = [img.flatten(0, 1) if img.dim() == 5 else img
                              for img in image_aux_list]
            image_aux_features_list = self.encode_images(image_aux_list)
            image_features          = image_aux_features_list[0]  # (B*M, 256, 1152)
            pred_image_features     = image_features.clone().detach()
        else:
            image_features      = image_embeds
            pred_image_features = image_features.clone().detach()

        # ── mm_projector ─────────────────────────────────────────
        dtype           = image_features.dtype
        image_features  = self.get_model().mm_projector(image_features).to(dtype)  # (B*M, 256, D)

        # ── 배치/이미지 차원 복원 ─────────────────────────────────
        batch_size      = input_ids.shape[0]
        total_images    = image_features.shape[0]
        images_per_batch = total_images // batch_size if batch_size > 0 else 1

        tokens_per_image  = image_features.shape[1]      # 256
        feature_dim       = image_features.shape[2]      # D (LLM hidden size)
        img_feature_dim   = pred_image_features.shape[2] # 1152 (SigLIP dim)

        image_features_batched      = image_features.view(batch_size, images_per_batch * tokens_per_image, feature_dim)
        pred_image_features_batched = pred_image_features.view(batch_size, images_per_batch * tokens_per_image, img_feature_dim)

        # ── 텍스트 임베딩 ─────────────────────────────────────────
        new_input_ids_for_emb = torch.where(input_ids == IMAGE_TOKEN_INDEX, 0, input_ids)
        input_embeds = self.get_model().embed_tokens(new_input_ids_for_emb)
        if not self.get_model().embed_tokens.weight.requires_grad:
            input_embeds = input_embeds.clone()

        # ── 이미지 임베딩 삽입 ────────────────────────────────────
        input_embeds = apply_custom_kernel(
            input_embeds, image_features_batched, vision_token_indices
        )

        # ── selected_features (GT vision features) ───────────────
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

        if use_query_mode:
            latent_queries = self.get_model().latent_queries
            if latent_queries is not None:
                expanded_lq = latent_queries.unsqueeze(0).expand(batch_size, -1, -1)   # (B, 256, D)
                expanded_lq = expanded_lq.repeat(1, images_per_batch, 1)               # (B, 256*M, D)

                # answer image mask 기반으로 query 위치 결정
                full_input_embed_mask = (input_ids == IMAGE_TOKEN_INDEX).int()
                full_is_start = (input_ids == getattr(self, 'im_start_id', -1))
                full_is_end   = (input_ids == getattr(self, 'im_end_id',   -1))
                full_is_ans   = (labels != IGNORE_INDEX) if labels is not None else torch.zeros_like(input_ids, dtype=torch.bool)

                full_start_in_ans = full_is_start & full_is_ans
                full_end_in_ans   = full_is_end   & full_is_ans

                full_region_markers = torch.zeros_like(full_input_embed_mask)
                full_region_markers = torch.where(full_start_in_ans, torch.ones_like(full_region_markers),  full_region_markers)
                full_region_markers = torch.where(full_end_in_ans,  -torch.ones_like(full_region_markers),  full_region_markers)
                full_answer_regions = full_region_markers.cumsum(dim=1) > 0

                answer_image_mask = (full_input_embed_mask.bool() & full_answer_regions).float()  # (B, L)

                # query를 answer image 위치에만 삽입
                zero_latent    = torch.zeros_like(input_embeds)
                latent_embedded = apply_custom_kernel(
                    zero_latent, expanded_lq, vision_token_indices
                )
                input_embeds = input_embeds + answer_image_mask.unsqueeze(-1) * (latent_embedded - input_embeds)

        # ── Object-Centric: <SLOT> 토큰 추가 ─────────────────────
        use_oc       = getattr(self.get_model().config, 'use_object_centric', False)
        attention_bias = None

        if use_oc:
            oc_max_slots   = getattr(self.get_model().config, 'oc_max_slots', 10)
            slot_token_emb = getattr(self, 'slot_token_emb', None)

            if slot_token_emb is not None:
                # (1, 1, D) → (B, max_slots, D)
                slot_tokens = slot_token_emb.expand(batch_size, oc_max_slots, -1)

                # 시퀀스에 slot 토큰 추가: [img | query | slots | text ...]
                # 주의: input_embeds는 이미 labels[:L] 길이로 clip된 상태일 수 있음
                # slot은 시퀀스 앞부분에 삽입 (이미지 직후)
                n_img   = tokens_per_image * images_per_batch      # 256
                n_query = tokens_per_image * images_per_batch      # 256 (query mode)

                # input_embeds를 3 구간으로 분리: [img+query | text]
                img_query_part = input_embeds[:, :n_img + n_query, :]  if use_query_mode else input_embeds[:, :n_img, :]
                text_part      = input_embeds[:, n_img + n_query:, :]  if use_query_mode else input_embeds[:, n_img:, :]

                # slot 삽입: [img+query | slots | text]
                input_embeds = torch.cat([img_query_part, slot_tokens, text_part], dim=1)

                # attention_mask 확장
                if attention_mask is not None:
                    slot_attn       = torch.ones(batch_size, oc_max_slots, device=input_embeds.device, dtype=attention_mask.dtype)
                    # img+query / slot / text 순서로 재구성
                    attn_img_query  = attention_mask[:, :n_img + n_query] if use_query_mode else attention_mask[:, :n_img]
                    attn_text       = attention_mask[:, n_img + n_query:] if use_query_mode else attention_mask[:, n_img:]
                    attention_mask  = torch.cat([attn_img_query, slot_attn, attn_text], dim=1)

                # labels 확장 (slot 위치는 IGNORE_INDEX)
                if labels is not None:
                    lbl_img_query = labels[:, :n_img + n_query] if use_query_mode else labels[:, :n_img]
                    lbl_text      = labels[:, n_img + n_query:] if use_query_mode else labels[:, n_img:]
                    slot_labels   = torch.full(
                        (batch_size, oc_max_slots), IGNORE_INDEX,
                        dtype=labels.dtype, device=labels.device,
                    )
                    labels = torch.cat([lbl_img_query, slot_labels, lbl_text], dim=1)

                # input_embed_mask 확장 (slot 위치는 0 - vision 위치 아님)
                # OC Attention Bias 생성
                from .object_centric.slot_generator import build_oc_attention_mask
                n_img_oc  = n_img
                n_qry_oc  = n_query if use_query_mode else 0
                oc_bias   = build_oc_attention_mask(
                    n_img=n_img_oc,
                    n_query=n_qry_oc,
                    n_slots=oc_max_slots,
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )  # (L_oc, L_oc) where L_oc = n_img + n_query + n_slots

                # (L_oc, L_oc) → (B, num_heads, L_total, L_total) 적용
                num_heads = getattr(self.get_model().config, 'num_attention_heads', 12)
                L_total   = input_embeds.shape[1]
                L_oc      = n_img_oc + n_qry_oc + oc_max_slots

                # OC 영역(L_oc)만 bias 적용, 나머지는 0
                full_bias = torch.zeros(L_total, L_total, device=input_embeds.device, dtype=input_embeds.dtype)
                full_bias[:L_oc, :L_oc] = oc_bias

                attention_bias = full_bias.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        # ── 시퀀스 길이 정렬 ─────────────────────────────────────
        if labels is not None:
            max_len = labels.shape[1]
            if input_embeds.shape[1] > max_len:
                input_embeds   = input_embeds[:, :max_len]
                attention_mask = attention_mask[:, :max_len] if attention_mask is not None else None
            elif input_embeds.shape[1] < max_len:
                # pad (unlikely but safe)
                pad_len      = max_len - input_embeds.shape[1]
                pad_embed    = torch.zeros(batch_size, pad_len, feature_dim, dtype=input_embeds.dtype, device=input_embeds.device)
                input_embeds = torch.cat([input_embeds, pad_embed], dim=1)

        # input_embed_mask: IMAGE_TOKEN 위치 AND answer region
        input_embed_mask = (input_ids == IMAGE_TOKEN_INDEX).int()
        if labels is not None:
            is_start     = (input_ids == getattr(self, 'im_start_id', -1))
            is_end       = (input_ids == getattr(self, 'im_end_id',   -1))
            is_answer    = (labels[:, :input_ids.shape[1]] != IGNORE_INDEX) if labels is not None else torch.zeros_like(input_ids, dtype=torch.bool)
            start_in_ans = is_start & is_answer
            end_in_ans   = is_end   & is_answer
            region_mrk   = torch.zeros_like(input_embed_mask)
            region_mrk   = torch.where(start_in_ans, torch.ones_like(region_mrk),  region_mrk)
            region_mrk   = torch.where(end_in_ans,  -torch.ones_like(region_mrk),  region_mrk)
            answer_rgn   = region_mrk.cumsum(dim=1) > 0
            input_embed_mask = (input_embed_mask.bool() & answer_rgn).int()

        selected_features = selected_features[:, :labels.shape[1]] if labels is not None and selected_features is not None else selected_features
        input_embed_mask  = input_embed_mask[:, :labels.shape[1]]  if labels is not None else input_embed_mask

        # extra_mm 구성 (query 모드에서 vision loss 계산용)
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
        """Special token 추가 및 embedding resize."""
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

            if model_args.tune_mm_mlp_adapter or getattr(model_args, 'tune_adapter_and_vision_head', False):
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False