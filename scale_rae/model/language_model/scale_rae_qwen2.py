"""
Scale-RAE Qwen2 Language Model with Object-Centric Extension

이슈 2 수정:
  - _forward_oc_train에서 gt_image_features가 None이면
    self._oc_gt_cache에서 자동으로 가져옴
    (scale_rae_arch.py의 prepare_inputs_labels_for_multimodal에서 캐싱)
  - 캐시 사용 후 반드시 del로 정리 (메모리 누수 방지)

이슈 4 수정:
  - n_img, n_query를 self.num_image_tokens 하드코딩 대신
    self._oc_images_per_batch를 참조해서 계산
  - multi-image 시에도 올바른 offset 보장
"""

from typing import List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

from ..scale_rae_arch import ScaleRAEMetaModel, ScaleRAEMetaForCausalLM, apply_custom_kernel
from scale_rae.utils import IS_XLA_AVAILABLE
from scale_rae.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_loss(
    predictions:     torch.Tensor,
    targets:         torch.Tensor,
    valid_positions: torch.Tensor,
    loss_type:       str = 'l2',
) -> torch.Tensor:
    mask_expanded = valid_positions.unsqueeze(-1).expand_as(predictions)
    feature_dim   = predictions.size(-1)

    if loss_type == 'l1':
        diff = torch.abs(predictions - targets)
    elif loss_type == 'smooth_l1':
        diff = F.smooth_l1_loss(predictions, targets, reduction='none')
    else:
        diff = (predictions - targets) ** 2

    masked_diff    = diff * mask_expanded
    per_token_loss = masked_diff.sum(dim=-1) / feature_dim
    total_loss     = per_token_loss.sum()
    num_valid      = valid_positions.sum() + 1e-8
    return total_loss / num_valid


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEQwenConfig(Qwen2Config):
    model_type                      = "cambrian_qwen"
    vision_loss                     = "regression-loss"
    vision_loss_mode                = "causal"
    vision_tower_aux_token_len_list = [256]

    use_object_centric = False
    oc_max_slots       = 10
    oc_d_model         = 1536


# ─────────────────────────────────────────────────────────────────────────────
# Qwen2 Model
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEQwenModel(ScaleRAEMetaModel, Qwen2Model):
    config_class = ScaleRAEQwenConfig

    def __init__(self, config: Qwen2Config):
        super(ScaleRAEQwenModel, self).__init__(config)

    def forward(
        self,
        input_ids:                             torch.LongTensor = None,
        attention_mask:                        Optional[torch.Tensor] = None,
        position_ids:                          Optional[torch.LongTensor] = None,
        past_key_values:                       Optional[List[torch.FloatTensor]] = None,
        inputs_embeds:                         Optional[torch.FloatTensor] = None,
        use_cache:                             Optional[bool] = None,
        output_attentions:                     Optional[bool] = None,
        output_hidden_states:                  Optional[bool] = None,
        return_dict:                           Optional[bool] = None,
        vision_tower_aux_feature_list:         Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size:             Optional[List[tuple]] = None,
        global_context_feature:               Optional[torch.Tensor] = None,
        attention_bias:                        Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids와 inputs_embeds를 동시에 지정할 수 없습니다.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("input_ids 또는 inputs_embeds 중 하나는 지정해야 합니다.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True`는 gradient checkpointing과 호환되지 않습니다.")
            use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device      = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_bias is not None:
            attention_mask = attention_bias
        elif self._attn_implementation == "flash_attention_2":
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length),
                inputs_embeds, past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length),
                inputs_embeds, past_key_values_length,
                sliding_window=getattr(self.config, 'sliding_window', None),
            )

        hidden_states      = inputs_embeds
        all_hidden_states  = () if output_hidden_states else None
        all_self_attns     = () if output_attentions    else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, attention_mask, position_ids,
                    past_key_values, output_attentions, use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

from ..diffusion_loss.diffloss import create_rf_projector


class ScaleRAEQwenForCausalLM(Qwen2ForCausalLM, ScaleRAEMetaForCausalLM):

    config_class = ScaleRAEQwenConfig

    def __init__(self, config: ScaleRAEQwenConfig):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type   = "cambrian_qwen"
        config.rope_scaling = None

        self.model   = ScaleRAEQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vision_loss      = getattr(config, 'vision_loss',      'diffusion-loss')
        self.vision_loss_mode = getattr(config, 'vision_loss_mode', 'query')
        self.vision_coef      = getattr(config, 'vision_coef',      1.0)
        self.vision_tower_aux_token_len_list = getattr(
            config, 'vision_tower_aux_token_len_list', [256]
        )
        self.diffusion_model_channels = getattr(config, 'diffusion_model_channels', 1152)
        self.num_image_tokens         = 256

        if self.vision_loss in ('diffusion-loss', 'ddt-loss'):
            vision_loss_mode_cfg = getattr(config, 'vision_loss_mode', 'query')

            if vision_loss_mode_cfg in ('query', 'query-block'):
                self.diff_head_config = {
                    "diffusion_tokens":   self.vision_tower_aux_token_len_list[0],
                    "diffusion_channels": self.diffusion_model_channels,
                    "z_channels":         config.hidden_size,
                    "model_hidden_size":  getattr(config, 'diffusion_model_hidden_size', 1152),
                    "model_depth":        getattr(config, 'diffusion_model_depth',       12),
                    "model_heads":        getattr(config, 'diffusion_model_heads',       16),
                    "guidance_scale":     1.0,
                    "use_mlp":            False,
                    "dit_cls":            getattr(config, 'dit_cls', 'DiT'),
                }
                if getattr(config, 'diffusion_model_z_channels', 0) != 0:
                    self.diff_head_projector     = nn.Linear(
                        config.hidden_size, config.diffusion_model_z_channels
                    )
                    self.use_diff_head_projector = True
                    self.diff_head_config["z_channels"] = config.diffusion_model_z_channels
                else:
                    self.use_diff_head_projector = False
            else:
                self.diff_head_config = {
                    "diffusion_tokens":   1,
                    "diffusion_channels": self.diffusion_model_channels,
                    "z_channels":         config.hidden_size,
                    "model_hidden_size":  getattr(config, 'diffusion_model_hidden_size', 1152),
                    "model_depth":        getattr(config, 'diffusion_model_depth',       12),
                    "model_heads":        getattr(config, 'diffusion_model_heads',       16),
                    "guidance_scale":     2.0,
                    "use_mlp":            True,
                    "dit_cls":            getattr(config, 'dit_cls', 'DiT'),
                }
                self.use_diff_head_projector = False

            self.diff_head = create_rf_projector(self.diff_head_config)

        elif self.vision_loss == 'regression-loss':
            self.vision_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1152),
            )

        if getattr(config, 'use_object_centric', False):
            self._init_object_centric_modules(config)

        self.post_init()

    # ─── OC 초기화 ───────────────────────────────────────────────

    def _init_object_centric_modules(self, config: ScaleRAEQwenConfig) -> None:
        from ..object_centric.slot_generator import SlotEOSDetector, ObjectTokenAggregator

        vision_loss_mode = getattr(config, 'vision_loss_mode', 'query')
        if vision_loss_mode not in ('query', 'half-query', 'query-block'):
            raise ValueError(
                f"[OC] Object-Centric 모드는 query vision_loss_mode가 필요합니다. "
                f"현재: '{vision_loss_mode}'"
            )

        oc_d_model   = getattr(config, 'oc_d_model',   config.hidden_size)
        oc_max_slots = getattr(config, 'oc_max_slots', 10)
        config.oc_d_model = oc_d_model

        self.slot_token_emb = nn.Parameter(torch.randn(1, 1, oc_d_model) * 0.02)
        self.slot_eos_detector = SlotEOSDetector(d_model=oc_d_model)

        num_heads = 16 if oc_d_model >= 3584 else 8
        self.slot_aggregator = ObjectTokenAggregator(
            d_model=oc_d_model, num_heads=num_heads
        )
        self.null_slot_token = nn.Parameter(torch.zeros(1, 1, oc_d_model))

        self.oc_max_slots = oc_max_slots
        self.oc_d_model   = oc_d_model

        logger.info(
            f"[OC] 모듈 초기화: d_model={oc_d_model}, "
            f"max_slots={oc_max_slots}, num_heads={num_heads}"
        )

    # ─── 유틸리티 ────────────────────────────────────────────────

    def get_model(self):
        return self.model

    def is_object_centric(self) -> bool:
        return getattr(self, 'slot_aggregator', None) is not None

    def _get_oc_offsets(self) -> Tuple[int, int, int]:
        """
        [이슈 4 수정]
        OC 시퀀스에서 각 영역의 offset을 반환.
        images_per_batch와 vision_loss_mode를 고려해서 계산.

        Returns:
            (n_img, n_query, n_slots)
            - n_img:   이미지 패치 토큰 수 (= num_image_tokens * images_per_batch)
            - n_query: latent query 토큰 수 (query mode일 때만 > 0)
            - n_slots: max slot 수
        """
        images_per_batch = getattr(self, '_oc_images_per_batch', 1)
        n_img   = self.num_image_tokens * images_per_batch

        use_query_mode = self.vision_loss_mode in ('query', 'half-query', 'query-block')
        n_query = self.num_image_tokens * images_per_batch if use_query_mode else 0

        n_slots = self.oc_max_slots
        return n_img, n_query, n_slots

    def load_vision_head(self, model_args) -> None:
        pretrain_path = getattr(model_args, 'pretrain_adapter_and_vision_head', None)
        if pretrain_path is None:
            return
        logger.info(f"[VisionHead] {pretrain_path} 에서 가중치 로드")
        if os.path.isdir(pretrain_path):
            # safetensors 디렉토리 (HuggingFace 체크포인트) → from_pretrained가 이미 로드함
            import json as _json
            index_path  = os.path.join(pretrain_path, 'model.safetensors.index.json')
            single_path = os.path.join(pretrain_path, 'model.safetensors')
            if os.path.isfile(index_path):
                from safetensors.torch import load_file as _st_load
                with open(index_path) as f:
                    index = _json.load(f)
                shard_files = sorted(set(index['weight_map'].values()))
                model_weights = {}
                for shard in shard_files:
                    model_weights.update(_st_load(os.path.join(pretrain_path, shard), device='cpu'))
            elif os.path.isfile(single_path):
                from safetensors.torch import load_file as _st_load
                model_weights = _st_load(single_path, device='cpu')
            else:
                logger.warning(f"[VisionHead] 디렉토리에서 가중치를 찾을 수 없음: {pretrain_path}")
                return
        else:
            weights       = torch.load(pretrain_path, map_location='cpu')
            model_weights = weights.get('model', weights)

        def get_w(w, key):
            return {k.split(key + '.')[1]: v for k, v in w.items() if key + '.' in k}

        if hasattr(self, 'vision_head') and any('vision_head.' in k for k in model_weights):
            self.vision_head.load_state_dict(get_w(model_weights, 'vision_head'), strict=False)
        if hasattr(self, 'diff_head') and any('diff_head.' in k for k in model_weights):
            self.diff_head.load_state_dict(get_w(model_weights, 'diff_head'), strict=False)

    # ─── OC Loss ────────────────────────────────────────────────

    def compute_oc_loss(
        self,
        context:           torch.Tensor,
        gt_image_features: torch.Tensor,
        eos_sim:           torch.Tensor,
        valid_slots:       torch.Tensor,
        K_gt:              torch.Tensor,
        answer_img_mask:   Optional[torch.Tensor] = None,
    ) -> dict:
        """
        OC 3종 Loss:
          total = 1.0 * fm_loss + 0.1 * eos_loss + 0.05 * div_loss
        """
        B         = context.shape[0]
        max_slots = eos_sim.shape[1]
        device    = context.device

        # 1. Flow Matching Loss
        if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
            context_proj = self.diff_head_projector(context)
        else:
            context_proj = context

        fm_loss_vec = self.diff_head.training_loss(z=context_proj, x=gt_image_features)

        if answer_img_mask is not None:
            mask    = answer_img_mask[:, 0].float()
            fm_loss = (fm_loss_vec * mask).sum() / (mask.sum() + 1e-8)
        else:
            fm_loss = fm_loss_vec.mean() if fm_loss_vec.dim() > 0 else fm_loss_vec

        # 2. EOS Loss — k번째 이후 슬롯은 모두 EOS
        eos_gt = torch.zeros(B, max_slots, device=device)
        for b in range(B):
            k = min(int(K_gt[b].item()), max_slots)
            eos_gt[b, k:] = 1.0

        eos_loss = F.binary_cross_entropy_with_logits(
            eos_sim * 5.0, eos_gt, reduction='mean'
        )

        # 3. Slot Diversity Loss — 유효 slot 쌍만 계산
        slots_norm = F.normalize(valid_slots, dim=-1)
        sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        eye        = torch.eye(max_slots, device=device).unsqueeze(0)

        valid_flag = torch.zeros(B, max_slots, device=device, dtype=torch.bool)
        for b in range(B):
            k = min(int(K_gt[b].item()), max_slots)
            valid_flag[b, :k] = True
        pair_valid = valid_flag.unsqueeze(2) & valid_flag.unsqueeze(1)
        pair_valid = pair_valid & ~eye.bool()

        off_diag   = sim_matrix * pair_valid.float()
        n_valid    = pair_valid.sum().clamp(min=1e-8)
        div_loss   = off_diag.sum() / n_valid

        total_loss = 1.0 * fm_loss + 0.1 * eos_loss + 0.05 * div_loss

        return {
            'fm_loss':    fm_loss,
            'eos_loss':   eos_loss,
            'div_loss':   div_loss,
            'total_loss': total_loss,
        }

    # ─── Forward ────────────────────────────────────────────────

    def forward(
        self,
        input_ids:            torch.LongTensor = None,
        attention_mask:       Optional[torch.Tensor] = None,
        position_ids:         Optional[torch.LongTensor] = None,
        past_key_values:      Optional[List[torch.FloatTensor]] = None,
        inputs_embeds:        Optional[torch.FloatTensor] = None,
        labels:               Optional[torch.LongTensor] = None,
        use_cache:            Optional[bool] = None,
        output_attentions:    Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images:               Optional[torch.FloatTensor] = None,
        return_dict:          Optional[bool] = None,
        vision_token_indices: Optional[torch.Tensor] = None,
        decoding:             Optional[bool] = False,
        answer_img_mask:      Optional[torch.Tensor] = None,
        reverse_vti:          Optional[torch.Tensor] = None,
        answer_token_mask:    Optional[torch.Tensor] = None,
        guidance_level:       Optional[float] = 1.0,
        images_gen:           Optional[torch.FloatTensor] = None,
        n_objects:            Optional[torch.Tensor] = None,
        gt_image_features:    Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        selected_features = input_embed_mask = attention_bias = extra_mm = None

        if inputs_embeds is None and images is not None:
            (
                input_ids, position_ids, attention_mask,
                past_key_values, inputs_embeds, labels,
                selected_features, input_embed_mask,
                attention_bias, extra_mm,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                images,
                vision_token_indices=vision_token_indices,
                answer_img_mask=answer_img_mask,
                reverse_vti=reverse_vti,
                answer_token_mask=answer_token_mask,
                images_gen=images_gen,
            )

        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_bias=attention_bias,
        )

        hidden_states = outputs[0]
        logits        = self.lm_head(hidden_states).float()

        if self.is_object_centric() and not decoding:
            return self._forward_oc_train(
                hidden_states, logits, labels, outputs, return_dict,
                n_objects=n_objects,
                gt_image_features=gt_image_features,
                answer_img_mask=answer_img_mask,
            )

        if self.is_object_centric() and decoding:
            pred_z = self._forward_oc_decode(hidden_states, guidance_level)
            return CausalLMOutputWithPast(
                loss=pred_z, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states, attentions=outputs.attentions,
            )

        return self._forward_standard(
            hidden_states, logits, labels, outputs,
            selected_features, input_embed_mask,
            extra_mm, return_dict, decoding,
            answer_img_mask, answer_token_mask, guidance_level,
        )

    # ─── OC 학습 forward ────────────────────────────────────────

    def _forward_oc_train(
        self,
        hidden_states:     torch.Tensor,
        logits:            torch.Tensor,
        labels:            Optional[torch.Tensor],
        outputs,
        return_dict:       bool,
        n_objects:         Optional[torch.Tensor],
        gt_image_features: Optional[torch.Tensor],
        answer_img_mask:   Optional[torch.Tensor],
    ) -> CausalLMOutputWithPast:

        # ── [이슈 4 수정] 정확한 offset 계산 ─────────────────────
        n_img, n_query, n_slots = self._get_oc_offsets()

        # ── [이슈 2 수정] gt_image_features 자동 확보 ────────────
        if gt_image_features is None:
            gt_image_features = getattr(self, '_oc_gt_cache', None)
            if gt_image_features is None:
                logger.warning(
                    "[OC] gt_image_features를 찾을 수 없습니다. "
                    "OC loss를 계산하지 않습니다. "
                    "scale_rae_arch.py의 prepare_inputs_labels_for_multimodal가 "
                    "올바르게 호출됐는지 확인하세요."
                )
            else:
                logger.debug("[OC] _oc_gt_cache에서 gt_image_features 로드.")

        # 캐시 즉시 정리 (메모리 누수 방지)
        if hasattr(self, '_oc_gt_cache'):
            del self._oc_gt_cache

        # ── hidden states에서 각 영역 추출 ───────────────────────
        base_context = hidden_states[:, n_img: n_img + n_query, :]
        raw_slots    = hidden_states[:, n_img + n_query: n_img + n_query + n_slots, :]

        # ── EOS 처리 ─────────────────────────────────────────────
        valid_slots, slot_mask, eos_sim = self.slot_eos_detector(raw_slots)

        # ── Aggregation ──────────────────────────────────────────
        context = self.slot_aggregator(base_context, valid_slots, slot_mask)

        # ── 언어 손실 (text가 있는 경우만) ────────────────────────
        loss = torch.tensor(0.0, device=hidden_states.device)
        self.loss_language = loss  # 로깅용 초기화

        if labels is not None:
            text_start   = n_img + n_query + n_slots
            shift_logits = logits[:, text_start - 1: -1, :].contiguous()
            shift_labels = labels[:, text_start:].contiguous()

            # text 부분에 실제 학습 label이 있을 때만 계산
            valid_text = (shift_labels != IGNORE_INDEX).any()
            if valid_text:
                loss_fct     = CrossEntropyLoss()
                lm_loss      = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1).to(shift_logits.device),
                )
                loss               = lm_loss
                self.loss_language = lm_loss

        # ── OC 손실 ──────────────────────────────────────────────
        if gt_image_features is not None and n_objects is not None:
            # 첫 forward에서 텐서 통계 출력 (디버그)
            if not getattr(self, '_oc_debug_logged', False):
                self._oc_debug_logged = True
                logger.info(
                    f"[OC DEBUG] gt_image_features: shape={gt_image_features.shape}, "
                    f"mean={gt_image_features.mean():.4f}, std={gt_image_features.std():.4f}, "
                    f"absmax={gt_image_features.abs().max():.4f}"
                )
                logger.info(
                    f"[OC DEBUG] context: shape={context.shape}, "
                    f"mean={context.mean():.4f}, std={context.std():.4f}"
                )
                logger.info(
                    f"[OC DEBUG] eos_sim: shape={eos_sim.shape}, "
                    f"mean={eos_sim.mean():.4f}, std={eos_sim.std():.4f}"
                )
                logger.info(f"[OC DEBUG] n_objects: {n_objects.tolist()}")

            oc_losses = self.compute_oc_loss(
                context=context,
                gt_image_features=gt_image_features,
                eos_sim=eos_sim,
                valid_slots=valid_slots,
                K_gt=n_objects,
                answer_img_mask=answer_img_mask,
            )
            loss = loss + oc_losses['total_loss'] * self.vision_coef

            self.loss_image_diff = oc_losses['fm_loss']
            self.oc_eos_loss     = oc_losses['eos_loss']
            self.oc_div_loss     = oc_losses['div_loss']
        else:
            # OC loss 없이 forward만 실행된 경우 (gt_image_features 미확보)
            self.loss_image_diff = torch.tensor(0.0, device=hidden_states.device)
            self.oc_eos_loss     = torch.tensor(0.0, device=hidden_states.device)
            self.oc_div_loss     = torch.tensor(0.0, device=hidden_states.device)

        # 디버깅용 캐시 저장
        self._oc_outputs = {
            'base_context': base_context,
            'raw_slots':    raw_slots,
            'valid_slots':  valid_slots,
            'slot_mask':    slot_mask,
            'eos_sim':      eos_sim,
            'context':      context,
        }

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    # ─── OC 추론 decode ─────────────────────────────────────────

    def _forward_oc_decode(
        self,
        hidden_states:  torch.Tensor,
        guidance_level: float = 1.0,
    ) -> torch.Tensor:

        n_img, n_query, n_slots = self._get_oc_offsets()

        base_context = hidden_states[:, n_img: n_img + n_query, :]
        raw_slots    = hidden_states[:, n_img + n_query: n_img + n_query + n_slots, :]

        valid_slots, slot_mask, eos_sim = self.slot_eos_detector(raw_slots)
        context = self.slot_aggregator(base_context, valid_slots, slot_mask)

        if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
            context_proj = self.diff_head_projector(context)
        else:
            context_proj = context

        pred_z = self.diff_head.infer(context_proj, guidance_level=guidance_level)
        return pred_z

    # ─── 표준 forward (비-OC) ────────────────────────────────────

    def _forward_standard(
        self, hidden_states, logits, labels, outputs,
        selected_features, input_embed_mask,
        extra_mm, return_dict, decoding,
        answer_img_mask, answer_token_mask, guidance_level,
    ) -> CausalLMOutputWithPast:

        if decoding:
            vision_loss_mode      = self.vision_loss_mode
            use_query_mode        = vision_loss_mode in ("query", "half-query", "query-block")
            generated_token_length = self.num_image_tokens if use_query_mode else 1

            if self.vision_loss == 'regression-loss':
                pred_z     = hidden_states[:, -generated_token_length:, :].squeeze(1)
                pred_z     = self.vision_head(pred_z)
                prediction = self.get_model().mm_projector(pred_z)
                hidden_states[:, -generated_token_length:, :] = prediction

            elif self.vision_loss in ('diffusion-loss', 'ddt-loss'):
                pred_z = hidden_states[:, -generated_token_length:, :].squeeze(1)
                if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
                    pred_z = self.diff_head_projector(pred_z)
                pred_z = self.diff_head.infer(pred_z, guidance_level=guidance_level)
                try:
                    prediction = self.get_model().mm_projector(pred_z)
                    hidden_states[:, -generated_token_length:, :] = prediction
                except Exception:
                    pass

            return CausalLMOutputWithPast(
                loss=pred_z, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states, attentions=outputs.attentions,
            )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct     = CrossEntropyLoss()
            loss         = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )
            self.loss_language = loss

            vision_loss_mode_cfg = getattr(self.get_model().config, 'vision_loss_mode', 'causal')
            if vision_loss_mode_cfg in ("query", "query-block") and extra_mm is not None:
                img_feats_raw, reverse_vti, answer_img_mask_em, prediction_target = extra_mm
                B, T, feature_dim = prediction_target.shape
                M                 = answer_img_mask_em.size(1) if answer_img_mask_em is not None else 1
                tokens_per_image  = T // M
                hidden_dim        = hidden_states.size(-1)

                hs_full    = hidden_states
                zeros_left = torch.zeros(B, T, hidden_dim, dtype=hs_full.dtype, device=hs_full.device)
                patch_hs   = apply_custom_kernel(zeros_left, hs_full, reverse_vti)

                if self.vision_loss in ('diffusion-loss', 'ddt-loss'):
                    patch_hs_r = patch_hs.view(B, M, tokens_per_image, hidden_dim).view(
                        B * M, tokens_per_image, hidden_dim
                    )
                    if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
                        patch_hs_r = self.diff_head_projector(patch_hs_r)
                    pred_r = prediction_target.view(B, M, tokens_per_image, feature_dim).view(
                        B * M, tokens_per_image, feature_dim
                    )
                    diff_loss_vec = self.diff_head.training_loss(z=patch_hs_r, x=pred_r)
                    diff_loss_mat = diff_loss_vec.view(B, M)

                    if answer_img_mask_em is not None:
                        masked    = diff_loss_mat * answer_img_mask_em.float()
                        mean_diff = masked.sum() / (answer_img_mask_em.sum() + 1e-8)
                    else:
                        mean_diff = diff_loss_mat.mean()

                    loss                = loss + mean_diff * self.vision_coef
                    self.loss_image_diff = mean_diff

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states, attentions=outputs.attentions,
        )

    # ─── OC 전용 Greedy Decode ───────────────────────────────────

    @torch.no_grad()
    def greedy_decode_oc(
        self,
        inputs_embeds:  torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids:   Optional[torch.LongTensor] = None,
        guidance_level: float = 1.0,
        return_slots:   bool  = False,
    ) -> dict:
        """
        OC 추론. inputs_embeds에는 prepare_inputs_labels_for_multimodal에서
        이미 slot 토큰이 삽입되어 있으므로 여기서 다시 추가하지 않는다.
        """

        B = inputs_embeds.shape[0]
        n_img, n_query, n_slots = self._get_oc_offsets()

        outputs       = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            position_ids=position_ids,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        base_context = hidden_states[:, n_img: n_img + n_query, :]
        raw_slots    = hidden_states[:, n_img + n_query: n_img + n_query + n_slots, :]

        self.eval()
        valid_slots, slot_mask, eos_sim = self.slot_eos_detector(raw_slots)
        context = self.slot_aggregator(base_context, valid_slots, slot_mask)

        if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
            context_proj = self.diff_head_projector(context)
        else:
            context_proj = context

        pred_image = self.diff_head.infer(context_proj, guidance_level=guidance_level)

        result = {'pred_image': pred_image, 'generated_ids': []}
        if return_slots:
            result['slots'] = {
                'base_context': base_context,
                'valid_slots':  valid_slots,
                'eos_sim':      eos_sim,
                'slot_mask':    slot_mask,
            }
        return result

    # ─── Generate ───────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        inputs:               Optional[torch.Tensor] = None,
        images:               Optional[torch.Tensor] = None,
        image_embeds:         Optional[torch.Tensor] = None,
        use_customize_greedy: Optional[bool]         = False,
        return_scores:        Optional[bool]         = False,
        eos_token_id:         Optional[int]          = None,
        guidance_level:       Optional[float]        = 1.0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids   = kwargs.pop("position_ids",   None)
        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("extra_mm", None)

        attention_bias = None
        if images is not None or image_embeds is not None:
            (
                input_ids, position_ids, attention_mask,
                past_key_values, inputs_embeds, labels,
                _, _, attention_bias, _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None,
                images=images, image_embeds=image_embeds,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if self.is_object_centric() and use_customize_greedy:
            return self.greedy_decode_oc(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=position_ids,
                guidance_level=guidance_level,
                return_slots=return_scores,
            )

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        return inputs


# ─────────────────────────────────────────────────────────────────────────────
# AutoConfig / AutoModel 등록
# ─────────────────────────────────────────────────────────────────────────────

AutoConfig.register("cambrian_qwen", ScaleRAEQwenConfig)
AutoModelForCausalLM.register(ScaleRAEQwenConfig, ScaleRAEQwenForCausalLM)