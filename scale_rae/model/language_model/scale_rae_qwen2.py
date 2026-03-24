"""
Scale-RAE Qwen2 Language Model with Object-Centric Extension

기존 Scale-RAE의 ScaleRAEQwenForCausalLM에 Object-Centric Slot Generation을 통합.

주요 변경사항:
  1. ScaleRAEQwenConfig에 OC 관련 설정 필드 추가
  2. __init__에서 OC 모듈 (SlotEOSDetector, ObjectTokenAggregator) 초기화
  3. forward에서 OC 분기 처리 (slot hidden state 추출 → EOS 판단 → Aggregation → FM Loss)
  4. compute_oc_loss: FM + EOS + Diversity 합산 손실
  5. greedy_decode_oc: OC 추론 전용 디코딩

GPU (CUDA) 환경 전용 - IS_XLA_AVAILABLE=False 가정.
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

# Scale-RAE 내부 모듈
from ..scale_rae_arch import ScaleRAEMetaModel, ScaleRAEMetaForCausalLM, apply_custom_kernel
from scale_rae.utils import IS_XLA_AVAILABLE
from scale_rae.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

# XLA는 GPU 환경에서 비활성화
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
    """
    토큰별 feature prediction loss (feature dimension으로 정규화).

    Args:
        predictions:     (B, T, F)
        targets:         (B, T, F)
        valid_positions: (B, T) binary mask (1 = valid)
        loss_type:       'l1' | 'l2' | 'smooth_l1'

    Returns:
        scalar loss
    """
    mask_expanded = valid_positions.unsqueeze(-1).expand_as(predictions)
    feature_dim   = predictions.size(-1)

    if loss_type == 'l1':
        diff = torch.abs(predictions - targets)
    elif loss_type == 'smooth_l1':
        diff = F.smooth_l1_loss(predictions, targets, reduction='none')
    else:   # l2 (default)
        diff = (predictions - targets) ** 2

    masked_diff      = diff * mask_expanded
    per_token_loss   = masked_diff.sum(dim=-1) / feature_dim   # (B, T)
    total_loss       = per_token_loss.sum()
    num_valid        = valid_positions.sum() + 1e-8
    return total_loss / num_valid


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEQwenConfig(Qwen2Config):
    """
    Scale-RAE + Object-Centric 통합 Config.

    OC 관련 필드:
        use_object_centric (bool):  OC 모드 활성화 (default: False)
        oc_max_slots       (int):   최대 slot 개수 (default: 10)
        oc_d_model         (int):   MLLM hidden size 자동 설정 (hidden_size와 동일)
    """

    model_type = "cambrian_qwen"

    # 기존 Scale-RAE 기본값
    vision_loss                    = "regression-loss"
    vision_loss_mode               = "causal"
    vision_tower_aux_token_len_list = [256]

    # ─── Object-Centric 설정 ──────────────────────────────────────
    use_object_centric = False
    oc_max_slots       = 10
    oc_d_model         = 1536    # Qwen2.5-1.5B 기본값; from_pretrained 시 hidden_size로 덮어씀


# ─────────────────────────────────────────────────────────────────────────────
# Qwen2 Model (Scale-RAE MetaModel 믹스인)
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAEQwenModel(ScaleRAEMetaModel, Qwen2Model):
    config_class = ScaleRAEQwenConfig

    def __init__(self, config: Qwen2Config):
        # GPU 환경에서는 eager attention 강제 불필요 (SDPA 사용)
        super(ScaleRAEQwenModel, self).__init__(config)

    def forward(
        self,
        input_ids:                          torch.LongTensor = None,
        attention_mask:                     Optional[torch.Tensor] = None,
        position_ids:                       Optional[torch.LongTensor] = None,
        past_key_values:                    Optional[List[torch.FloatTensor]] = None,
        inputs_embeds:                      Optional[torch.FloatTensor] = None,
        use_cache:                          Optional[bool] = None,
        output_attentions:                  Optional[bool] = None,
        output_hidden_states:               Optional[bool] = None,
        return_dict:                        Optional[bool] = None,
        vision_tower_aux_feature_list:      Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size:          Optional[List[tuple]] = None,
        global_context_feature:            Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions    = output_attentions  if output_attentions  is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache if use_cache is not None else self.config.use_cache
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids와 inputs_embeds를 동시에 지정할 수 없습니다.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("input_ids 또는 inputs_embeds 중 하나는 지정해야 합니다.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True`는 gradient checkpointing과 호환되지 않습니다. `use_cache=False`로 설정합니다.")
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

        # ── Attention mask 준비 (GPU/SDPA path) ──────────────────
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
                sliding_window=getattr(self.config, 'sliding_window', None),
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns    = () if output_attentions    else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
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
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Model: ScaleRAEQwenForCausalLM
# ─────────────────────────────────────────────────────────────────────────────

from ..diffusion_loss.diffloss import create_rf_projector


class ScaleRAEQwenForCausalLM(Qwen2ForCausalLM, ScaleRAEMetaForCausalLM):
    """
    Scale-RAE + Object-Centric Slot Generation 통합 모델.

    Object-Centric 모드 (use_object_centric=True):
      입력 시퀀스: [img_patch×256 | learnable_query×256 | <SLOT>×max_slots]
      OC Attention mask를 통해 slot간 causal 관계 형성
      SlotEOSDetector로 가변 개수 slot 추출
      ObjectTokenAggregator로 고정 크기 context 생성
      diff_head로 Flow Matching Loss 계산
    """

    config_class = ScaleRAEQwenConfig

    def __init__(self, config: ScaleRAEQwenConfig):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type  = "cambrian_qwen"
        config.rope_scaling = None

        self.model   = ScaleRAEQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── Vision Loss 설정 ──────────────────────────────────────
        self.vision_loss      = getattr(config, 'vision_loss',      'diffusion-loss')
        self.vision_loss_mode = getattr(config, 'vision_loss_mode', 'query')
        self.vision_coef      = getattr(config, 'vision_coef',      1.0)
        self.vision_tower_aux_token_len_list = getattr(config, 'vision_tower_aux_token_len_list', [256])
        self.diffusion_model_channels = getattr(config, 'diffusion_model_channels', 1152)
        self.num_image_tokens = 256

        # ── Diffusion Head 초기화 ─────────────────────────────────
        if self.vision_loss in ('diffusion-loss', 'ddt-loss'):
            vision_loss_mode_cfg = getattr(config, 'vision_loss_mode', 'query')

            if vision_loss_mode_cfg in ('query', 'query-block'):
                self.diff_head_config = {
                    "diffusion_tokens":    self.vision_tower_aux_token_len_list[0],
                    "diffusion_channels":  self.diffusion_model_channels,
                    "z_channels":          config.hidden_size,
                    "model_hidden_size":   getattr(config, 'diffusion_model_hidden_size', 1152),
                    "model_depth":         getattr(config, 'diffusion_model_depth',       12),
                    "model_heads":         getattr(config, 'diffusion_model_heads',       16),
                    "guidance_scale":      1.0,
                    "use_mlp":             False,
                    "dit_cls":             getattr(config, 'dit_cls', 'DiT'),
                }
                if getattr(config, 'diffusion_model_z_channels', 0) != 0:
                    self.diff_head_projector    = nn.Linear(config.hidden_size, config.diffusion_model_z_channels)
                    self.use_diff_head_projector = True
                    self.diff_head_config["z_channels"] = config.diffusion_model_z_channels
                else:
                    self.use_diff_head_projector = False

            else:
                # causal 또는 기타 모드 (단순 regression 대용)
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

        # ── Object-Centric 모듈 초기화 ──────────────────────────────
        use_oc = getattr(config, 'use_object_centric', False)
        if use_oc:
            self._init_object_centric_modules(config)

        self.post_init()

    # ─────────────────────────────────────────────────────────────────────────
    # Object-Centric 초기화
    # ─────────────────────────────────────────────────────────────────────────

    def _init_object_centric_modules(self, config: ScaleRAEQwenConfig) -> None:
        """
        Object-Centric 관련 모듈 초기화.
        use_object_centric=True일 때 __init__에서 호출.

        추가되는 파라미터:
          - slot_token_emb   : <SLOT> 특수 토큰 임베딩 (학습 가능)
          - slot_eos_detector: SlotEOSDetector (eos_emb 포함)
          - slot_aggregator  : ObjectTokenAggregator
          - null_slot_token  : 슬롯 소거용 null embedding
        """
        from .object_centric.slot_generator import SlotEOSDetector, ObjectTokenAggregator

        # oc_d_model: config에서 읽되, 없으면 hidden_size 사용
        oc_d_model   = getattr(config, 'oc_d_model',   config.hidden_size)
        oc_max_slots = getattr(config, 'oc_max_slots', 10)

        # config에 d_model 기록 (나중에 참조 가능)
        config.oc_d_model = oc_d_model

        # <SLOT> 토큰 임베딩: (1, 1, D) → forward에서 expand
        self.slot_token_emb = nn.Parameter(
            torch.randn(1, 1, oc_d_model) * 0.02
        )

        # EOS 판단 모듈 (eos_emb 포함)
        self.slot_eos_detector = SlotEOSDetector(d_model=oc_d_model)

        # Slot → Context 변환
        # num_heads: d_model에 맞게 자동 결정 (8 또는 16)
        num_heads = 16 if oc_d_model >= 3584 else 8
        self.slot_aggregator = ObjectTokenAggregator(
            d_model=oc_d_model,
            num_heads=num_heads,
        )

        # 슬롯 소거용 null token (0 초기화)
        self.null_slot_token = nn.Parameter(torch.zeros(1, 1, oc_d_model))

        self.oc_max_slots = oc_max_slots
        self.oc_d_model   = oc_d_model

        logger.info(
            f"[OC] Object-Centric 모듈 초기화 완료: "
            f"d_model={oc_d_model}, max_slots={oc_max_slots}, num_heads={num_heads}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 유틸리티
    # ─────────────────────────────────────────────────────────────────────────

    def get_model(self):
        return self.model

    def is_object_centric(self) -> bool:
        """OC 모드 활성화 여부"""
        return getattr(self, 'slot_aggregator', None) is not None

    def load_vision_head(self, model_args) -> None:
        """pretrain_adapter_and_vision_head 체크포인트에서 헤드 가중치 로드."""
        pretrain_path = getattr(model_args, 'pretrain_adapter_and_vision_head', None)
        if pretrain_path is None:
            return

        logger.info(f"[VisionHead] {pretrain_path} 에서 가중치 로드")
        weights = torch.load(pretrain_path, map_location='cpu')
        model_weights = weights.get('model', weights)

        def get_w(w, key):
            return {k.split(key + '.')[1]: v for k, v in w.items() if key + '.' in k}

        if hasattr(self, 'vision_head') and any('vision_head.' in k for k in model_weights):
            self.vision_head.load_state_dict(get_w(model_weights, 'vision_head'), strict=False)

        if hasattr(self, 'diff_head') and any('diff_head.' in k for k in model_weights):
            self.diff_head.load_state_dict(get_w(model_weights, 'diff_head'), strict=False)

    def _init_weights(self, module: nn.Module) -> None:
        """OC 모듈은 자체 초기화 유지, 나머지는 Qwen2 기본 초기화."""
        oc_modules = ['slot_eos_detector', 'slot_aggregator']
        for oc_name in oc_modules:
            oc_mod = getattr(self, oc_name, None)
            if oc_mod is not None:
                for child in oc_mod.modules():
                    if module is child:
                        return   # OC 모듈은 건드리지 않음
        super()._init_weights(module)

    # ─────────────────────────────────────────────────────────────────────────
    # Object-Centric Loss
    # ─────────────────────────────────────────────────────────────────────────

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
        Object-Centric 3종 Loss 계산.

        Loss 구성:
          total = 1.0 * fm_loss + 0.1 * eos_loss + 0.05 * div_loss

          fm_loss  : Flow Matching (GT = RAE.encode(전체 이미지), crop/Hungarian 없음)
          eos_loss : K_gt번째 slot이 EOS가 되도록 학습
          div_loss : slot끼리 cosine similarity 최소화 (다양한 object 표현 유도)

        Args:
            context:           (B, 256, D) - ObjectTokenAggregator 출력
            gt_image_features: (B, 256, 1152) - RAE.encode(전체 이미지) GT
            eos_sim:           (B, max_slots) - SlotEOSDetector 출력 similarity
            valid_slots:       (B, max_slots, D) - EOS 처리된 슬롯
            K_gt:              (B,) long - COCO GT object 수
            answer_img_mask:   (B, M) optional - answer 이미지 마스크 (multi-image 시)

        Returns:
            dict: fm_loss, eos_loss, div_loss, total_loss
        """
        B         = context.shape[0]
        max_slots = eos_sim.shape[1]
        device    = context.device

        # ── 1. Flow Matching Loss ─────────────────────────────────
        # context: (B, 256, D) → diff_head의 z (conditioning)
        # gt_image_features: (B, 256, 1152) → diff_head의 x (target)
        if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
            context_proj = self.diff_head_projector(context)
        else:
            context_proj = context

        # diff_head.training_loss returns per-sample loss tensor
        fm_loss_vec = self.diff_head.training_loss(
            z=context_proj,
            x=gt_image_features,
        )  # (B,) 또는 scalar

        if answer_img_mask is not None:
            # Multi-image 모드: answer image만 손실 계산
            # answer_img_mask: (B, M) → 이미지당 하나라고 가정 (M=1)
            mask = answer_img_mask[:, 0].float()   # (B,)
            fm_loss = (fm_loss_vec * mask).sum() / (mask.sum() + 1e-8)
        else:
            fm_loss = fm_loss_vec.mean() if fm_loss_vec.dim() > 0 else fm_loss_vec

        # ── 2. EOS Loss ───────────────────────────────────────────
        # K_gt번째 슬롯을 EOS로, 나머지는 non-EOS로 학습
        eos_gt = torch.zeros(B, max_slots, device=device)
        for b in range(B):
            k = min(int(K_gt[b].item()), max_slots - 1)
            eos_gt[b, k] = 1.0

        eos_loss = F.binary_cross_entropy_with_logits(
            eos_sim * 5.0,   # scale: gradient 안정화
            eos_gt,
            reduction='mean',
        )

        # ── 3. Slot Diversity Loss ───────────────────────────────
        # 유효 슬롯끼리 cosine similarity off-diagonal 최소화
        slots_norm  = F.normalize(valid_slots, dim=-1)           # (B, max, D)
        sim_matrix  = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # (B, max, max)
        eye         = torch.eye(max_slots, device=device).unsqueeze(0)   # (1, max, max)
        off_diag    = sim_matrix * (1.0 - eye)                            # 대각선 제외
        n_pairs     = max_slots * (max_slots - 1)
        div_loss    = off_diag.sum() / (B * n_pairs + 1e-8)

        # ── 총 Loss ──────────────────────────────────────────────
        total_loss = 1.0 * fm_loss + 0.1 * eos_loss + 0.05 * div_loss

        return {
            'fm_loss':    fm_loss,
            'eos_loss':   eos_loss,
            'div_loss':   div_loss,
            'total_loss': total_loss,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:           torch.LongTensor = None,
        attention_mask:      Optional[torch.Tensor] = None,
        position_ids:        Optional[torch.LongTensor] = None,
        past_key_values:     Optional[List[torch.FloatTensor]] = None,
        inputs_embeds:       Optional[torch.FloatTensor] = None,
        labels:              Optional[torch.LongTensor] = None,
        use_cache:           Optional[bool] = None,
        output_attentions:   Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images:              Optional[torch.FloatTensor] = None,
        return_dict:         Optional[bool] = None,
        vision_token_indices: Optional[torch.Tensor] = None,
        decoding:            Optional[bool] = False,
        answer_img_mask:     Optional[torch.Tensor] = None,
        reverse_vti:         Optional[torch.Tensor] = None,
        answer_token_mask:   Optional[torch.Tensor] = None,
        guidance_level:      Optional[float] = 1.0,
        images_gen:          Optional[torch.FloatTensor] = None,
        # ── Object-Centric 추가 인자 ──────────────────────────────
        n_objects:           Optional[torch.Tensor] = None,   # (B,) GT object 수
        gt_image_features:   Optional[torch.Tensor] = None,   # (B, 256, 1152) RAE encode 결과
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ── 멀티모달 입력 전처리 ─────────────────────────────────
        selected_features = input_embed_mask = attention_bias = extra_mm = None

        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                selected_features,
                input_embed_mask,
                attention_bias,
                extra_mm,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
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

        # ── MLLM forward ─────────────────────────────────────────
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
        )

        hidden_states = outputs[0]   # (B, L, D)
        logits        = self.lm_head(hidden_states).float()

        # ── Object-Centric 분기 ───────────────────────────────────
        if self.is_object_centric() and not decoding:
            return self._forward_oc_train(
                hidden_states, logits, labels,
                outputs, return_dict,
                n_objects=n_objects,
                gt_image_features=gt_image_features,
                answer_img_mask=answer_img_mask,
            )

        # ── Decoding 분기 (OC 추론) ───────────────────────────────
        if self.is_object_centric() and decoding:
            pred_z = self._forward_oc_decode(hidden_states, guidance_level)
            return CausalLMOutputWithPast(
                loss=pred_z,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )

        # ── 기존 비-OC 학습 경로 ─────────────────────────────────
        return self._forward_standard(
            hidden_states, logits, labels, outputs,
            selected_features, input_embed_mask,
            extra_mm, return_dict, decoding,
            answer_img_mask, answer_token_mask,
            guidance_level,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OC 학습 forward
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_oc_train(
        self,
        hidden_states:      torch.Tensor,
        logits:             torch.Tensor,
        labels:             Optional[torch.Tensor],
        outputs,
        return_dict:        bool,
        n_objects:          Optional[torch.Tensor],
        gt_image_features:  Optional[torch.Tensor],
        answer_img_mask:    Optional[torch.Tensor],
    ) -> CausalLMOutputWithPast:
        """
        OC 학습용 forward.

        시퀀스 구조 (prepare_inputs_labels_for_multimodal에서 구성됨):
          [img_patch×256 | learnable_query×256 | <SLOT>×max_slots | text tokens ...]

        hidden_states에서 각 영역을 추출하여 OC 손실 계산.
        """
        B    = hidden_states.shape[0]
        n_img    = self.num_image_tokens       # 256
        n_query  = self.num_image_tokens       # 256 (latent_queries와 동일)
        n_slots  = self.oc_max_slots

        # ── 시퀀스에서 각 영역 추출 ─────────────────────────────
        # 주의: inputs_embeds 구성 순서와 일치해야 함
        # scale_rae_arch.py의 prepare_inputs_labels_for_multimodal에서:
        #   [img_embed×256 | latent_query×256 | slot_tokens×max_slots | text ...]
        base_context = hidden_states[:, n_img: n_img + n_query, :]                   # (B, 256, D)
        raw_slots    = hidden_states[:, n_img + n_query: n_img + n_query + n_slots, :]  # (B, max, D)

        # ── EOS 처리 ─────────────────────────────────────────────
        valid_slots, slot_mask, eos_sim = self.slot_eos_detector(raw_slots)

        # ── Aggregation → context ────────────────────────────────
        context = self.slot_aggregator(base_context, valid_slots, slot_mask)  # (B, 256, D)

        # ── 언어 손실 ────────────────────────────────────────────
        loss = torch.tensor(0.0, device=hidden_states.device)
        if labels is not None:
            # text 부분 (slot 이후)의 언어 손실
            text_start = n_img + n_query + n_slots
            shift_logits = logits[:, text_start - 1: -1, :].contiguous()
            shift_labels = labels[:, text_start:].contiguous()
            loss_fct     = CrossEntropyLoss()
            shift_logits  = shift_logits.view(-1, self.config.vocab_size)
            shift_labels  = shift_labels.view(-1).to(shift_logits.device)
            lm_loss       = loss_fct(shift_logits, shift_labels)
            loss          = lm_loss
            self.loss_language = lm_loss

        # ── OC 손실 계산 ─────────────────────────────────────────
        if gt_image_features is not None and n_objects is not None:
            oc_losses = self.compute_oc_loss(
                context=context,
                gt_image_features=gt_image_features,
                eos_sim=eos_sim,
                valid_slots=valid_slots,
                K_gt=n_objects,
                answer_img_mask=answer_img_mask,
            )
            # OC 손실을 기존 언어 손실에 추가
            loss = loss + oc_losses['total_loss'] * self.vision_coef

            # 로깅용 저장
            self.loss_image_diff = oc_losses['fm_loss']
            self.oc_eos_loss     = oc_losses['eos_loss']
            self.oc_div_loss     = oc_losses['div_loss']

        # OC 출력 저장 (trainer logging 및 디버깅용)
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

    # ─────────────────────────────────────────────────────────────────────────
    # OC 추론 decode
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_oc_decode(
        self,
        hidden_states:  torch.Tensor,
        guidance_level: float = 1.0,
    ) -> torch.Tensor:
        """
        OC 추론: slot 추출 → aggregation → diff_head 생성.

        Returns:
            pred_z: (B, 256, 1152) - 예측된 이미지 latent
        """
        n_img   = self.num_image_tokens
        n_query = self.num_image_tokens
        n_slots = self.oc_max_slots

        base_context = hidden_states[:, n_img: n_img + n_query, :]
        raw_slots    = hidden_states[:, n_img + n_query: n_img + n_query + n_slots, :]

        # EOS hard cut (추론 시 self.training=False 이므로 hard mask 적용)
        valid_slots, slot_mask, eos_sim = self.slot_eos_detector(raw_slots)
        context = self.slot_aggregator(base_context, valid_slots, slot_mask)

        if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
            context_proj = self.diff_head_projector(context)
        else:
            context_proj = context

        pred_z = self.diff_head.infer(context_proj, guidance_level=guidance_level)
        return pred_z

    # ─────────────────────────────────────────────────────────────────────────
    # 기존 비-OC 표준 forward
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_standard(
        self,
        hidden_states,
        logits,
        labels,
        outputs,
        selected_features,
        input_embed_mask,
        extra_mm,
        return_dict,
        decoding,
        answer_img_mask,
        answer_token_mask,
        guidance_level,
    ) -> CausalLMOutputWithPast:
        """
        기존 비-OC 경로 (causal / query 모드) 학습 및 추론.
        """
        # ── Decoding 모드 ─────────────────────────────────────────
        if decoding:
            vision_loss_mode = self.vision_loss_mode
            use_query_mode   = vision_loss_mode in ("query", "half-query", "query-block")
            generated_token_length = self.num_image_tokens if use_query_mode else 1

            if self.vision_loss == 'regression-loss':
                pred_z = hidden_states[:, -generated_token_length:, :].squeeze(1)
                pred_z = self.vision_head(pred_z)
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
                loss=pred_z,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )

        # ── 학습 모드 ─────────────────────────────────────────────
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct     = CrossEntropyLoss()
            shift_logits  = shift_logits.view(-1, self.config.vocab_size)
            shift_labels  = shift_labels.view(-1).to(shift_logits.device)
            loss          = loss_fct(shift_logits, shift_labels)
            self.loss_language = loss

            # ── query 모드 vision loss ────────────────────────────
            vision_loss_mode_cfg = getattr(self.get_model().config, 'vision_loss_mode', 'causal')

            if vision_loss_mode_cfg in ("query", "query-block") and extra_mm is not None:
                img_feats_raw, reverse_vti, answer_img_mask_em, prediction_target = extra_mm
                B, T, feature_dim = prediction_target.shape
                M = answer_img_mask_em.size(1) if answer_img_mask_em is not None else 1
                tokens_per_image = T // M
                hidden_dim       = hidden_states.size(-1)

                # hidden states 수집
                hs_full    = hidden_states
                zeros_left = torch.zeros(B, T, hidden_dim, dtype=hs_full.dtype, device=hs_full.device)
                patch_hs   = apply_custom_kernel(zeros_left, hs_full, reverse_vti)

                if self.vision_loss in ('diffusion-loss', 'ddt-loss'):
                    # reshape
                    patch_hs_r = patch_hs.view(B, M, tokens_per_image, hidden_dim).view(B * M, tokens_per_image, hidden_dim)
                    if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
                        patch_hs_r = self.diff_head_projector(patch_hs_r)
                    pred_r = prediction_target.view(B, M, tokens_per_image, feature_dim).view(B * M, tokens_per_image, feature_dim)

                    diff_loss_vec = self.diff_head.training_loss(z=patch_hs_r, x=pred_r)
                    diff_loss_mat = diff_loss_vec.view(B, M)

                    if answer_img_mask_em is not None:
                        masked = diff_loss_mat * answer_img_mask_em.float()
                        mean_diff = masked.sum() / (answer_img_mask_em.sum() + 1e-8)
                    else:
                        mean_diff = diff_loss_mat.mean()

                    loss = loss + mean_diff * self.vision_coef
                    self.loss_image_diff = mean_diff

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

    # ─────────────────────────────────────────────────────────────────────────
    # OC 전용 Greedy Decode
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode_oc(
        self,
        inputs_embeds:      torch.Tensor,
        attention_mask:     Optional[torch.Tensor] = None,
        position_ids:       Optional[torch.LongTensor] = None,
        eos_token_id:       Optional[Union[int, List[int]]] = None,
        max_new_tokens:     int = 512,
        guidance_level:     float = 1.0,
        return_slots:       bool = False,
    ) -> dict:
        """
        OC 모드 전용 greedy decode.

        OC 시퀀스 구성 후 single forward pass로 처리.

        Args:
            inputs_embeds:  (B, L, D) - 기 구성된 입력 임베딩
                            ([img×256 | query×256 | slot×max_slots] 포함)
            attention_mask: (B, L)
            position_ids:   (B, L)
            eos_token_id:   텍스트 EOS
            max_new_tokens: 텍스트 생성 최대 토큰
            guidance_level: CFG scale (1.0 = no CFG)
            return_slots:   True이면 슬롯 정보도 반환

        Returns:
            dict:
                pred_image:     (B, 256, 1152) 예측 이미지 latent
                generated_ids:  생성된 텍스트 token ids
                slots (opt):    dict with base_context, valid_slots, eos_sim
        """
        B = inputs_embeds.shape[0]
        n_img   = self.num_image_tokens
        n_query = self.num_image_tokens
        n_slots = self.oc_max_slots

        # <SLOT> 토큰 붙이기 (prepare_inputs에서 미리 안 붙인 경우)
        slot_tokens = self.slot_token_emb.expand(B, n_slots, -1)
        inputs_embeds_oc = torch.cat([inputs_embeds, slot_tokens], dim=1)

        if attention_mask is not None:
            slot_attn     = torch.ones(B, n_slots, device=inputs_embeds.device)
            attention_mask = torch.cat([attention_mask, slot_attn], dim=1)

        # ── OC Attention Mask 적용 ────────────────────────────────
        from .object_centric.slot_generator import build_oc_attention_mask
        L_base = inputs_embeds.shape[1]
        # 간단한 full-attention 기본값 (OC 마스크는 prepare에서 처리됨)
        # 여기서는 slot→slot causal만 추가 적용
        oc_mask = build_oc_attention_mask(
            n_img=n_img, n_query=n_query, n_slots=n_slots,
            device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )  # (L_slot, L_slot) where L_slot = n_img+n_query+n_slots

        # forward
        outputs = self.model(
            inputs_embeds=inputs_embeds_oc,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state   # (B, L_oc, D)

        # ── Slot 추출 및 이미지 생성 ─────────────────────────────
        base_context = hidden_states[:, n_img: n_img + n_query, :]
        raw_slots    = hidden_states[:, n_img + n_query: n_img + n_query + n_slots, :]

        # 추론 모드이므로 SlotEOSDetector는 hard cut 사용
        self.eval()
        valid_slots, slot_mask, eos_sim = self.slot_eos_detector(raw_slots)
        context = self.slot_aggregator(base_context, valid_slots, slot_mask)

        if hasattr(self, 'use_diff_head_projector') and self.use_diff_head_projector:
            context_proj = self.diff_head_projector(context)
        else:
            context_proj = context

        pred_image = self.diff_head.infer(context_proj, guidance_level=guidance_level)

        result = {
            'pred_image':    pred_image,      # (B, 256, 1152)
            'generated_ids': [],
        }

        if return_slots:
            result['slots'] = {
                'base_context': base_context,
                'valid_slots':  valid_slots,
                'eos_sim':      eos_sim,
                'slot_mask':    slot_mask,
            }

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Greedy decode (기존 호환 유지)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        inputs:               Optional[torch.Tensor] = None,
        images:               Optional[torch.Tensor] = None,
        image_embeds:         Optional[torch.Tensor] = None,
        use_customize_greedy: Optional[bool]         = False,
        return_scores:        Optional[bool]         = False,
        start_image_token_id: Optional[int]          = None,
        end_image_token_id:   Optional[int]          = None,
        eos_token_id:         Optional[int]          = None,
        guidance_level:       Optional[float]        = 1.0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids   = kwargs.pop("position_ids",   None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds`는 직접 지원하지 않습니다.")

        extra_mm = kwargs.pop("extra_mm", None)

        if images is not None or image_embeds is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                selected_features,
                input_embed_mask,
                attention_bias,
                extra_mm,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images=images,
                image_embeds=image_embeds,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # OC 모드: 전용 greedy decode
        if self.is_object_centric() and use_customize_greedy:
            return self.greedy_decode_oc(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                eos_token_id=eos_token_id,
                guidance_level=guidance_level,
                return_slots=return_scores,
                **{k: v for k, v in kwargs.items() if k in ('max_new_tokens',)},
            )

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        return inputs


# ─────────────────────────────────────────────────────────────────────────────
# AutoConfig / AutoModel 등록
# ─────────────────────────────────────────────────────────────────────────────

AutoConfig.register("cambrian_qwen", ScaleRAEQwenConfig)
AutoModelForCausalLM.register(ScaleRAEQwenConfig, ScaleRAEQwenForCausalLM)