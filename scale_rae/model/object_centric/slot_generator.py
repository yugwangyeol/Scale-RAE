"""
Object-Centric Slot Generator

MLLM의 Causal Attention을 활용하여 이미지에서 가변 개수의 Object Slot을 생성.
LLM이 텍스트를 생성하듯, 각 slot은 이전 slot들이 표현한 것과 다른 object를 표현.

입력 시퀀스 구조:
  [img_patch×256 | learnable_query×256 | <SLOT>×max_slots]

Attention Mask:
  - img_patch, learnable_query: bi-directional (서로 full attention)
  - slot_i: img_patch와 learnable_query 전체 + slot_0..i-1만 attend (causal)
  - slot_i는 slot_j(j>=i)를 볼 수 없음

출력:
  - base_context: learnable_query 위치의 hidden states (B, 256, D)
  - raw_slots: <SLOT> 위치의 hidden states (B, max_slots, D)
  - eos_sim: 각 slot과 eos_emb의 cosine similarity (B, max_slots)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SlotEOSDetector(nn.Module):
    """
    각 slot이 EOS인지 판단.
    별도 head 없이 learnable eos_emb와의 cosine similarity로 판단.
    LLM의 <EOS> 토큰 생성과 동일한 원리.

    학습 시: soft sigmoid mask → end-to-end gradient 흐름 보장
    추론 시: 첫 번째 EOS 이후 hard cut → 실제 가변 개수 추출

    Args:
        d_model (int): MLLM hidden size (Qwen2.5-1.5B=1536, 7B=3584)
    """

    def __init__(self, d_model: int):
        super().__init__()
        # EOS 판단 기준 임베딩 (학습 가능)
        self.eos_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)

    def forward(
        self,
        raw_slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_slots: (B, max_slots, D) - MLLM slot 위치 hidden states

        Returns:
            valid_slots: (B, max_slots, D) - EOS 이후 슬롯은 0으로 마스킹
            slot_mask:   (B, max_slots) bool - True = padding (EOS 이후)
            eos_sim:     (B, max_slots) float - 각 슬롯과 EOS 임베딩의 cosine similarity
        """
        B, max_slots, D = raw_slots.shape

        # cosine similarity: eos_emb와 각 슬롯 비교
        slots_norm = F.normalize(raw_slots, dim=-1)            # (B, max, D)
        eos_norm   = F.normalize(self.eos_emb, dim=-1)         # (1, D)
        eos_sim    = (slots_norm * eos_norm).sum(dim=-1)        # (B, max) [-1, 1]

        if self.training:
            # 학습 시: soft mask (미분 가능 → end-to-end 학습)
            # 0.5 threshold 기준 sigmoid로 부드럽게 마스킹
            valid_weights = 1.0 - torch.sigmoid((eos_sim - 0.5) * 10.0)  # (B, max)
            valid_slots   = raw_slots * valid_weights.unsqueeze(-1)
            slot_mask     = (valid_weights < 0.1)                          # True = padding
        else:
            # 추론 시: hard cut (첫 번째 EOS 이후 전부 0)
            is_eos     = (eos_sim > 0.5)                                   # (B, max) bool
            # cumsum >= 1이 되는 시점부터 padding으로 처리
            slot_mask  = is_eos.float().cumsum(dim=-1) >= 1.0             # (B, max) bool
            slot_mask  = slot_mask.bool()
            valid_slots = raw_slots.clone()
            valid_slots[slot_mask] = 0.0

        return valid_slots, slot_mask, eos_sim


class ObjectTokenAggregator(nn.Module):
    """
    가변 K개의 slot을 고정 크기 context (256, D)로 변환.

    핵심 설계:
      Q = base_context  (MLLM learnable_query 출력, 기존 Scale-RAE 파라미터 재사용)
      K = V = valid_slots (EOS 처리된 슬롯들)
      output = CrossAttention(Q, K, V) + base_context  (residual)

    residual 덕분에 슬롯이 0개여도 base_context만으로 이미지 전체 정보 보존.
    파라미터 효율적: 새로운 fixed_query를 만들지 않고 기존 것 재사용.

    Args:
        d_model (int): hidden size (Qwen2.5-1.5B=1536, 7B=3584)
        num_heads (int): multi-head attention heads
    """

    def __init__(self, d_model: int = 1536, num_heads: int = 8):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model({d_model}) must be divisible by num_heads({num_heads})"

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        base_context: torch.Tensor,
        valid_slots:  torch.Tensor,
        slot_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            base_context: (B, 256, D) - MLLM learnable_query 위치의 hidden states
            valid_slots:  (B, K, D)   - EOS 처리된 slots (K = max_slots, padding 포함)
            slot_mask:    (B, K) bool  - True = padding slot (key_padding_mask)

        Returns:
            context: (B, 256, D) - DiT AdaLN conditioning에 들어갈 고정 크기 context
        """
        # --- Cross-Attention: base_context ← valid_slots ---
        delta, _ = self.cross_attn(
            query=base_context,
            key=valid_slots,
            value=valid_slots,
            key_padding_mask=slot_mask,   # padding slot 무시
        )

        # Post-LN residual connection (Pre-LN 대비 학습 안정성 우선)
        x = self.norm1(base_context + delta)

        # FFN + residual
        x = self.norm2(x + self.ffn(x))

        return x   # (B, 256, D)


def build_oc_attention_mask(
    n_img:    int,
    n_query:  int,
    n_slots:  int,
    device:   torch.device,
    dtype:    torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Object-Centric용 Additive Attention Mask 생성.

    시퀀스 레이아웃:
      [img × n_img | query × n_query | slot × n_slots]
      총 길이 L = n_img + n_query + n_slots

    Attention 규칙:
      - img  ↔ img  : bi-directional (full attention)
      - img  ↔ query: bi-directional (full attention)
      - query↔ query: bi-directional (full attention)
      - slot_i → img   전체     : attend 가능
      - slot_i → query 전체     : attend 가능
      - slot_i → slot_j (j < i) : causal attend (이전 슬롯만)
      - slot_i → slot_j (j >= i): 차단 (-inf)  자기 자신 포함

    반환값이 0이면 attend, -inf이면 차단.
    Qwen2 등 transformers의 additive attention mask 형식과 동일.
    causal mask에 더해지므로 0을 더하면 기존 causal 유지, -inf를 더하면 강제 차단.

    Args:
        n_img   : 이미지 패치 토큰 수 (일반적으로 256)
        n_query : learnable query 토큰 수 (일반적으로 256)
        n_slots : 최대 slot 수
        device  : torch device
        dtype   : 마스크 dtype (float32 권장)

    Returns:
        mask: (L, L) additive attention mask
              0.0 = attend, float('-inf') = block
    """
    L = n_img + n_query + n_slots
    # 기본값: 모두 차단
    mask = torch.full((L, L), float('-inf'), device=device, dtype=dtype)

    # ── img ↔ img : full attention ─────────────────────────────────
    mask[:n_img, :n_img] = 0.0

    # ── img ↔ query : bi-directional ──────────────────────────────
    mask[:n_img,  n_img:n_img + n_query] = 0.0
    mask[n_img:n_img + n_query, :n_img]  = 0.0

    # ── query ↔ query : full attention ─────────────────────────────
    mask[n_img:n_img + n_query, n_img:n_img + n_query] = 0.0

    # ── slot_i → img   전체 attend ─────────────────────────────────
    mask[n_img + n_query:, :n_img] = 0.0

    # ── slot_i → query 전체 attend ─────────────────────────────────
    mask[n_img + n_query:, n_img:n_img + n_query] = 0.0

    # ── slot → slot : strictly causal ──────────────────────────────
    # slot_i는 slot_0 .. slot_{i-1}만 attend (자기 자신 제외)
    slot_start = n_img + n_query
    for i in range(n_slots):
        if i > 0:
            # slot_i → slot_0 .. slot_{i-1}
            mask[slot_start + i, slot_start: slot_start + i] = 0.0
        # slot_i는 자기 자신(slot_i)과 slot_{i+1..}을 attend 못함 → 이미 -inf

    return mask   # (L, L)