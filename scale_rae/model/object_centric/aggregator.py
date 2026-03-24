"""
Slot Manipulation Utilities

슬롯 조작 (제거/교체/이식)을 위한 유틸리티 함수들.
ObjectTokenAggregator와 함께 추론 시 사용.

사용 예시:
    # 특정 object 소거
    modified_slots = remove_slot(slots, target_idx=2, null_token=model.null_slot_token)
    context = aggregator(base_context, modified_slots, slot_mask)

    # 다른 이미지의 object 이식
    transferred = transfer_slot(slots_target, slots_source, target_idx=1, source_idx=0)
    context = aggregator(base_context, transferred, slot_mask)
"""

import torch
import torch.nn as nn
from typing import Optional


def remove_slot(
    slots:      torch.Tensor,
    target_idx: int,
    null_token: torch.Tensor,
) -> torch.Tensor:
    """
    특정 인덱스의 슬롯을 null token으로 교체하여 해당 object를 소거.

    Args:
        slots:      (B, K, D) - 원본 슬롯 텐서
        target_idx: 제거할 슬롯 인덱스 (0-based)
        null_token: (1, 1, D) 또는 (B, 1, D) - 학습된 null embedding
                    (model.null_slot_token 사용 권장)

    Returns:
        modified_slots: (B, K, D) - target_idx 위치가 null_token으로 교체됨

    Example:
        # 2번째 object 소거
        modified = remove_slot(slots, target_idx=2, null_token=model.null_slot_token)
    """
    B = slots.shape[0]
    modified = slots.clone()

    # null_token shape 처리: (1, 1, D) → (B, D)
    null = null_token.expand(B, -1, -1)  # (B, 1, D)
    modified[:, target_idx, :] = null.squeeze(1)  # (B, D)

    return modified


def transfer_slot(
    slots_target: torch.Tensor,
    slots_source: torch.Tensor,
    target_idx:   int,
    source_idx:   int,
) -> torch.Tensor:
    """
    source 이미지의 특정 slot을 target 이미지의 특정 위치에 이식.
    Object-level compositional generation / editing에 사용.

    Args:
        slots_target: (B, K, D) - 수신 이미지의 슬롯 (수정 대상)
        slots_source: (B, K, D) - 기여 이미지의 슬롯 (이식 소스)
        target_idx:   교체할 target 슬롯 인덱스
        source_idx:   가져올 source 슬롯 인덱스

    Returns:
        modified_slots: (B, K, D) - target_idx 위치가 source의 source_idx로 교체됨

    Example:
        # 이미지A의 0번째 object를 이미지B의 1번째 위치에 이식
        result = transfer_slot(slots_B, slots_A, target_idx=1, source_idx=0)
    """
    modified = slots_target.clone()
    modified[:, target_idx, :] = slots_source[:, source_idx, :]
    return modified


def swap_slots(
    slots_a:  torch.Tensor,
    slots_b:  torch.Tensor,
    idx_a:    int,
    idx_b:    int,
) -> tuple:
    """
    두 이미지 간 특정 slot을 서로 교환.

    Args:
        slots_a: (B, K, D)
        slots_b: (B, K, D)
        idx_a:   slots_a에서 교환할 인덱스
        idx_b:   slots_b에서 교환할 인덱스

    Returns:
        (modified_a, modified_b): 교환된 슬롯 텐서 쌍

    Example:
        # A의 고양이 ↔ B의 개 위치 교환
        new_a, new_b = swap_slots(slots_a, slots_b, idx_a=0, idx_b=1)
    """
    mod_a = slots_a.clone()
    mod_b = slots_b.clone()
    tmp          = mod_a[:, idx_a, :].clone()
    mod_a[:, idx_a, :] = mod_b[:, idx_b, :]
    mod_b[:, idx_b, :] = tmp
    return mod_a, mod_b


def interpolate_slots(
    slots_a: torch.Tensor,
    slots_b: torch.Tensor,
    idx_a:   int,
    idx_b:   int,
    alpha:   float = 0.5,
) -> torch.Tensor:
    """
    두 slot 사이를 선형 보간하여 부드러운 object transition 생성.

    Args:
        slots_a: (B, K, D)
        slots_b: (B, K, D)
        idx_a:   slots_a에서 보간할 슬롯 인덱스
        idx_b:   slots_b에서 보간할 슬롯 인덱스
        alpha:   보간 비율 (0.0 = slots_a, 1.0 = slots_b)

    Returns:
        interpolated: (B, D) - 보간된 슬롯 벡터

    Example:
        # 두 이미지의 첫 번째 object 중간 표현
        mid_slot = interpolate_slots(slots_a, slots_b, idx_a=0, idx_b=0, alpha=0.5)
    """
    slot_a = slots_a[:, idx_a, :]  # (B, D)
    slot_b = slots_b[:, idx_b, :]  # (B, D)
    return (1.0 - alpha) * slot_a + alpha * slot_b


class NullSlotToken(nn.Module):
    """
    학습 가능한 null slot token.
    remove_slot에서 object 소거 시 사용.
    ScaleRAEQwenForCausalLM에서 self.null_slot_token 으로 등록됨.

    Args:
        d_model (int): hidden size
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 0으로 초기화: "아무것도 없는" 상태를 표현
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self) -> torch.Tensor:
        """Returns: (1, 1, D)"""
        return self.token