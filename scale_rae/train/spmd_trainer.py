"""
Scale-RAE Trainer (GPU 버전, Object-Centric 확장 포함)

변경사항 (이슈 2, 3 수정 + COCO-only 데이터 파이프라인):
  - Text 없음: conversations 필드 무시, 이미지만 사용
  - n_objects: instances_train2017.json에서 image_id별 annotation 수 카운트
  - gt_image_features: DataCollator에서 raw image tensor를 배치에 포함
    → model.forward에서 vision tower로 encode해서 사용
  - 이슈 3 (시퀀스 길이): OC slot 추가 전 text 부분 clip 로직은 scale_rae_arch.py에서 처리
  - DataCollatorForSupervisedDataset: raw_images 필드 추가
"""

import os
import re
import math
import copy
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

import transformers
import tokenizers
from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from packaging import version

from PIL import Image

# Scale-RAE 내부
import scale_rae
from scale_rae.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from scale_rae import conversation as conversation_lib
from scale_rae.mm_utils import tokenizer_image_token
from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM

logger_module = logging.getLogger(__name__)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path:          Optional[str]   = field(default="facebook/opt-125m")
    version:                     Optional[str]   = field(default="v0")
    freeze_backbone:             bool            = field(default=False)
    tune_mm_mlp_adapter:         bool            = field(default=False)
    tune_adapter_and_vision_head: bool           = field(default=False)
    vision_tower_aux_list:       Optional[str]   = field(default=None)
    mm_vision_select_layer:      Optional[int]   = field(default=-1)
    pretrain_mm_mlp_adapter:     Optional[str]   = field(default=None)
    pretrain_adapter_and_vision_head: Optional[str] = field(default=None)
    mm_projector_type:           Optional[str]   = field(default='linear')
    mm_use_im_start_end:         bool            = field(default=False)
    mm_use_im_patch_token:       bool            = field(default=True)
    mm_vision_select_feature:    Optional[str]   = field(default="patch")
    vision_tower_aux_token_len_list: Optional[str] = field(default=None)
    vision_hidden_size:          Optional[int]   = field(default=1024)
    connector_only:              bool            = field(default=True)
    normalize_vision:            bool            = field(default=True)
    vision_loss:                 Optional[str]   = field(default="diffusion-loss")
    vision_loss_mode:            Optional[str]   = field(default="query")
    vision_coef:                 Optional[float] = field(default=1.0)
    dit_cls:                     Optional[str]   = field(default="DiT")
    diffusion_model_hidden_size: Optional[int]   = field(default=1152)
    diffusion_model_channels:    Optional[int]   = field(default=1152)
    diffusion_model_depth:       Optional[int]   = field(default=12)
    diffusion_model_heads:       Optional[int]   = field(default=16)
    diffusion_model_z_channels:  Optional[int]   = field(default=0)
    si_token_len:                int             = field(default=729)
    miv_token_len:               int             = field(default=196)
    unfreeze_mm_vision_tower:    bool            = field(default=False)

    # ─── Object-Centric 설정 ──────────────────────────────────────
    use_object_centric: bool = field(
        default=False,
        metadata={"help": "Object-Centric Slot Generation 모드 활성화"},
    )
    oc_max_slots: int = field(
        default=10,
        metadata={"help": "최대 slot 개수 (EOS 슬롯 포함)"},
    )
    # ─── COCO annotation 경로 ─────────────────────────────────────
    coco_annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "instances_train2017.json 경로 (n_objects 추출용)"},
    )


@dataclass
class DataArguments:
    data_path:             str             = field(default=None)
    lazy_preprocess:       bool            = False
    image_folder:          Optional[str]   = field(default=None)
    is_multimodal:         bool            = False
    image_aspect_ratio:    str             = 'square'
    image_position:        int             = 35
    max_images_per_sample: int             = 1
    anyres_max_subimages:  int             = 1
    video_folder:          str             = ""
    video_fps:             int             = 1
    video_max_frames:      int             = 1
    video_force_sample:    bool            = False
    add_time_instruction:  bool            = False
    # ─── COCO annotation 경로 ─────────────────────────────────────
    coco_annotation_path:  Optional[str]   = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir:               Optional[str]   = field(default=None)
    optim:                   str             = field(default="adamw_torch")
    remove_unused_columns:   bool            = field(default=False)
    freeze_mm_mlp_adapter:   bool            = field(default=False)
    unfreeze_mm_vision_tower: bool           = field(default=False)
    model_max_length:        int             = field(default=2048)
    bits:                    int             = field(default=16)
    lora_enable:             bool            = False
    lora_r:                  int             = 64
    lora_alpha:              int             = 16
    lora_dropout:            float           = 0.05
    lora_weight_path:        str             = ""
    lora_bias:               str             = "none"
    mm_projector_lr:         Optional[float] = None
    diff_head_lr:            Optional[float] = None
    group_by_modality_length: bool           = field(default=False)
    mm_vision_tower_lr:      Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# COCO Annotation 로더
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_n_objects(annotation_path: str) -> Dict[str, int]:
    """
    instances_train2017.json에서 image_id별 object 수를 카운트.

    Dataset.__init__에서 1회만 호출. Worker 수만큼 복사되지 않도록
    최대한 가볍게 dict만 반환.

    Returns:
        {str(image_id): n_objects} - image_id를 str로 통일 (파일명 파싱과 맞추기 위해)
    """
    if annotation_path is None or not os.path.exists(annotation_path):
        logger_module.warning(
            f"[OC] COCO annotation 파일을 찾을 수 없음: {annotation_path}. "
            "n_objects=0으로 대체합니다."
        )
        return {}

    logger_module.info(f"[OC] COCO annotation 로드 중: {annotation_path}")
    n_objects_map: Dict[str, int] = {}

    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    for ann in coco.get('annotations', []):
        img_id = str(ann['image_id'])
        n_objects_map[img_id] = n_objects_map.get(img_id, 0) + 1

    logger_module.info(
        f"[OC] COCO annotation 로드 완료: {len(n_objects_map)}개 이미지, "
        f"평균 {sum(n_objects_map.values()) / max(len(n_objects_map), 1):.1f} objects/image"
    )
    return n_objects_map


# ─────────────────────────────────────────────────────────────────────────────
# Dataset - COCO Only (Text 없음)
# ─────────────────────────────────────────────────────────────────────────────

class COCOReconstructionDataset(Dataset):
    """
    COCO 이미지 전용 재구성 데이터셋.

    - Text 없음: conversations 무시, 이미지만 로드
    - n_objects: instances_train2017.json에서 image_id별 annotation 수
    - 각 샘플: {image_tensor, n_objects, image_id, raw_image_path}

    data_path 형식:
        각 줄이 JSON인 jsonl 파일.
        {"image": "000000391895.jpg"} 또는
        {"image": "000000391895.jpg", "image_id": "391895"}
        image_id 없으면 파일명에서 숫자 파싱.
    """

    def __init__(
        self,
        data_path:     str,
        data_args:     DataArguments,
        model_configs  = None,
    ):
        super().__init__()
        self.data_path     = data_path
        self.data_args     = data_args
        self.model_configs = model_configs

        # offset 인덱스 구축 (random access용)
        self._build_offset_index()

        # COCO n_objects 매핑 로드 (1회)
        coco_ann_path = getattr(data_args, 'coco_annotation_path', None)
        self.n_objects_map = load_coco_n_objects(coco_ann_path)

        logger_module.info(
            f"[COCODataset] {self.length}개 샘플 로드. "
            f"n_objects_map size: {len(self.n_objects_map)}"
        )

    def _build_offset_index(self):
        self.offsets = []
        with open(self.data_path, "rb") as f:
            off = 0
            for line in f:
                line = line.strip()
                if line:  # 빈 줄 스킵
                    self.offsets.append(off)
                off += len(line) + 1  # +1 for newline
        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    @property
    def modality_lengths(self):
        """LengthGroupedSampler용: 모두 이미지 샘플이므로 양수 고정값."""
        return [256] * self.length  # tokens_per_image

    def _parse_image_id(self, dat: dict, image_path: str) -> str:
        """
        image_id 추출.
        json에 명시된 경우 우선, 없으면 파일명에서 숫자 파싱.
        """
        if 'image_id' in dat:
            return str(dat['image_id'])
        # 파일명에서 숫자만 추출: "000000391895.jpg" → "391895"
        basename = os.path.splitext(os.path.basename(image_path))[0]
        digits = re.sub(r'^0+', '', re.sub(r'\D', '', basename))
        return digits if digits else basename

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self._getitem_(i)
        except Exception as e:
            logger_module.warning(f"인덱스 {i} 로드 오류: {e}")
            import random
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _getitem_(self, i) -> Dict[str, torch.Tensor]:
        with open(self.data_path, "rb") as f:
            f.seek(self.offsets[i])
            line = f.readline()
        dat = json.loads(line)

        # ── 이미지 경로 파싱 ─────────────────────────────────────
        image_file = dat.get('image', dat.get('source_path', ''))
        if not image_file:
            raise ValueError(f"샘플 {i}에 image 필드 없음: {dat}")

        image_folder = self.data_args.image_folder or ''
        full_path = (
            image_file if os.path.isabs(image_file)
            else os.path.join(image_folder, image_file)
        )

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"이미지 파일 없음: {full_path}")

        image = Image.open(full_path).convert('RGB')

        # ── image_id 및 n_objects 추출 ────────────────────────────
        image_id = self._parse_image_id(dat, image_file)
        oc_max_slots = getattr(self.model_configs, 'oc_max_slots', 10)

        # instances_train2017.json 기반 n_objects
        raw_n_objects = self.n_objects_map.get(image_id, 0)
        # EOS 슬롯 위치를 고려해 max_slots-1로 클리핑
        n_objects = min(raw_n_objects, oc_max_slots - 1)

        if raw_n_objects == 0:
            logger_module.debug(
                f"[OC] image_id={image_id} annotation 없음 (n_objects=0). "
                "EOS가 첫 슬롯에 올 것."
            )

        # ── 이미지 전처리 ─────────────────────────────────────────
        processor_aux_list = self.data_args.image_processor_aux_list
        processor_aux = processor_aux_list[0]

        if self.data_args.image_aspect_ratio == 'square':
            image_tensor = processor_aux.preprocess(
                image, return_tensors='pt'
            )['pixel_values'][0]  # (3, H, W)
        else:
            raise NotImplementedError(
                f"image_aspect_ratio={self.data_args.image_aspect_ratio} "
                "는 COCO OC 모드에서 미지원. 'square' 사용."
            )

        # ── vision_token_indices 구성 ─────────────────────────────
        tokens_per_image = self.data_args.vision_tower_aux_token_len_list[0]
        T       = self.data_args.max_images_per_sample * tokens_per_image
        PAD_VAL = T + 1
        vti = torch.full((T,), PAD_VAL, dtype=torch.long)
        vti[:tokens_per_image] = torch.arange(tokens_per_image, dtype=torch.long)
        vision_token_indices = vti.sort()[1]

        # ── input_ids: 이미지 토큰만으로 구성 ────────────────────
        # Text 없음 → IMAGE_TOKEN_INDEX × tokens_per_image
        input_ids = torch.full(
            (tokens_per_image,), IMAGE_TOKEN_INDEX, dtype=torch.long
        )
        # labels: 전부 IGNORE (vision loss만 사용)
        labels = torch.full(
            (tokens_per_image,), IGNORE_INDEX, dtype=torch.long
        )

        # ── 이미지 패딩 (max_images_per_sample 기준) ─────────────
        image_aux_padded = torch.zeros(
            self.data_args.max_images_per_sample,
            3,
            processor_aux.crop_size['height'],
            processor_aux.crop_size['width'],
        )
        image_aux_padded[0] = image_tensor

        # pseudo_img_tokens (padding용)
        pseudo_img_tokens = torch.full(
            ((self.data_args.max_images_per_sample - 1) * tokens_per_image,),
            IMAGE_TOKEN_INDEX,
            dtype=torch.long,
        )

        return {
            'input_ids':            input_ids,
            'labels':               labels,
            'image_aux_list':       [image_aux_padded],
            'vision_token_indices': vision_token_indices,
            'pseudo_img_tokens':    pseudo_img_tokens,
            'n_objects':            torch.tensor(n_objects, dtype=torch.long),
            'image_id':             image_id,
            'image_size':           image.size,  # (W, H)
            'has_image':            True,
            'has_video':            False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorForCOCODataset:
    """
    COCO OC 전용 Collator.

    핵심 변경:
      - n_objects: (B,) long tensor
      - raw_images: (B, 3, H, W) float tensor
        → model.forward에서 vision tower로 encode → gt_image_features
      - text 없으므로 tokenizer pad 불필요하지만 호환성 유지
    """

    tokenizer:             transformers.PreTrainedTokenizer
    max_images_per_sample: int  = 1
    tokens_per_image:      int  = 256

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        B = len(instances)

        # ── input_ids / labels ───────────────────────────────────
        # Text 없음: 모두 이미지 토큰만
        max_len = max(len(inst['input_ids']) for inst in instances)

        input_ids_list, labels_list = [], []
        for inst in instances:
            ids = inst['input_ids']
            lbl = inst['labels']
            # pad to max_len
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                lbl = torch.cat([lbl, torch.full((pad_len,), IGNORE_INDEX)])
            input_ids_list.append(ids)
            labels_list.append(lbl)

        input_ids      = torch.stack(input_ids_list)
        labels         = torch.stack(labels_list)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # ── vision_token_indices ─────────────────────────────────
        vti_list = [inst['vision_token_indices'] for inst in instances]
        vision_token_indices = torch.stack(vti_list)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            vision_token_indices=vision_token_indices,
        )

        # ── images ───────────────────────────────────────────────
        if 'image_aux_list' in instances[0]:
            # image_aux_list: list of [padded_tensor(max_imgs, 3, H, W)]
            image_aux_list = [inst['image_aux_list'] for inst in instances]
            image_aux_list = [list(x) for x in zip(*image_aux_list)]
            batch['images'] = torch.cat(image_aux_list[0], dim=0)  # (B*max_imgs, 3, H, W)

        # ── n_objects ────────────────────────────────────────────
        if 'n_objects' in instances[0]:
            batch['n_objects'] = torch.stack(
                [inst['n_objects'] for inst in instances]
            )  # (B,)

        # ── answer_img_mask / reverse_vti ────────────────────────
        # OC + COCO-only: 모든 이미지가 "answer" (재구성 대상)
        Mmax = self.max_images_per_sample
        P    = self.tokens_per_image
        Tmax = Mmax * P
        Lmax = max_len

        ans_img_list, rev_vti_list = [], []
        for ids, vti in zip(input_ids, vision_token_indices):
            # 모든 이미지가 answer
            ans_img_mask = torch.ones(Mmax, dtype=torch.bool)
            ans_img_mask[1:] = False  # max_imgs_per_sample=1이면 [True]
            ans_img_list.append(ans_img_mask)

            # reverse_vti
            rev_vti = torch.arange(Tmax, dtype=torch.long)
            img_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
            n_img_toks = img_pos.numel()
            if n_img_toks > 0:
                patch_rows = (vti[img_pos] - Lmax).clamp(0, Tmax - 1)
                rev_vti[patch_rows] = Tmax + img_pos
            rev_vti_list.append(rev_vti)

        batch['answer_img_mask'] = torch.stack(ans_img_list)    # (B, Mmax)
        batch['reverse_vti']     = torch.stack(rev_vti_list)    # (B, Tmax)
        batch['answer_token_mask'] = (input_ids == IMAGE_TOKEN_INDEX)  # (B, L)

        return batch


# ─────────────────────────────────────────────────────────────────────────────
# LengthGroupedSampler
# ─────────────────────────────────────────────────────────────────────────────

class LengthGroupedSampler(Sampler):
    def __init__(self, batch_size, world_size, lengths=None, generator=None):
        if lengths is None:
            raise ValueError("lengths 필수")
        self.batch_size  = batch_size
        self.world_size  = world_size
        self.lengths     = lengths
        self.generator   = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices    = torch.randperm(len(self.lengths), generator=self.generator)
        mega_size  = self.world_size * self.batch_size
        megabatches = [indices[i:i+mega_size].tolist() for i in range(0, len(indices), mega_size)]
        megabatches = [sorted(mb, key=lambda x: self.lengths[x], reverse=True) for mb in megabatches]
        return iter([i for mb in megabatches for i in mb])


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ScaleRAETrainer(Trainer):
    """
    Scale-RAE 커스텀 트레이너 (COCO OC 전용).

    compute_loss:
      - n_objects를 batch에서 꺼내 model.forward에 전달
      - gt_image_features는 model 내부에서 자동 생성 (_oc_gt_cache 경유)
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=self.train_dataset.modality_lengths,
            )
        return super()._get_train_sampler()

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        OC 모드:
          - n_objects: batch에서 추출해 model.forward에 전달
          - gt_image_features: None 전달 → model 내부 _oc_gt_cache 사용
            (scale_rae_arch.py의 prepare_inputs_labels_for_multimodal에서 캐싱됨)
        """
        n_objects = inputs.pop("n_objects", None)

        # gt_image_features는 model.forward 내부에서 자동 처리
        # (prepare_inputs_labels_for_multimodal → _oc_gt_cache)
        outputs = model(
            **inputs,
            n_objects=n_objects,
            gt_image_features=None,  # 내부에서 _oc_gt_cache로 처리
        )

        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        if self.args.diff_head_lr is not None:
            diff_head_params = [n for n, _ in opt_model.named_parameters() if "diff_head" in n]
            oc_params = [
                n for n, _ in opt_model.named_parameters()
                if any(k in n for k in [
                    "slot_token_emb", "slot_eos_detector",
                    "slot_aggregator", "null_slot_token"
                ])
            ]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_params
                        and n not in diff_head_params
                        and n not in oc_params
                        and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_params
                        and n not in diff_head_params
                        and n not in oc_params
                        and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_params
                        and (n in diff_head_params or n in oc_params)
                        and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.diff_head_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_params
                        and (n in diff_head_params or n in oc_params)
                        and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.diff_head_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_params and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_params and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            logs["loss"]          = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4
            )
            logs["learning_rate"] = self._get_learning_rate()

            # OC 전용 손실 로깅
            if hasattr(model, 'loss_image_diff') and model.loss_image_diff is not None:
                logs["loss_fm"]  = round(model.loss_image_diff.item(), 4)
            if hasattr(model, 'oc_eos_loss') and model.oc_eos_loss is not None:
                logs["loss_eos"] = round(model.oc_eos_loss.item(), 4)
            if hasattr(model, 'oc_div_loss') and model.oc_div_loss is not None:
                logs["loss_div"] = round(model.oc_div_loss.item(), 4)
            if hasattr(model, 'loss_language') and model.loss_language is not None:
                logs["loss_language"] = round(model.loss_language.item(), 4)

            tr_loss -= tr_loss
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# ─────────────────────────────────────────────────────────────────────────────
# Data Module 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_supervised_data_module(
    tokenizer:     transformers.PreTrainedTokenizer,
    data_args:     DataArguments,
    model_configs,
) -> Dict:

    train_dataset = COCOReconstructionDataset(
        data_path=data_args.data_path,
        data_args=data_args,
        model_configs=model_configs,
    )

    tpi = (
        model_configs.vision_tower_aux_token_len_list[0]
        if hasattr(model_configs, 'vision_tower_aux_token_len_list')
        else 256
    )

    data_collator = DataCollatorForCOCODataset(
        tokenizer=tokenizer,
        max_images_per_sample=getattr(data_args, 'max_images_per_sample', 1),
        tokens_per_image=tpi,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main train()
# ─────────────────────────────────────────────────────────────────────────────

def find_all_linear_names(model):
    cls = nn.Linear
    lora_names = set()
    mm_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(kw in name for kw in mm_keywords):
            continue
        if isinstance(module, cls):
            parts = name.split('.')
            lora_names.add(parts[0] if len(parts) == 1 else parts[-1])
    lora_names.discard('lm_head')
    return list(lora_names)


def train(attn_implementation=None):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ── Config 로드 ───────────────────────────────────────────────
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    config.vision_loss                  = model_args.vision_loss
    config.vision_loss_mode             = model_args.vision_loss_mode
    config.vision_coef                  = model_args.vision_coef
    config.diffusion_model_hidden_size  = model_args.diffusion_model_hidden_size
    config.diffusion_model_channels     = model_args.diffusion_model_channels
    config.diffusion_model_depth        = model_args.diffusion_model_depth
    config.diffusion_model_heads        = model_args.diffusion_model_heads
    config.diffusion_model_z_channels   = model_args.diffusion_model_z_channels
    config.dit_cls                      = model_args.dit_cls

    # OC 설정
    config.use_object_centric = model_args.use_object_centric
    config.oc_max_slots       = model_args.oc_max_slots
    config.oc_d_model         = config.hidden_size

    # ── 모델 로드 ─────────────────────────────────────────────────
    model = ScaleRAEQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token    = "<|endoftext|>"
    tokenizer.pad_token_id = 151643

    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        model_args.version, conversation_lib.conv_templates["qwen_2"]
    )

    # ── Vision modules 초기화 ─────────────────────────────────────
    if model_args.vision_tower_aux_list is not None:
        model_args.vision_tower_aux_list           = json.loads(model_args.vision_tower_aux_list)
        model_args.vision_tower_aux_token_len_list = json.loads(model_args.vision_tower_aux_token_len_list)
        model_args.unfreeze_mm_vision_tower        = training_args.unfreeze_mm_vision_tower

        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        model.load_vision_head(model_args=model_args)

        vision_tower_aux_list = model.get_vision_tower_aux_list()
        if not training_args.unfreeze_mm_vision_tower:
            for vt in vision_tower_aux_list:
                vt.to(dtype=compute_dtype)

        data_args.image_processor_aux_list = [vt.image_processor for vt in vision_tower_aux_list]
        data_args.is_multimodal            = True

        # vision_tower_aux_token_len_list을 data_args에도 저장
        data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list

        model.config.image_aspect_ratio              = data_args.image_aspect_ratio
        model.config.tokenizer_model_max_length      = tokenizer.model_max_length
        model.config.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.si_token_len                    = model_args.si_token_len
        model.config.miv_token_len                   = model_args.miv_token_len
        model.config.use_object_centric              = model_args.use_object_centric
        model.config.oc_max_slots                    = model_args.oc_max_slots

    # ── COCO annotation 경로 전달 ─────────────────────────────────
    data_args.coco_annotation_path = getattr(model_args, 'coco_annotation_path', None)

    # ── Freeze / Trainable 설정 ───────────────────────────────────
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if model_args.tune_adapter_and_vision_head:
        model.requires_grad_(False)
        tune_keywords = [
            'mm_projector', 'vision_head', 'diff_head', 'latent_queries',
            'slot_token_emb', 'slot_eos_detector', 'slot_aggregator', 'null_slot_token',
        ]
        for name, param in model.named_parameters():
            if any(kw in name for kw in tune_keywords):
                param.requires_grad = True

    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if 'mm_projector' in name:
                param.requires_grad = True

    # LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if model_args.mm_use_im_start_end:
        vocab = tokenizer.get_vocab()
        if DEFAULT_IM_START_TOKEN in vocab:
            model.im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            model.im_end_id   = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

    # ── Data module ──────────────────────────────────────────────
    data_module = make_supervised_data_module(tokenizer, data_args, model.config)

    # ── Trainer ─────────────────────────────────────────────────
    trainer = ScaleRAETrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    resume = training_args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        from peft import get_peft_model_state_dict
        state_dict = get_peft_model_state_dict(model)
        if training_args.local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    else:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()