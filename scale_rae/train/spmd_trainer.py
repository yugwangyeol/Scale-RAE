"""
Scale-RAE Trainer (GPU 버전, Object-Centric 확장 포함)

TPU/XLA 코드 제거, GPU (CUDA) + torchrun 환경 전용.

주요 변경사항:
  - LazySupervisedDataset._getitem_: n_objects (COCO GT object 수) 추출
  - DataCollatorForSupervisedDataset.__call__: n_objects collate
  - ScaleRAETrainer: OC 손실 로깅 추가
  - ModelArguments: use_object_centric, oc_max_slots 필드 추가
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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len: int = 2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    """Qwen2 chat template 기반 전처리."""
    roles     = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)

    special_tokens = tokenizer.additional_special_tokens_ids
    im_start, im_end = special_tokens[0], special_tokens[1]

    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    unmask_tokens_idx = [198, im_start, im_end]

    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target   += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            role    = roles.get(conv.get("role", conv.get("from")), "user")
            content = conv.get("content", conv.get("value", ""))
            encoded = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_id += encoded
            target   += encoded if role == "assistant" else [IGNORE_INDEX] * len(encoded)

        for idx, eid in enumerate(input_id):
            if eid in unmask_tokens_idx:
                target[idx] = eid
            if eid == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets   = torch.tensor(targets,   dtype=torch.long)
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    # fallback
    from scale_rae.mm_utils import tokenizer_image_token
    conversations = []
    conv          = conversation_lib.default_conversation.copy()
    roles         = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(p, tokenizer, return_tensors='pt') for p in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(sources, data_args):
    if not data_args.is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


class LazySupervisedDataset(Dataset):
    """
    지연 로딩 supervised dataset.

    Object-Centric 확장:
      - 각 샘플에 'annotations' 필드가 있으면 n_objects 추출
      - n_objects: EOS 위치 결정에 사용 (COCO GT object 수)
    """

    def __init__(
        self,
        data_path:     str,
        tokenizer:     transformers.PreTrainedTokenizer,
        data_args:     DataArguments,
        model_configs  = None,
    ):
        super().__init__()
        self.tokenizer     = tokenizer
        self.data_path     = data_path
        self.data_args     = data_args
        self.model_configs = model_configs

        # offset index 구축 (빠른 random access)
        self._build_offset_index()

    def _build_offset_index(self):
        self.offsets = []
        with open(self.data_path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)
        self.length = len(self.offsets)

    def __len__(self):
        return self.length

    def _has_image(self, sample: dict) -> bool:
        return "image" in sample and str(sample['image']) not in ('', 'None', 'none', 'nan')

    def _has_video(self, sample: dict) -> bool:
        return "video" in sample and str(sample['video']) not in ('', 'None', 'none', 'nan')

    @property
    def modality_lengths(self):
        lengths = []
        with open(self.data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                has_img = self._has_image(sample)
                cur_len = sum(len(c['value'].split()) for c in sample['conversations'])
                lengths.append(cur_len if has_img else -cur_len)
        return lengths

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

        sources  = [dat]
        has_image = self._has_image(dat)
        has_video = self._has_video(dat)

        assert not (has_image and has_video), "이미지와 비디오를 동시에 사용할 수 없습니다."

        # 이미지 토큰 삽입
        if has_image or has_video:
            for source in sources:
                if DEFAULT_IMAGE_TOKEN not in json.dumps(source['conversations']):
                    source['conversations'][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + source['conversations'][0]['value']

        vision_token_len      = self.data_args.vision_tower_aux_token_len_list[0]
        processor_aux_list    = self.data_args.image_processor_aux_list
        tokens_per_image      = vision_token_len

        if has_image:
            image_file = dat['image']
            if not isinstance(image_file, list):
                image_file = [image_file]
            image_folder = self.data_args.image_folder

            images = []
            for img_path in image_file:
                full_path = img_path if os.path.isabs(img_path) else os.path.join(image_folder, img_path)
                images.append(Image.open(full_path).convert('RGB'))

            max_length = self.model_configs.tokenizer_model_max_length
            if len(images) > (max_length // vision_token_len) - 1:
                import random
                return self.__getitem__(random.randint(0, len(self) - 1))

            image_size = images[0].size

            # Square mode 처리
            image_aux_list = []
            if self.data_args.image_aspect_ratio == 'square':
                for image in images:
                    img_tensor = processor_aux_list[0].preprocess(image, return_tensors='pt')['pixel_values'][0]
                    image_aux_list.append(img_tensor)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
            )

        else:
            sources    = copy.deepcopy([e["conversations"] for e in sources])
            images     = []
            image_size = (336, 336)

        # 토크나이징
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
            )

        # labels 전체 IGNORE면 skip
        if (data_dict['labels'] != IGNORE_INDEX).sum() == 0:
            import random
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 이미지 패딩
        if has_image and self.data_args.image_aspect_ratio == 'square':
            n_imgs = len(image_aux_list)
            processor_aux = processor_aux_list[0]
            image_aux_padded = torch.zeros(
                self.data_args.max_images_per_sample,
                3,
                processor_aux.crop_size['height'],
                processor_aux.crop_size['width'],
            )
            for idx, img_t in enumerate(image_aux_list):
                image_aux_padded[idx] = img_t
            data_dict['image_aux_list'] = [image_aux_padded]

            # vision_token_indices
            T       = self.data_args.max_images_per_sample * tokens_per_image
            used    = n_imgs * tokens_per_image
            PAD_VAL = T + 1
            vti = torch.full((T,), PAD_VAL, dtype=torch.long)
            if used:
                vti[:used] = torch.arange(used, dtype=torch.long)
            data_dict["vision_token_indices"] = vti.sort()[1]

            # input_ids 재구성
            input_ids = data_dict['input_ids']
            labels    = data_dict['labels']
            img_pos   = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            max_imgs  = min(len(img_pos), self.data_args.max_images_per_sample)

            new_ids, new_lbl, last = [], [], 0
            for idx, pos in enumerate(img_pos[:max_imgs]):
                new_ids.append(input_ids[last:pos])
                new_lbl.append(labels[last:pos])
                new_ids.append(torch.full((tokens_per_image,), IMAGE_TOKEN_INDEX, dtype=input_ids.dtype))
                new_lbl.append(torch.full((tokens_per_image,), IGNORE_INDEX,       dtype=labels.dtype))
                last = pos + 1
            if last < len(input_ids):
                new_ids.append(input_ids[last:])
                new_lbl.append(labels[last:])

            data_dict['input_ids'] = torch.cat(new_ids)
            data_dict['labels']    = torch.cat(new_lbl)
            data_dict['pseudo_img_tokens'] = torch.full(
                ((self.data_args.max_images_per_sample - n_imgs) * tokens_per_image,),
                IMAGE_TOKEN_INDEX, dtype=input_ids.dtype,
            )

            # 길이 클리핑
            ml = self.model_configs.tokenizer_model_max_length
            if len(data_dict['input_ids']) > ml:
                data_dict['input_ids'] = data_dict['input_ids'][:ml]
                data_dict['labels']    = data_dict['labels'][:ml]

        elif not has_image and self.data_args.is_multimodal:
            # 이미지 없는 텍스트 전용 샘플
            processor_aux = processor_aux_list[0]
            image_aux_padded = torch.zeros(
                self.data_args.max_images_per_sample, 3,
                processor_aux.crop_size['height'], processor_aux.crop_size['width'],
            )
            data_dict['image_aux_list'] = [image_aux_padded]

            T       = self.data_args.max_images_per_sample * tokens_per_image
            PAD_VAL = T + 1
            vti = torch.full((T,), PAD_VAL, dtype=torch.long)
            data_dict["vision_token_indices"] = vti.sort()[1]
            data_dict['pseudo_img_tokens'] = torch.full(
                (self.data_args.max_images_per_sample * tokens_per_image,),
                IMAGE_TOKEN_INDEX,
            )

        data_dict['image_size'] = image_size
        data_dict['has_video']  = has_video
        data_dict['has_image']  = has_image

        # ─── Object-Centric: GT object 수 추출 ──────────────────
        use_oc       = getattr(self.model_configs, 'use_object_centric', False)
        oc_max_slots = getattr(self.model_configs, 'oc_max_slots', 10)

        if use_oc and has_image:
            # COCO annotations 필드에서 object 수 추출
            # dat 구조: {"image": ..., "conversations": ..., "annotations": [...]}
            # annotations가 없으면 0
            n_objects = len(dat.get('annotations', []))
            # max_slots - 1 로 클리핑 (마지막 슬롯이 EOS)
            n_objects = min(n_objects, oc_max_slots - 1)
            data_dict['n_objects'] = torch.tensor(n_objects, dtype=torch.long)
        else:
            data_dict['n_objects'] = torch.tensor(0, dtype=torch.long)

        return data_dict


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorForSupervisedDataset:
    """
    Supervised dataset collator (Object-Centric 확장 포함).

    OC 확장:
      - n_objects: (B,) long tensor - GT object 수
      - answer_img_mask, reverse_vti: vision loss 계산용
    """

    tokenizer:             transformers.PreTrainedTokenizer
    max_images_per_sample: int  = 1
    tokens_per_image:      int  = 256
    video_max_frames:      int  = 0
    miv_token_len:         int  = 196

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"]   for inst in instances]
        labels    = [inst["labels"]      for inst in instances]
        pseudo    = [inst.get("pseudo_img_tokens", torch.tensor([], dtype=torch.long)) for inst in instances]
        vti       = [inst.get("vision_token_indices", torch.tensor([], dtype=torch.long)) for inst in instances]

        max_len      = self.tokenizer.model_max_length
        pad_id       = self.tokenizer.pad_token_id

        def pad_or_trunc(seq, max_l, pad_val, right=True):
            seq = seq[:max_l]
            if len(seq) < max_l:
                pad = torch.full((max_l - len(seq),), pad_val, dtype=seq.dtype)
                seq = torch.cat([seq, pad]) if right else torch.cat([pad, seq])
            return seq

        input_ids = torch.stack([pad_or_trunc(t, max_len, pad_id)     for t in input_ids])
        labels    = torch.stack([pad_or_trunc(t, max_len, IGNORE_INDEX) for t in labels])
        pseudo    = torch.stack([pad_or_trunc(t, max_len, pad_id)     for t in pseudo])

        # vision_token_indices를 offset으로 변환
        token_indices_list = []
        for _ids, _vti in zip(input_ids, vti):
            _tok = torch.arange(max_len)
            n_img_toks = (_ids == IMAGE_TOKEN_INDEX).sum()
            if len(_vti) > 0:
                _tok[_ids == IMAGE_TOKEN_INDEX] = _vti[:n_img_toks] + max_len
            token_indices_list.append(_tok)

        vision_token_indices = torch.stack(token_indices_list)
        attention_mask       = input_ids.ne(pad_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            vision_token_indices=vision_token_indices,
        )

        # 이미지 처리
        if 'image_aux_list' in instances[0]:
            image_aux_list = [inst['image_aux_list'] for inst in instances]
            image_aux_list = [list(x) for x in zip(*image_aux_list)]
            if all(x.shape == image_aux_list[0][0].shape for x in image_aux_list[0]):
                batch["images"] = [torch.cat(imgs, dim=0) for imgs in image_aux_list][0]
            else:
                raise NotImplementedError("이미지 shape 불일치")

        # ── answer_img_mask, reverse_vti 구성 ───────────────────
        Lmax = max_len
        Mmax = self.max_images_per_sample
        P    = self.tokens_per_image
        Tmax = Mmax * P

        ans_tok_list  = []
        ans_img_list  = []
        rev_vti_list  = []

        for ids, lab, vti_t in zip(input_ids, labels, vision_token_indices):
            ans_img_mask = torch.zeros(Mmax, dtype=torch.bool)
            img_pos      = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
            n_img_toks   = img_pos.numel()
            n_images     = n_img_toks // P

            for img_idx in range(n_images):
                first_tok = img_pos[img_idx * P]
                if first_tok > 0 and lab[first_tok - 1] != IGNORE_INDEX:
                    ans_img_mask[img_idx] = True

            ans_tok_mask = torch.zeros(Lmax, dtype=torch.bool)
            for img_idx in range(n_images):
                if ans_img_mask[img_idx]:
                    start = img_idx * P
                    end   = start + P
                    tok   = img_pos[start:end]
                    ans_tok_mask[tok] = True

            rev_vti = torch.arange(Tmax, dtype=torch.long)
            if n_img_toks > 0:
                patch_rows = (vti_t[img_pos] - Lmax)
                rev_vti[patch_rows] = Tmax + img_pos

            ans_tok_list.append(ans_tok_mask)
            ans_img_list.append(ans_img_mask)
            rev_vti_list.append(rev_vti)

        batch['answer_token_mask'] = torch.stack(ans_tok_list)
        batch['answer_img_mask']   = torch.stack(ans_img_list)
        batch['reverse_vti']       = torch.stack(rev_vti_list)

        # ─── Object-Centric: n_objects collate ─────────────────
        if 'n_objects' in instances[0]:
            batch['n_objects'] = torch.stack([inst['n_objects'] for inst in instances])

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
    Scale-RAE 커스텀 트레이너.

    OC 확장:
      - n_objects를 batch에서 꺼내 model.forward에 전달
      - OC 손실 (fm_loss, eos_loss, div_loss) 로깅
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
        OC 모드일 때 n_objects와 gt_image_features를 model.forward에 전달.
        """
        # n_objects 추출
        n_objects = inputs.pop("n_objects", None)

        # gt_image_features: 학습 시 images_gen 또는 images에서 RAE encode 결과
        # 현재 구현: None 전달 → model.forward 내부에서 처리
        # 실제 사용 시 데이터 파이프라인에서 gt_image_features를 준비해야 함
        gt_image_features = inputs.pop("gt_image_features", None)

        outputs = model(
            **inputs,
            n_objects=n_objects,
            gt_image_features=gt_image_features,
        )

        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """파라미터 그룹별 learning rate 설정."""
        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        if self.args.diff_head_lr is not None:
            diff_head_params = [n for n, _ in opt_model.named_parameters() if "diff_head" in n]
            oc_params        = [
                n for n, _ in opt_model.named_parameters()
                if any(k in n for k in ["slot_token_emb", "slot_eos_detector", "slot_aggregator", "null_slot_token"])
            ]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in decay_params and n not in diff_head_params and n not in oc_params and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n not in decay_params and n not in diff_head_params and n not in oc_params and p.requires_grad],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in decay_params and (n in diff_head_params or n in oc_params) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.diff_head_lr,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n not in decay_params and (n in diff_head_params or n in oc_params) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.args.diff_head_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_params and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_params and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    # ── logging: OC 손실 포함 ─────────────────────────────────────

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            logs["loss"]          = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if hasattr(model, 'loss_language') and model.loss_language is not None:
                logs["loss_language"] = round(model.loss_language.item(), 4)

            # OC 전용 손실
            if hasattr(model, 'loss_image_diff') and model.loss_image_diff is not None:
                logs["loss_fm"]  = round(model.loss_image_diff.item(), 4)
            if hasattr(model, 'oc_eos_loss') and model.oc_eos_loss is not None:
                logs["loss_eos"] = round(model.oc_eos_loss.item(), 4)
            if hasattr(model, 'oc_div_loss') and model.oc_div_loss is not None:
                logs["loss_div"] = round(model.oc_div_loss.item(), 4)

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
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        model_configs=model_configs,
    )

    tpi = model_configs.vision_tower_aux_token_len_list[0] \
          if hasattr(model_configs, 'vision_tower_aux_token_len_list') else 256

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        max_images_per_sample=getattr(data_args, 'max_images_per_sample', 1),
        tokens_per_image=tpi,
        video_max_frames=getattr(data_args, 'video_max_frames', 0),
        miv_token_len=getattr(data_args, 'miv_token_len', 196),
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

    compute_dtype = (torch.float16 if training_args.fp16 else
                     (torch.bfloat16 if training_args.bf16 else torch.float32))

    # ── 모델 로드 ─────────────────────────────────────────────────
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    # vision 설정
    config.vision_loss             = model_args.vision_loss
    config.vision_loss_mode        = model_args.vision_loss_mode
    config.vision_coef             = model_args.vision_coef
    config.diffusion_model_hidden_size = model_args.diffusion_model_hidden_size
    config.diffusion_model_channels    = model_args.diffusion_model_channels
    config.diffusion_model_depth       = model_args.diffusion_model_depth
    config.diffusion_model_heads       = model_args.diffusion_model_heads
    config.diffusion_model_z_channels  = model_args.diffusion_model_z_channels
    config.dit_cls                     = model_args.dit_cls

    # ── OC 설정 적용 ──────────────────────────────────────────────
    config.use_object_centric = model_args.use_object_centric
    config.oc_max_slots       = model_args.oc_max_slots
    config.oc_d_model         = config.hidden_size   # LLM hidden_size와 동일

    model = ScaleRAEQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    # ── tokenizer ────────────────────────────────────────────────
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

    # ── vision modules 초기화 ────────────────────────────────────
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

        model.config.image_aspect_ratio           = data_args.image_aspect_ratio
        model.config.tokenizer_model_max_length   = tokenizer.model_max_length
        model.config.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        model.config.si_token_len  = model_args.si_token_len
        model.config.miv_token_len = model_args.miv_token_len
        model.config.use_object_centric = model_args.use_object_centric
        model.config.oc_max_slots       = model_args.oc_max_slots

    # ── Freeze / Trainable 설정 ───────────────────────────────────
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if model_args.tune_adapter_and_vision_head:
        model.requires_grad_(False)
        tune_keywords = [
            'mm_projector', 'vision_head', 'diff_head', 'latent_queries',
            # OC 모듈 학습 가능
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

    # ── im_start / im_end token id 저장 ─────────────────────────
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

    # 체크포인트 재개
    resume = training_args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    model.config.use_cache = True

    # 최종 저장
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