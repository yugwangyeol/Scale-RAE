#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Scale-RAE Object-Centric 학습 스크립트 (GPU B200 / A100)
#
# 사용법:
#   bash scripts/train_oc.sh
#
# 환경변수:
#   MODEL_PATH  : Scale-RAE pretrained 모델 경로
#   DATA_PATH   : COCO reconstruction jsonl 경로
#   OUTPUT_DIR  : 체크포인트 저장 경로
#   NUM_GPUS    : GPU 수 (default: 8)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:-nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B}"
DATA_PATH="${DATA_PATH:-/data/coco_oc_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/oc_stage1}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "===== Scale-RAE Object-Centric Training ====="
echo "Model:  $MODEL_PATH"
echo "Data:   $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs:   $NUM_GPUS"
echo "=============================================="

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    -m scale_rae.train.spmd_trainer \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --version qwen_2 \
    \
    --vision_tower_aux_list '["google/siglip-so400m-patch14-384-interp256"]' \
    --vision_tower_aux_token_len_list '[256]' \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    \
    --vision_loss diffusion-loss \
    --vision_loss_mode query \
    --vision_coef 1.0 \
    --diffusion_model_hidden_size 1152 \
    --diffusion_model_channels 1152 \
    --diffusion_model_depth 12 \
    --diffusion_model_heads 16 \
    --dit_cls DiT \
    \
    --use_object_centric True \
    --oc_max_slots 10 \
    \
    --tune_adapter_and_vision_head True \
    \
    --data_path "${DATA_PATH}" \
    --image_folder /data/coco/images \
    --image_aspect_ratio square \
    --max_images_per_sample 1 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --diff_head_lr 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --model_max_length 2048 \
    --group_by_modality_length False \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 10 \
    --report_to wandb \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    2>&1 | tee "${OUTPUT_DIR}/train.log"