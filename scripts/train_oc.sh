#!/bin/bash
# =============================================================================
# Scale-RAE Object-Centric Training Script (B200 Blackwell)
#
# [???] Scale-RAE-LAB ???? ??:
#   bash scripts/train_oc.sh
#
# [?? ??]
#   ????? Scale-RAE-LAB/ ???? ???? ?? ???? ???.
#   ../  ? Scale-RAE-LAB ? ?? ???? (?: /home/jovyan/)
#   ./   ? Scale-RAE-LAB ??
# =============================================================================

# ?? ?? ??????????????????????????????????????????????????????????????????????
MODEL_PATH="${MODEL_PATH:-../data/Scale-RAE-Qwen1.5B_DiT2.4B}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${MODEL_PATH}}"

# ?? ??? ????????????????????????????????????????????????????????????????????
DATA_PATH="${DATA_PATH:-../data/coco_oc_train.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../data/coco/train2017}"
COCO_ANNOTATION="${COCO_ANNOTATION:-../data/coco/annotations/instances_train2017.json}"

# ?? ?? ??????????????????????????????????????????????????????????????????????
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/oc_stage1}"
NUM_GPUS="${NUM_GPUS:-8}"

# ?? W&B ???????????????????????????????????????????????????????????????????????
export WANDB_PROJECT="${WANDB_PROJECT:-Scale-RAE-OC}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-oc_stage1}"

# =============================================================================

mkdir -p "${OUTPUT_DIR}"

# 환경 설정 ─────────────────────────────────────────────────────────────────
# Scale-RAE-LAB 루트를 PYTHONPATH에 추가
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# scale_rae conda 환경의 Python 명시적 사용 (환경이 활성화 안 된 경우에도 동작)
SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
if [ -f "${SCALE_RAE_ENV}/bin/python" ]; then   
    PYTHON="${SCALE_RAE_ENV}/bin/python"
elif [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    PYTHON="$(which python)"
fi
echo "Python:      ${PYTHON}"

# 로컬 pip 패키지(~/.local/lib)보다 conda 환경 패키지가 우선 적용되도록 설정
# 이 설정이 없으면 로컬에 설치된 패키지가 conda env 패키지보다 먼저 로드되어
# numpy/sklearn 바이너리 비호환 오류가 발생할 수 있음
export PYTHONNOUSERSITE=1

# conda 환경의 libstdc++ 우선 사용 (CXXABI 호환성)
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

# B200 Blackwell: NCCL ??? ??
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# =============================================================================

echo "===== Scale-RAE Object-Centric Training (B200 Blackwell) ====="
echo "Model:       $MODEL_PATH"
echo "Pretrain:    $PRETRAIN_CKPT"
echo "Data:        $DATA_PATH"
echo "Images:      $IMAGE_FOLDER"
echo "Annotation:  $COCO_ANNOTATION"
echo "Output:      $OUTPUT_DIR"
echo "GPUs:        $NUM_GPUS"
echo "W&B project: $WANDB_PROJECT  run: $EXPERIMENT_NAME"
echo "=============================================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    -m scale_rae.train.spmd_trainer \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --pretrain_adapter_and_vision_head "${PRETRAIN_CKPT}" \
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
    --diffusion_model_hidden_size 2048 \
    --diffusion_model_channels 1152 \
    --diffusion_model_z_channels 2048 \
    --diffusion_model_depth 32 \
    --diffusion_model_heads 32 \
    --dit_cls DiT \
    \
    --use_object_centric True \
    --oc_max_slots 10 \
    \
    --tune_adapter_and_vision_head True \
    \
    --coco_annotation_path "${COCO_ANNOTATION}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --image_aspect_ratio square \
    --max_images_per_sample 1 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 100 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --diff_head_lr 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 True \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --group_by_modality_length False \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --report_to wandb \
    --run_name "${EXPERIMENT_NAME}" \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

# =============================================================================
# [??] LoRA ?? ? ?? ???? torchrun ??? ??:
#
#   --lora_enable True \
#   --lora_r 64 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
# =============================================================================
