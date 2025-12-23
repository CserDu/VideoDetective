#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=0

llama2_ckpt_path=''
qwen2_ckpt_path=''
qwen2_vl_ckpt_path=''
dockerdata_llama2_ckpt_path=''
dockerdata_qwen2_5_vl_ckpt_path=''
# Training Arguments
LOCAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=pretrain
RUN_NAME=002-visual-pretrain
OUTP_DIR=results
export CUDA_VISIBLE_DEVICES='1'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

pretrain_stage12_ckpt_dir=results/pretrain/007-stage1+2_align/checkpoint-2732
pretrain_stage2_ckpt_dir=results/pretrain/004-stage2/checkpoint-2108
pretrain_stage1_ckpt_dir=results/pretrain/003-stage1_align/checkpoint-3120
audio_pretrain_ckpt_dir=results/pretrain/017-qwen2_5_vl_audio_pretrain/checkpoint-7509
visual_pretrain_ckpt_dir=results/pretrain/018-qwen2_5_vl_visual_pretrain/checkpoint-1630

python scripts/pretrain/infer_qwen2_5_vl.py \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --model_name_or_path $dockerdata_qwen2_5_vl_ckpt_path \
    --freeze_backbone True \
    --lora_enable False \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --audio_branch True \
    --visual_branch True \
    --ckpt_dir $visual_pretrain_ckpt_dir \
    --device cuda:0 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --use_reentrant False \
    --lr_scheduler_type "cosine" \
    --save_total_limit 10 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --ddp_find_unused_parameters True \
    --run_name $RUN_NAME \
