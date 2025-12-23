#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=16
MASTER_PORT=6666
RANK=0

llama2_ckpt_path=''
qwen2_ckpt_path=''
qwen2_vl_ckpt_path=''
dockerdata_llama2_ckpt_path=''
dockerdata_qwen2_5_vl_ckpt_path=''

# Training Arguments
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=pretrain
RUN_NAME=018-qwen2_5_vl_visual_pretrain
OUTP_DIR=results
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# --deepspeed data/deepspeed/stage3.json \
torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/pretrain/pretrain_qwen2_5_vl.py \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --deepspeed deepspeed/stage2-offload.json \
    --model_name_or_path $dockerdata_qwen2_5_vl_ckpt_path \
    --exp_desc 'pretrain qwen2.5 vl, visual' \
    --freeze_backbone True \
    --lora_enable False \
    --bits 32 \
    --lora_r 8 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --num_train_epochs 1 \
    --audio_branch False \
    --visual_branch True \
    --training_stage "visual-pretrain" \
    --save_modules visual.merger \
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
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --ddp_find_unused_parameters True \
    --run_name $RUN_NAME \

