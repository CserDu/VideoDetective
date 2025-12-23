#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=0

llama2_ckpt_path=''
qwen2_ckpt_path=''
qwen2_vl_ckpt_path=''
vicuna_ckpt_path=''
dockerdata_llama2_ckpt_path=''
dockerdata_qwen2_5_vl_ckpt_path=''

# Training Arguments
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
RUN_NAME=002-memory
OUTP_DIR=results
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

### pretrain ckpt
audio_pretrain_ckpt_path=results/pretrain/017-qwen2_5_vl_audio_pretrain/checkpoint-7509/pretrain_weights.bin
visual_pretrain_ckpt_path=results/pretrain/018-qwen2_5_vl_visual_pretrain/checkpoint-1630/pretrain_weights.bin
### finetune ckpt
lora_qa_ckpt_dir=results/finetune/010-qwen2_5_vl-lora/checkpoint-950
lora_qa_caption_ckpt_dir=results/finetune2/003-lora_qa_caption/checkpoint-724

# --deepspeed data/deepspeed/stage3.json \
python scripts/finetune2/inference_qwen2_5_vl.py \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --model_name_or_path $dockerdata_qwen2_5_vl_ckpt_path \
    --exp_desc 'infer' \
    --freeze_backbone True \
    --lora_enable True \
    --bits 16 \
    --lora_r 8 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --device cuda:0 \
    --audio_branch True \
    --visual_branch True \
    --use_memory False \
    --use_caption True \
    --audio_pretrain_ckpt_path $audio_pretrain_ckpt_path \
    --visual_pretrain_ckpt_path 'none' \
    --finetune_ckpt_dir $lora_qa_caption_ckpt_dir \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.3 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --gradient_checkpointing False \
    --half_precision_backend "auto" \
    --use_reentrant False \
    --lr_scheduler_type "cosine" \
    --save_total_limit 10 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --ddp_find_unused_parameters True \
    --run_name $RUN_NAME \


