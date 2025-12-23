#!/bin/bash

# Environment Variables
WORLD_SIZE=4
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=3
MASTER_ADDR='11.254.18.12'

llama2_ckpt_path=''
qwen2_ckpt_path=''
qwen2_vl_ckpt_path=''
vicuna_ckpt_path=''
dockerdata_llama2_ckpt_path=''
dockerdata_qwen2_5_vl_ckpt_path=''

# Training Arguments
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune2
RUN_NAME=010-lora_qa_caption_full_video_4x8
OUTP_DIR=results
# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

audio_pretrain_ckpt_path=results/pretrain/017-qwen2_5_vl_audio_pretrain/checkpoint-7509/pretrain_weights.bin
visual_pretrain_ckpt_path=results/pretrain/018-qwen2_5_vl_visual_pretrain/checkpoint-1630/pretrain_weights.bin
pretrain_stage12_align_ckpt_path=results/pretrain/006-stage1+2_align_2/checkpoint-5464/pretrain_weights.bin


# torchrun --nproc_per_node $NPROC_PER_NODE \
#     --master_port $MASTER_PORT \
torchrun --master_addr $MASTER_ADDR \
    --node_rank $RANK \
    --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    --nnodes $WORLD_SIZE \
    scripts/finetune2/finetune_qwen2_5_vl.py \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --deepspeed deepspeed/stage2-offload.json \
    --model_name_or_path $dockerdata_qwen2_5_vl_ckpt_path \
    --exp_desc 'no visual pretrain, use caption, full video, 4x8' \
    --freeze_backbone True \
    --lora_enable True \
    --bits 16 \
    --lora_r 8 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --use_memory False \
    --use_caption True \
    --question_after_shot False \
    --audio_branch True \
    --visual_branch True \
    --audio_pretrain_ckpt_path $audio_pretrain_ckpt_path \
    --visual_pretrain_ckpt_path 'none' \
    --num_train_epochs 2 \
    --save_modules 'lora' \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.33 \
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


