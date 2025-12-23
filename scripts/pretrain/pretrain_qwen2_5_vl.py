import os,sys
sys.path.append(os.getcwd())
import logging
import torch
import json
from dataclasses import asdict
from os.path import join
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')
import pathlib
from transformers import HfArgumentParser
from configs.config import ModelArgs,TrainingArgs,DataArgs
from dataset.pretrain_dataset import PretrainDataset,DataCollatorForPretrainDataset
from dataset.audio_processor import AudioProcessor
# from dataset.video_processor import VideoProcessor,ImageProcessor
from scripts.pretrain.trainer import LongVideoTrainer

from utils.util import (
    set_seed,
    find_all_linear_names,
    rank0_print,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    write2txt
)

local_rank=None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def main(attn_implementation=None):

    global local_rank
    set_seed(42)

    parser = HfArgumentParser([ModelArgs, DataArgs, TrainingArgs])
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir
    save_config = {
        'model_args':asdict(model_args),
        'data_args':asdict(data_args),
        'training_args':asdict(training_args),
    }
    os.makedirs(output_dir,exist_ok=True)
    with open(join(output_dir,'saved_config.json'),'w') as f:
        f.write(json.dumps(save_config,indent=4))
    
    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.bf16:
        compute_dtype = torch.bfloat16
    elif training_args.fp16:
        compute_dtype = torch.float16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    from models.qwen2_5_vl.movie_qwen2_5 import MovieForCausalLM
    model = MovieForCausalLM.from_pretrained(
        pretrain_model_name_or_path,
        torch_dtype = compute_dtype,
    )
    model.config.use_cache = False
    
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=['q_proj','k_proj','v_proj','o_proj'],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print(local_rank, "Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    from transformers import Qwen2_5_VLProcessor
    processor = Qwen2_5_VLProcessor.from_pretrained(pretrain_model_name_or_path)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
    
    model.get_model().init_multimodal_modules(d_model=3584,audio_branch=model_args.audio_branch,
                                              visual_branch=model_args.visual_branch)
    if model_args.audio_branch:
        model.get_model().audio_encoder.requires_grad_(False)
    model.visual.requires_grad_(False)
    rank0_print('init multimodal finished...')
    
    ori_token_nums = len(tokenizer)
    added_tokens = ['<|audio_start|>','<|audio_pad|>','<|audio_end|>']
    num_new_tokens = tokenizer.add_tokens(added_tokens,special_tokens=True)
    MM_token_nums = len(tokenizer)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    mm_tokens = ['<|audio_pad|>','<|image_pad|>','<|video_pad|>']
    mm_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[0] for text in mm_tokens]
    model.get_model().image_token_id = mm_ids[1]
    model.get_model().audio_token_id = mm_ids[0]
    model.get_model().video_token_id = mm_ids[2]
    rank0_print('ori token nums: ',ori_token_nums,' MM token nums: ',MM_token_nums)
    
    save_modules = training_args.save_modules
    rank0_print(f'save_modules: {save_modules}')
    matched_keys = save_modules.split(',')
    for name, param in model.named_parameters():
        requ = False
        for key in matched_keys:
            if key in name:
                requ = True
                break
        param.requires_grad_(requ)
    
    if local_rank == 0:
        write2txt(fp=join(output_dir,'model.txt'),info=str(model),mode='w')
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad==True:
                write2txt(fp=join(output_dir,'model_trainable_params.txt'),info=name + '  ' + str(param.shape))
                params.append(param.numel())
        trainable_params = sum(params) / 1e6
        rank0_print(f'trainable_params: {trainable_params}MB')
        write2txt(fp=join(output_dir,'model_trainable_params.txt'),info=f'trainable_params: {trainable_params}MB')
    
    audio_processor = AudioProcessor()
    dataset = PretrainDataset(audio_processor=audio_processor,image_processor=image_processor,
                              video_processor=image_processor,training_stage=training_args.training_stage,
                              mode='train', tokenizer=tokenizer)
    collator = DataCollatorForPretrainDataset(tokenizer,mode='train')
    
    trainer = LongVideoTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
        # file_logger=FileLogger(os.makedirs(training_args.log_path,exist_ok=True)),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
    if training_args.local_rank == 0:
        model.config.save_pretrained(training_args.output_dir)
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_weights.bin'))


if __name__ == "__main__":
    main()



