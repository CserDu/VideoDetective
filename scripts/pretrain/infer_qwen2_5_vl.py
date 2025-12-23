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
from tqdm import tqdm
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
from configs.config import ModelArgs,TrainingArgs,DataArgs, InferenceArgs
from dataset.pretrain_dataset import PretrainDataset,DataCollatorForPretrainDataset
from dataset.audio_processor import AudioProcessor
from dataset.video_processor import VideoProcessor,ImageProcessor

from utils.util import (
    set_seed,
    find_all_linear_names,
    rank0_print,
    prepare_sample,
    write2json
)

local_rank=None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def main(attn_implementation=None):

    global local_rank
    set_seed(42)

    parser = HfArgumentParser([ModelArgs, DataArgs, TrainingArgs, InferenceArgs])
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()

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

    ### load ckpt
    ckpt_dir = infer_args.ckpt_dir
    ckpt = torch.load(join(ckpt_dir,'pretrain_weights.bin'),map_location='cpu')
    model.load_state_dict(ckpt,strict=False)

    audio_pretrain_ckpt_path = 'results/pretrain/017-qwen2_5_vl_audio_pretrain/checkpoint-7509/pretrain_weights.bin'
    ckpt = torch.load(audio_pretrain_ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)

    device = infer_args.device
    torch.cuda.set_device(device)
    model.npu()
    model.eval()
    
    audio_processor = AudioProcessor()
    dataset = PretrainDataset(audio_processor=audio_processor,image_processor=image_processor,
                              video_processor=image_processor,training_stage=training_args.training_stage,
                              mode='test', tokenizer=tokenizer,test_filepath='data/test_samples.json')
    collator = DataCollatorForPretrainDataset(tokenizer,mode='test')
    loader = DataLoader(dataset=dataset,batch_size=1,shuffle=False,collate_fn=collator,drop_last=False)
    target_idx = 2
    for i, sample in enumerate(loader):
        if i != target_idx:
            continue
        # print(sample)
        # exit(-1)
        batch_metadata = sample.pop('batch_metadata',{})
        bs = len(batch_metadata)
        sample = prepare_sample(sample,dtype=compute_dtype)
        sample.update({
            'max_new_tokens': 64,
            'use_cache':True,
        })
        print(sample['input_ids'].shape)
        generated_ids = model.generate(**sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(sample['input_ids'], generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # batch_text = tokenizer.batch_decode(output,skip_special_tokens = True)
        for batch_id in range(bs):
            metadata = batch_metadata[batch_id]
            predict = output_texts[batch_id]
            metadata.update({'predict':predict})
            print(metadata)
        break


if __name__ == "__main__":
    main()


