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
import time
from torch.utils.data import DataLoader, Subset
from transformers import HfArgumentParser,Qwen2VLImageProcessor
from configs.config import ModelArgs,TrainingArgs,DataArgs, InferenceArgs
from dataset.movie_dataset import MovieDataset,DataCollatorForMovieDataset
from dataset.audio_processor import AudioProcessor

from utils.util import (
    set_seed,
    find_all_linear_names,
    rank0_print,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    write2txt,
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
        # attn_implementation="eager" # used for save attn weights.
    )
    model.config.use_cache = True
    
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
    
    ### load pretrain ckpt
    audio_pretrain_ckpt_path = training_args.audio_pretrain_ckpt_path
    visual_pretrain_ckpt_path = training_args.visual_pretrain_ckpt_path
    if training_args.lora_enable:
        audio_ckpt = torch.load(audio_pretrain_ckpt_path,map_location='cpu')
        model.model.load_state_dict(audio_ckpt,strict=False)
        if visual_pretrain_ckpt_path != 'none':
            visual_ckpt = torch.load(visual_pretrain_ckpt_path,map_location='cpu')
            model.model.load_state_dict(visual_ckpt,strict=False)
    else:
        audio_ckpt = torch.load(audio_pretrain_ckpt_path,map_location='cpu')
        model.load_state_dict(audio_ckpt,strict=False)
        if visual_pretrain_ckpt_path != 'none':
            visual_ckpt = torch.load(visual_pretrain_ckpt_path,map_location='cpu')
            model.load_state_dict(visual_ckpt,strict=False)
    rank0_print('init multimodal finished...')
    
    ### add special tokens
    ori_token_nums = len(tokenizer)
    added_tokens = ['<|audio_start|>','<|audio_pad|>','<|audio_end|>']
    num_new_tokens = tokenizer.add_tokens(added_tokens,special_tokens=True)
    MM_token_nums = len(tokenizer)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.add_tokens(['<split>'],special_tokens=True)
    mm_tokens = ['<|audio_pad|>','<|image_pad|>','<|video_pad|>','<split>']
    mm_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[0] for text in mm_tokens]
    ids_2_tokens = {ids: token for ids, token in zip(mm_ids,mm_tokens)}
    tokens_2_ids = {token: ids for ids, token in zip(mm_ids,mm_tokens)}
    model.get_model().image_token_id = tokens_2_ids['<|image_pad|>']
    model.get_model().audio_token_id = tokens_2_ids['<|audio_pad|>']
    model.get_model().video_token_id = tokens_2_ids['<|video_pad|>']
    model.get_model().tokens_2_ids = tokens_2_ids
    rank0_print('ori token nums: ',ori_token_nums,' MM token nums: ',MM_token_nums)
    
    ### load finetune ckpt
    ckpt_dir = infer_args.finetune_ckpt_dir
    device = infer_args.device
    ckpt = torch.load(join(ckpt_dir,'finetune_weights.bin'),map_location='cpu')
    # if training_args.lora_enable:
    #     model.model.load_state_dict(ckpt,strict=False)
    # else:
    #     model.load_state_dict(ckpt,strict=False)
    model.load_state_dict(ckpt,strict=False)
    device = infer_args.device
    torch.cuda.set_device(device)
    model.npu()
    model.eval()

    ### dataset
    use_memory = training_args.use_memory
    use_caption = training_args.use_caption
    question_after_shot = training_args.question_after_shot
    audio_processor = AudioProcessor()
    dataset = MovieDataset(image_processor=image_processor,audio_processor=audio_processor,
                           video_processor=image_processor,tokenizer=tokenizer,mode='test',
                           use_memory=use_memory,use_caption=use_caption, question_after_shot=question_after_shot)
    collator = DataCollatorForMovieDataset(tokenzer=tokenizer,mode='test')
    # subset_indices = list(range(0, 10))
    # subset = Subset(dataset, subset_indices)
    subset = dataset
    loader = DataLoader(subset, batch_size=1,shuffle=False,collate_fn=collator,drop_last=False)
    pabr = tqdm(total=len(loader),desc=f'infer')
    device = torch.cuda.current_device()
    for i, sample in enumerate(loader):
        batch_metadata = sample.pop('batch_metadata',[{}])
        # print(batch_metadata)
        bs = len(batch_metadata)
        sample = prepare_sample(sample,dtype = compute_dtype)
        sample.update({
            'max_new_tokens': 512,
            'use_cache':True,
            # 'output_attentions':True
        })
        # print(sample['input_ids'].shape)
        # start_memory = torch.cuda.memory_allocated(device=device)
        # start_time = time.time()
        generated_ids, input_ids = model.generate(**sample)
        # end_time = time.time()
        # end_memory = torch.cuda.memory_allocated(device=device)
        # use_memory = (end_memory - start_memory) / 1024 / 1024
        # print(f'start_memory: {start_memory/1024.1024}, end_memory: {end_memory/1024/1024}, use memory: {use_memory:.3f}GB')
        # print('time: ',(end_time - start_time))
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens = False)
        for batch_id in range(bs):
            metadata = batch_metadata[batch_id]
            predict = output_texts[batch_id]
            metadata.update({'predict':predict})
            # write2json(fp=join(ckpt_dir,'infer_longvideo_bench_2.jsonl'),dict_data=metadata)
            # print(metadata)
            print(f'===== qa:\n{metadata["qa"]}\n answer: {metadata["answer"]}\n ===== predict\n {predict}\n')
        pabr.update(1)
        break



if __name__ == "__main__":
    main()



