
import os
import sys
import pytz
import json
import torch
from torch import nn
import shutil
import pathlib
import time
import pickle
import logging
import string
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Mapping, Iterable, Union
import transformers
import jsonlines


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('Asia/Shanghai')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def rank0_print(local_rank,*args):
    if local_rank == 0:
        print(*args)


# def cal_params(model: nn.Module):
#     params = sum(p.numel() for p in model.parameters())



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def write2txt(fp,info,mode='a'):
    with open(fp,mode=mode) as f:
        f.write(info)
        f.write('\n')


def prepare_sample(data: Union[torch.Tensor, Any], device='cuda', dtype=None) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_sample(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_sample(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        if dtype is not None:
            data = data.to(dtype)
        # if isinstance(data.dtype,torch.FloatTensor) and dtype is not None:
        #     # kwargs.update({"dtype": dtype})
        #     data = data.to(dtype=dtype)
        return data.to(**kwargs)
    return data


def nested_to_dtype(data: Union[torch.Tensor, Any], dtype) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_sample(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_sample(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(dtype)
    return data


def write2json(fp,dict_data,mode='a'):
    with jsonlines.open(fp,mode=mode) as f:
        f.write(dict_data)


from moviepy.editor import *
def split_video(video_path, start_time, end_time, output_path, size = None):
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    if size is not None:
        clip = clip.resize(newsize=size)
    clip.write_videofile(output_path)


def plot_rectangle_on_image(image,box_ratio,save_path):
    import cv2
    if isinstance(image,str):
        image = cv2.imread(image)
    h,w,_ = image.shape
    # print('h: ',h,' w:',w)
    top_left = (int(box_ratio[0]*w),int(box_ratio[1]*h))
    bottom_right = (int(box_ratio[2]*w),int(box_ratio[3]*h))
    # print(top_left,bottom_right)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imwrite(save_path,image)

from PIL import Image,ImageOps
def pad_pil_image(image):
    width, height = image.size
    max_size = max(width, height)
    pad_width = (max_size - width) // 2
    pad_height = (max_size - height) // 2
    padded_image = ImageOps.expand(image, (pad_width, pad_height, pad_width, pad_height), fill="black")
    # resized_image = padded_image.resize(size)
    return padded_image

import cv2
def pil_image_to_cv2(pil_image):
    cv2_image = np.array(pil_image)
    cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_RGB2BGR)
    return cv2_image


from moviepy.editor import concatenate_videoclips, VideoFileClip
def merge_shot_video_list(shot_video_path_list, output_path):
    clips = []
    for video_file in shot_video_path_list:
        clip = VideoFileClip(video_file)
        clips.append(clip)
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path)


import jsonlines
def rewrite_jsonl(file_path, output_path = None):
    result = []
    with jsonlines.open(file_path,'r') as f:
        for i, sample in enumerate(f):
            dict_data = {'idx':i}
            keys = sample.keys()
            for key in keys:
                dict_data[key] = sample[key]
            result.append(dict_data)
    
    if output_path is None:
        output_path = file_path[:-6] + '.json'
    with open(output_path,'w') as f:
        f.write(json.dumps(result,indent=4,ensure_ascii=False))

