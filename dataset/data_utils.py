from __future__ import annotations
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')

from transformers import Qwen2TokenizerFast
from typing import Dict
from PIL import Image
import math
from decord import cpu
from transformers.trainer_pt_utils import LabelSmoother

import base64
import logging
import librosa
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
import numpy as np
from packaging import version
from PIL import Image
from torchvision import io, transforms
import av
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 2
FPS_MAX_FRAMES = 66666666

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor



def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)

@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    # print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend



def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes



def _read_video_torchvision(
    ele: dict,
) -> torch.Tensor:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    return video



def _read_video_decord(
    ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path,width=224,height=224, ctx=cpu(0))
    # print('11111')
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    # logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    # print('2222, idx: ',idx, '  total frames: ',total_frames)
    video = vr.get_batch(idx).asnumpy()
    # print('3333')
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video



def _read_video_pyav(ele: dict,) -> torch.Tensor:
    video_path = ele["video"]
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.width = 224
    stream.height = 224
    
    total_frames = stream.frames
    video_fps = stream.average_rate
    print('total_frames: ',total_frames,' fps: ',video_fps)
    if total_frames <= 1:
        print('path: ', video_path)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    # print('2222, idx: ', idx, '  total frames: ', total_frames)

    video = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in idx:
            img = frame.to_image()
            img = img.resize((224, 224), Image.BILINEAR)
            img = np.array(img)
            video.append(img)
    
    video = np.stack(video)
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    
    return video


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    'pyav':_read_video_pyav
}

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image] | list[torch.Tensor]:
    if isinstance(ele["video"], str):
        # video_reader_backend = get_video_reader_backend()
        video_reader_backend = 'decord'
        video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        nframes, _, height, width = video.shape
        # print('read end... shape: ',video.shape)

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float() # t, c, h, w
        t = video.shape[0]
        print(f'{ele["video"]} use frames: {t}')

        # assert not ('merged_shot_list' in ele and 'window_size' in ele), 'merged_shot_list and window_size all exist in ele.'
        if 'merged_shot_list' in ele: # for memory, split long video into some shots by merged_shot_list.
            merged_shot_list = ele['merged_shot_list']
            # print(f'{ele["video"]} use frames: {t} \n merged_shot_list: {merged_shot_list}')
            shot_video_list = []
            # print('into merged shot list...')
            for shot in merged_shot_list:
                start_frame = int(shot['st'])
                end_frame = int(shot['et']) + 1
                end_frame = min(end_frame, t)
                shot_video = video[start_frame : end_frame]
                if shot_video.shape[0] == 0: # use last frame.
                    # print('use last frame.')
                    shot_video = video[t - 1 : t]
                # print('shot_video: ',shot_video.shape)
                shot_video_list.append(shot_video)
            # print('end merged shot list')
            return shot_video_list

            # shot_video_list = []
            # for i in range(int(nframes//15)):
            #     start_frame = 15 * i
            #     end_frame = 15 * (i + 1)
            #     end_frame = min(end_frame, t)
            #     shot_video = video[start_frame : end_frame]
            #     if shot_video.shape[0] == 0: # use last frame.
            #         # print('use last frame.')
            #         shot_video = video[t - 5 : t]
            #     # print('shot_video: ',shot_video.shape)
            #     shot_video_list.append(shot_video)
            # return shot_video_list

        elif 'shot_nums' in ele:
            shot_nums = ele['shot_nums']
            shot_duration = t / shot_nums
            shot_video_list = []
            for i in range(shot_nums):
                start_sample = int(i * shot_duration)
                end_sample = int((i + 1) * shot_duration)
                end_sample = min(end_sample, t)
                shot_video = video[start_sample: end_sample]
                if shot_video.shape[0] == 0: # use last frame.
                    # print('use last frame.')
                    shot_video = video[t - 1 : t]
                shot_video_list.append(shot_video)
            return shot_video_list
        
        elif 'window_size' in ele:  # for lora, split long video into some shots by window_size.
            window_size = ele['window_size']
            shot_video_list = []
            window_nums = int(t // window_size)
            if t % window_size != 0:
                window_nums += 1
            # print('window_nums: ',window_nums)
            for i in range(window_nums):
                start_frame = i * window_size
                end_frame = (i + 1) * window_size
                end_frame = min(end_frame, t)
                shot_video = video[start_frame : end_frame]
                # print('shot_video: ',shot_video.shape)
                shot_video_list.append(shot_video)
            return shot_video_list
        
        return [video]
    else:
        ### original: process multiple image inputs
        # assert isinstance(ele["video"], (list, tuple))
        # process_info = ele.copy()
        # process_info.pop("type", None)
        # process_info.pop("video", None)
        # images = [
        #     fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
        #     for video_element in ele["video"]
        # ]
        # nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        # if len(images) < nframes:
        #     images.extend([images[-1]] * (nframes - len(images)))
        # return images

        ### update: process multiple shot video inputs
        assert isinstance(ele['video'], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        # print('into fetch multiple shot video: ',ele)
        shot_videos = []
        for video_path in ele['video']:
            process_info.update({'video':video_path})
            video = fetch_video(process_info)
            shot_videos.append(video)
        return shot_videos


import decord
def fetch_video_quick(ele):
    # print(ele)
    shot_videos = []
    for video_path in ele['video']:
        vr = decord.VideoReader(video_path,width=224,height=224)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        fps = 0.3
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)

        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        # print(video.shape)
        # video = transforms.functional.resize(
        #     video,
        #     [224, 224],
        #     interpolation=InterpolationMode.BICUBIC,
        #     antialias=True,
        # ).float()
        shot_videos.append(video)
    return shot_videos



from transformers import WhisperFeatureExtractor
import torchaudio.compliance.kaldi as ta_kaldi

whisper_path=''
whisper_processor = WhisperFeatureExtractor.from_pretrained(whisper_path,local_files_only=True)

def preprocess_for_beats(source: torch.Tensor,fbank_mean: float = 15.41663,fbank_std: float = 6.55582,) -> torch.Tensor:
    # source: bs,L
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0) # bs, len, 128
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank


def fetch_audio(ele: dict, sr = 16000, mono = True):
    if isinstance(ele['audio'],str): # single shot audio
        audio, sr = librosa.load(ele['audio'], sr = sr, mono = mono)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio,sil),axis=0)
        if 'merged_shot_list' in ele:
            merged_shot_list = ele['merged_shot_list']
            shot_audio_list = []
            for shot in merged_shot_list:
                start_sample = int(shot['st'] * sr)
                end_sample = int(shot['et'] * sr)
                end_sample = min(end_sample, len(audio))
                shot_audio = audio[start_sample : end_sample]
                if len(shot_audio) < sr:  # pad to 1s.
                    # print('audio pad to 1s.')
                    sil = np.zeros(sr - len(shot_audio), dtype=float)
                    shot_audio = np.concatenate((shot_audio,sil),axis=0)
                shot_audio_list.append(shot_audio)
            return shot_audio_list

        elif 'shot_nums' in ele:
            shot_nums = ele['shot_nums']
            shot_duration = len(audio) / shot_nums
            shot_audio_list = []
            for i in range(shot_nums):
                start_sample = int(i * shot_duration)
                end_sample = int((i + 1) * shot_duration)
                end_sample = min(end_sample, len(audio))
                shot_audio = audio[start_sample: end_sample]
                if len(shot_audio) < sr:  # pad to 1s.
                    # print('audio pad to 1s.')
                    sil = np.zeros(sr - len(shot_audio), dtype=float)
                    shot_audio = np.concatenate((shot_audio,sil),axis=0)
                shot_audio_list.append(shot_audio)
            return shot_audio_list
        
        elif 'window_size' in ele:
            window_size = ele['window_size']
            shot_audio_list = []
            shot_duration = window_size * sr
            shot_nums = len(audio) // shot_duration
            if len(audio) % shot_duration != 0:
                shot_nums += 1
            for i in range(shot_nums):
                start_sample = int(i * shot_duration)
                end_sample = int((i+1) * shot_duration)
                end_sample = min(end_sample, len(audio))
                shot_audio = audio[start_sample : end_sample]
                if len(shot_audio) < sr:  # pad to 1s.
                    # print('audio pad to 1s.')
                    sil = np.zeros(sr - len(shot_audio), dtype=float)
                    shot_audio = np.concatenate((shot_audio,sil),axis=0)
                shot_audio_list.append(shot_audio)
            return shot_audio_list
        
        return audio

    else:  # multiple shot audio
        assert isinstance(ele['audio'], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("audio", None)
        shot_audios = []
        for audio_path in ele['audio']:
            process_info.update({'audio':audio_path})
            audio = fetch_audio(process_info)
            shot_audios.append(audio)
        
        return shot_audios
    
    
def apply_qwen2_vl_chat_template(conv, add_generation_prompt = True, add_ids = True):
    image_count, video_count, audio_count = 0, 0, 0
    instruction = ''
    pre_role = None
    for i, message in enumerate(conv):
        if i == 0 and message['role'] != 'system':
            instruction += '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
        role = message['role']
        instruction += f'<|im_start|>{role}\n'
        if isinstance(message['content'],str):
            instruction += message["content"] + '<|im_end|>\n'
            # if pre_role is not None and role != pre_role:
            #     instruction += '<|im_end|>\n'
        elif isinstance(message['content'],list):
            for content in message['content']:
                if content['type'] == 'image':
                    image_count += 1
                    if add_ids:
                        instruction += f'Picture {image_count}: '
                    instruction += '<|vision_start|><|image_pad|><|vision_end|>'
                elif content['type'] == 'video':
                    video_count += 1
                    if add_ids:
                        instruction += f'Video {video_count}: '
                    instruction += '<|vision_start|><|video_pad|><|vision_end|>'
                elif content['type'] == 'audio':
                    audio_count += 1
                    if add_ids:
                        instruction += f'Audio {audio_count}: '
                    instruction += '<audio><|audio_pad|></audio>'
                elif content['type'] == 'text':
                    instruction += content['text']
            # if pre_role is not None and role != pre_role:
            #     instruction += '<|im_end|>\n'
            instruction += '<|im_end|>\n'

        pre_role = role

    if add_generation_prompt:
        instruction += '<|im_start|>assistant\n'
    
    return instruction


def tokenize(tokenizer, text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


### extract multimodal info from multiple conversations
def extract_mm_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    mm_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if 'image' in ele or 'audio' in ele or 'video' in ele:
                        mm_infos.append(ele)
    
    return mm_infos


def process_mm_info(conversations: list[dict] | list[list[dict]],):
    mm_infos = extract_mm_info(conversations)
    ## Read images or videos or audio
    image_inputs = []
    video_inputs = []
    audio_inputs = []
    for mm_info in mm_infos:
        if "image" in mm_info:
            mm_info.update({'resized_height':224,'resized_width':224})
            image_inputs.append(fetch_image(mm_info))
        elif "video" in mm_info:
            mm_info.update({'resized_height':224,'resized_width':224,'fps':1.0})
            video_inputs.append(fetch_video(mm_info))
            # video_inputs.append(fetch_video_quick(mm_info))
        elif 'audio' in mm_info:
            audio_inputs.append(fetch_audio(mm_info))
        else:
            raise ValueError("image, audio or video should in content.")
    
    return image_inputs, audio_inputs, video_inputs


