import json
import ast
import os
import csv
import glob,re
import random
from os.path import join,exists
import numpy as np
import random
import librosa
# import pandas as pd
from typing import Sequence,Dict
from dataclasses import dataclass,asdict
import jsonlines
from PIL import Image,ImageOps
from decord import VideoReader

import torch
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizer, Qwen2Tokenizer
from transformers import Qwen2_5_VLImageProcessor

from dataset.audio_processor import AudioProcessor
from dataset.video_processor import ImageProcessor,VideoProcessor
from dataset.data_utils import (
    fetch_image, 
    fetch_audio, 
    fetch_video_quick, 
    fetch_video,
    process_mm_info,
    # apply_qwen2_5_vl_chat_template
)

'''
train, ASR sample nums: 680072, disgard nums: 69
train, SQA sample nums: 281037
train, SVIT sample nums: 113732
train Next-QA sample nums: 34132
train, id-aware sample nums: 60000
train, baaicaption sample nums: 8123
train, cinepile sample nums: 29983
train, ego-4d sample nums: 744
'''

'''
"Provide the bounding boxes of the mentioned objects.", 
"Include the coordinates for each mentioned object.", 
"Locate the objects with their coordinates."
'''

size = {
    'resized_height':224,
    'resized_width':224
}

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


class PretrainDataset(Dataset):

    def __init__(
        self,
        audio_processor: AudioProcessor,
        image_processor: Qwen2_5_VLImageProcessor,
        video_processor: Qwen2_5_VLImageProcessor,
        tokenizer: Qwen2Tokenizer,
        training_stage = 'stage1',
        mode = 'train',
        test_filepath = 'data/test_samples.json',
    ):
        super().__init__()

        self.audio_processor = audio_processor
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.samples = []
        self.tot = 0
        self.mode = mode
        self.training_stage = training_stage

        if training_stage == 'stage1' and self.mode == 'train':
            self.add_asr_samples(max_nums= 200 * 1000)
            # self.add_SVIT_samples()
            # self.add_baaicaption_samples()
            # self.add_Next_QA_samples()
            # self.add_ego_4d_samples()

        elif self.training_stage == 'stage2' and self.mode == 'train':
            self.add_sqa_samples(max_nums = 200 * 1000)
            self.add_ID_aware_samples()
            self.add_CinePile_samples()
            self.add_Next_QA_samples()
            self.add_ego_4d_samples()

        elif self.training_stage == 'stage1&2' and self.mode == 'train':
            # audio
            self.add_asr_samples()
            self.add_sqa_samples()
            # # image
            self.add_SVIT_samples()
            self.add_ID_aware_samples()
            # # video
            self.add_ego_4d_samples()
            self.add_Next_QA_samples()
            # self.add_baaicaption_samples()
            # audio-video
            self.add_CinePile_samples()
        
        elif self.training_stage == 'audio-pretrain' and self.mode == 'train':
            self.add_asr_samples()
            self.add_sqa_samples()
            # self.add_librispeech_samples()
        
        elif self.training_stage == 'visual-pretrain' and self.mode == 'train':
            # image
            self.add_SVIT_samples()
            self.add_ID_aware_samples()
            # video
            self.add_ego_4d_samples()
            self.add_Next_QA_samples()
            # self.add_baaicaption_samples()
            
        
        elif self.mode == 'test':
            self.add_test_samples(test_filepath)


    def add_asr_samples(self, max_nums = None):
        data_root = ''
        # max_nums = 150 * 1000
        punctuations_map = {
            '<COMMA>': ',',
            '<PERIOD>': '.',
            '<QUESTIONMARK>': '?',
            '<EXCLAMATIONPOINT>': '!'
        }
        garbage_tags = ['<SIL>','<MUSIC>','<NOISE>','<OTHER>']
        prompts = [
            "Can you transcribe the speech into a written format?",
            "Listen to the speech and write down its content.",
            "What is the content of the speech you heard?",
            "Please write down the transcription of the speech.",
            "Please transcribe the speech into a written format.",
            "Write down the content of the speech you heard.",
            "Can you write down the transcription of the speech?",
            "Put the speech into a written format.",
            "Please help me to transcribe the speech into a written format.",
            "Recognize the content of the speech you heard.",
            "Can you recognize what you heard in the speech?",
            "Recognize the speech and write it down in a written format.",
            "Listen to the speech and recognize its content.",
            "Give me the transcription of the speech you heard.",
            "Recognize the speech and give me the transcription."
        ]
        tot = 0
        disgard_nums = 0
        filenames = [f'm_chunks_{str(i).zfill(4)}_metadata.csv' for i in range(69)]
        
        for chunk_id, filename in enumerate(filenames):
            filepath = join(data_root,filename)
            with open(filepath,'r') as f:
                rows = csv.reader(f)
                for i, row in enumerate(rows):
                    if i == 0:
                        disgard_nums += 1
                        continue
                    sid, speaker, text_tn, begin_time, end_time, title, url, path, aid, source, codec, channels, md5, speaker, category = row
                    audio_path = join(data_root,f'm_chunks_{str(chunk_id).zfill(4)}', sid + '.wav')
                    for gar_tag in garbage_tags:
                        # text_tn = text_tn.replace(gar_tag,'')
                        if gar_tag in text_tn:
                            continue  # disgard
                    for punc, char in punctuations_map.items():
                        text_tn = text_tn.replace(' '+punc,char)
                    # duration = float(end_time) - float(begin_time)
                    text_tn = text_tn.capitalize()
                    prompt = random.sample(prompts,1)[0]
                    conv = [
                        {'role':'user','content':[
                            {'type':'audio','audio':audio_path},
                            {'type':'text','text':prompt}
                        ]},
                        {'role':'assistant','content':text_tn}
                    ]
                    self.samples.append({
                        'conv':conv,
                        'metadata':{'audio_path':audio_path,'text_tn':text_tn}
                    })
                    tot += 1

                    if max_nums is not None and tot >= max_nums:
                        break
            
            if max_nums is not None and tot >= max_nums:
                break
        
        print(f'{self.mode}, ASR sample nums: {tot}, disgard nums: {disgard_nums}')
        self.tot += tot


    def add_librispeech_samples(self):
        data_root = ''
        train_files = ['train-clean-100.json','train-clean-360.json','train-other-500.json']
        tot = 0
        for file in train_files:
            with open(join(data_root,file),'r') as f:
                samples = json.load(f)
            for sample in samples:
                split = sample['split']
                reader_id = sample['reader_id']
                chapter_id = sample['chapter_id']
                filename = sample['filename']
                content = sample['content']

                audio_path = join(data_root,split,reader_id,chapter_id,reader_id+'-'+chapter_id+'-'+filename+'.flac')
                conv = [
                    {'role':'user','content':[
                        {'type':'audio','audio':audio_path},
                        {'type':'text','text':' Listen to the speech and recognize its content.'}
                    ]},
                    {'role':'assistant','content':content.lower()+'.'}
                ]
                self.samples.append(
                    {
                        'conv':conv,
                        'metadata':{'audio_path':audio_path},
                    }
                )
                tot += 1
        
        print(f'librispeech sample nums: {tot}')
        self.tot += tot


    def add_sqa_samples(self, max_nums = None):
        # max_nums = 150 * 1000
        data_root = ''
        with open('','r') as f:
            data = json.load(f)
        samples = data['annotation']
        tot = 0
        for sample in samples:
            path = sample['path']
            text = sample['text']
            task = sample['task']
            if task == 'QA' and 'LibriSpeech' in path:
                question = sample['Q']
                audio_path = data_root + path
                answer = text.capitalize()
                conv = [
                    {'role':'user','content':[
                        {'type':'audio','audio':audio_path},
                        {'type':'text','text':question}
                    ]},
                    {'role':'assistant','content':answer}
                ]
                self.samples.append({
                    'conv':conv,
                    'metadata':{'audio_path':audio_path,'question':question,'answer':answer}
                })
                tot += 1
                if max_nums is not None and tot >= max_nums:
                    break

        
        print(f'{self.mode}, SQA sample nums: {tot}')
        self.tot += tot


    def add_SVIT_samples(self):
        data_root = ''
        with open(join(data_root,'format-llava/SVIT_core_150K.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        for sample in samples:
            _id = sample['id']
            image = sample.get('image',None)
            if image is None:
                continue
            image_path = join(data_root,image[3:])
            conversations = sample['conversations']
            turn_nums = len(conversations)
            # assert turn_nums % 2 == 0
            if turn_nums != 2:
                continue
            key = 'content' if 'content' in conversations[0] else 'value'
            conv = []
            for i in range(turn_nums // 2):
                user_content = conversations[i][key]
                if '<image>\n' in user_content:
                    user_content = user_content.replace('<image>\n','')
                    conv.append({
                        'role':'user',
                        'content':[
                            {'type':'image','image':image_path},
                            {'type':'text','text':user_content}
                        ]
                    })
                else:
                    conv.append({
                        'role':'user',
                        'content':user_content
                    })
                
                gpt_output = conversations[i+1][key]
                conv.append({
                    'role':'assistant',
                    'content':gpt_output
                })
            
            self.samples.append({
                'conv':conv,
                'metadata':{'image_path':image_path}
            })
            tot += 1

        print(f'{self.mode}, SVIT sample nums: {tot}')
        

    def add_bunny_samples(self):
        data_root = ''
        with open(join(data_root,'bunny.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        for sample in samples:
            image = sample['image']
            image_path = join(data_root,'bunny_data',image)
            conversations = sample['conversations']
            turn_nums = len(conversations)
            assert turn_nums % 2 == 0
            key = 'content' if 'content' in conversations[0] else 'value'
            conv = []

            for i in range(turn_nums // 2):
                user_content = conversations[i][key]
                if '<image>\n' in user_content:
                    user_content = user_content.replace('<image>\n','')
                    conv.append({
                        'role':'user',
                        'content':[
                            {'type':'image','image':image_path},
                            {'type':'text','text':user_content}
                        ]
                    })
                else:
                    conv.append({
                        'role':'user',
                        'content':user_content
                    })
                
                gpt_output = conversations[i+1][key]
                conv.append({
                    'role':'assistant',
                    'content':gpt_output
                })
            
            self.samples.append({
                'conv':conv,
                'metadata':{'image_path':image_path}
            })

            tot += 1
        
        print(f'{self.mode}, bunny sample nums: {tot}')
        self.tot += tot


    def add_baaicaption_samples(self):
        data_root = ''
        with open(join(data_root,'new_baaicaption.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        invalid_video_path = [
            ''
        ]
        for sample in samples:
            video = sample['video']
            video_path = join(data_root,video[13:])
            if video_path in invalid_video_path:
                continue
            # if not exists(video_path):
            #     print('baai: ',video_path)
            conversations = sample['conversations']
            assert len(conversations) == 2
            user_content = conversations[0]['value']
            assert '<image>\n' in user_content
            user_content = user_content.replace('<image>\n','')
            gpt_output = conversations[1]['value']
            conv = [
                {'role':'user','content':[
                    {'type':'video','video':video_path},
                    {'type':'text','text':user_content}
                ]},
                {'role':'assistant','content':gpt_output}
            ]
            self.samples.append({
                'conv':conv,
                'metadata':{'video_path':video_path}
            })
            tot += 1
        print(f'{self.mode}, baaicaption sample nums: {tot}')
        self.tot += tot


    def add_ego_4d_samples(self):
        data_root = ''
        with open(join(data_root,'ego_4d.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        for sample in samples:
            video = sample['video']
            video_path = join(data_root,video.replace('vqa/',''))
            if not exists(video_path):
                print('ego-4d  ',video_path)
            conversations = sample['conversations']
            assert len(conversations) == 2
            user_content = conversations[0]['value']
            assert '<image>\n' in user_content
            user_content = user_content.replace('<image>\n','')
            gpt_output = conversations[1]['value']
            conv = [
                {'role':'user','content':[
                    {'type':'video','video':video_path},
                    {'type':'text','text':user_content}
                ]},
                {'role':'assistant','content':gpt_output}
            ]
            self.samples.append({
                'conv':conv,
                'metadata':{'video_path':video_path}
            })
            tot += 1
        
        print(f'{self.mode}, ego-4d sample nums: {tot}')
        self.tot += tot


    def add_ID_aware_samples(self):
        data_root = ''
        tot = 0
        with open(join(data_root,'beta_gpt4v_mix_mini_new.json'),'r') as f:
            samples = json.load(f)
        for sample in samples:
            id = sample['id']
            conversations = sample['conversations']
            user_content = conversations[0]['value']
            gpt_output = conversations[1]['value']

            user_content = user_content.replace('\n','')
            pattern = "<img>(.*?)</img>"
            matches = re.findall(pattern,user_content)
            image_path = []
            for match in matches:
                user_content = user_content.replace(match,'<image_pad>')
                path = join(data_root, match[2:])
                image_path.append(path)
            content_seg = user_content.split('<img><image_pad></img>')
            assert len(content_seg) == len(image_path) + 1
            conv = []
            items = []
            for i in range(len(content_seg)):
                items.append({
                    'type':'text','text':content_seg[i]
                })
                if i != len(content_seg)-1:
                    items.append({
                        'type':'image',
                        'image':image_path[i]
                    })
            
            if '<box>' in gpt_output:
                numbers = re.findall(r'\d+',gpt_output)
                numbers = [format(int(num)/1000,'.3f') for num in numbers]
                # output = re.sub(r'\d+','{}',output).format(*numbers)
                gpt_output = f'[{",".join(numbers)}]'

            conv = [
                {'role':'user','content':items},
                {'role':'assistant','content':gpt_output}
            ]
            self.samples.append({
                'conv':conv,
                'metadata':{}
            })
            tot += 1
        
        print(f'{self.mode}, id-aware sample nums: ',tot)
        self.tot += tot


    def add_Next_QA_samples(self):
        data_root = ''
        with open(join(data_root,'nextqa.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        for sample in samples:
            video = sample['video']
            video_path = join(data_root,video.replace('reasoning/next_qa/',''))
            conversations = sample['conversations']
            assert len(conversations) == 2
            question = conversations[0]['value']
            question = question.replace('<image>\n','')
            answer = conversations[1]['value']
            conv = [
                {'role':'user','content':[
                    {'type':'video','video':video_path},
                    {'type':'text','text':question}

                ]},
                {'role':'assistant','content':answer}
            ]
            self.samples.append({
                'conv':conv,
                'metadata':{'video_path':video_path,'question':question,'answer':answer}
            })
            tot += 1
        
        self.tot += tot
        print(f'{self.mode}, Next-QA sample nums: {tot}')


    def add_avsr_samples(self):
        pass

    
    def add_CinePile_samples(self):
        data_root = ''
        with open(join(data_root,'cinepine_30k.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        invalid_path = [
            
        ]
        for sample in samples:
            _id = sample['id']
            video = sample['video']
            conversations = sample['conversations']
            video_path = join(data_root,'cinepile',video.split('/')[-1])
            if video_path in invalid_path:
                continue
            # if not exists(video_path):
            #     print('cinepile: ',video_path)
            audio_path = join(data_root,'audio',video.split('/')[-1][:-4]+'.mp3')
            # if not exists(video_path) or not exists(audio_path):
            #     print('cineplei ',audio_path,video_path)
            user_content = conversations[0]['value']
            gpt_output = conversations[1]['value']
            user_content = user_content.replace('<image>\n','')
            conv = [
                {'role':'user','content':[
                    {'type':'audio','audio':audio_path},
                    {'type':'video','video':video_path},
                    {'type':'text','text':user_content}
                ]},
                {'role':'assistant','content':gpt_output}
            ]
            self.samples.append({
                'conv':conv,
                'metadata':{'audio_path':audio_path,'video_path':video_path}
            })
            tot += 1
        
        print(f'{self.mode}, cinepile sample nums: {tot}')
        self.tot += tot


    def add_test_samples(self, filepath):
        with open(filepath,'r') as f:
            samples = json.load(f)
        tot = 0
        for conv in samples:
            # print(conv)
            assert len(conv) == 2
            self.samples.append({
                'conv':conv,
                'metadata':{}
            })
            tot += 1

        self.tot += tot
        print(f'{self.mode}, sample nums: {tot}')


    def __len__(self):
        return len(self.samples)
    

    def load_image(self,image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224,224))
        return image
    

    def load_video(self,video_path):
        vr = VideoReader(video_path,width=224,height=224)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        fps = 2
        min_frames = 2
        max_frames = 768
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, 2)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        return video


    def load_audio(self,audio_path):
        audio, sr = librosa.load(audio_path, sr = 16000, mono = True)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio,sil),axis=0)
        return audio
    

    def extract_mm_info(self, conv):
        mm_infos = []
        for message in conv:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if 'image' in ele or 'audio' in ele or 'video' in ele or 'av' in ele:
                        mm_infos.append(ele)
        
        return mm_infos


    def get_mm_info(self,conv):
        mm_infos = self.extract_mm_info(conv)
        ## Read images or videos or audio
        image_inputs = []
        video_inputs = []
        audio_inputs = []
        for mm_info in mm_infos:
            if "image" in mm_info:
                mm_info.update({'resized_height':224,'resized_width':224})
                image_inputs.append(fetch_image(mm_info))
                # image_inputs.append(self.image_processor(mm_info['image'])) # c,h,w
            elif "video" in mm_info:
                mm_info.update({'resized_height':224,'resized_width':224,'fps':1.0,'max_frames':90})
                video_inputs.append(fetch_video(mm_info))
                # inputs = self.video_processor(mm_info['video'],shot_duration=30) # 30s for every seg
                # video_inputs.append(inputs) # TCHW
            elif 'audio' in mm_info:
                audio_inputs.append(fetch_audio(mm_info))
                # spec, fbank = self.audio_processor(mm_info['audio'], shot_duration=30)
                # audio_inputs.append([spec,fbank]) # spec,fbank
            elif 'av' in mm_info:
                audio_path, video_path = mm_info['av']
                spec, fbank = self.audio_processor(audio_path)
                audio_inputs.append([spec,fbank]) # spec,fbank
                video_inputs.append(self.video_processor(video_path)) # TCHW
            else:
                raise ValueError("image, audio or video should in content.")
        return image_inputs, audio_inputs, video_inputs


    def llama2_MM_tokenize(self,conv):
        turn_nums = len(conv)
        im_start = self.tokenizer('[INST]').input_ids
        im_end = self.tokenizer('[/INST]').input_ids
        nl_tokens = self.tokenizer('\n').input_ids
        _system = self.tokenizer('system').input_ids + nl_tokens
        _user = self.tokenizer('user').input_ids + nl_tokens
        _assistant = self.tokenizer('assistant').input_ids + nl_tokens
        
        system_message = "You are a helpful assistant."
        image_count, video_count, audio_count, av_count = 0, 0, 0, 0
        input_ids, output_ids = [], []
        input_text, output_text = '', ''
        
        ### system message
        system = im_start + _system + self.tokenizer(system_message).input_ids + im_end + nl_tokens
        input_ids += system
        input_text += '[INST]system\nYou are a helpful assistant.[/INST]\n'
        output_ids += [-100] * len(system)

        for i in range(turn_nums):
            ### 
            role = conv[i]['role']
            content = conv[i]['content']
            if isinstance(content,str):  ## user or gpt
                if role == 'user':
                    _input_id = im_start + _user + self.tokenizer(content).input_ids + im_end + nl_tokens
                    input_ids += _input_id
                    input_text += f'[INST]user\n{content}[/INST]\n'
                    output_ids += [-100] * len(_input_id)
                else:
                    _input_id = im_start + _assistant + self.tokenizer(content).input_ids + im_end + nl_tokens
                    input_ids += _input_id
                    input_text += f'[INST]assistant\n{content}[/INST]\n'
                    output_ids += [-100] * len(im_start) + [-100] * len(_assistant) + self.tokenizer(content).input_ids + im_end + nl_tokens
                    output_text += f'{content}[/INST]\n'
            
            elif isinstance(content,list): ## user multimodal content, including audio, video, image and text.
                _input_id = im_start + _user
                input_text += f'[INST]user\n'
                for item in content:
                    _type = item['type']
                    # if _type == 'image':
                    #     image_count += 1
                    #     _input_id += self.tokenizer(f'Image {image_count}: ').input_ids + self.tokenizer('<image>').input_ids + self.tokenizer('<image_pad>').input_ids + self.tokenizer('</image>').input_ids + nl_tokens
                    #     input_text += f'Image {image_count}: <image><image_pad></image>\n' 
                    # elif _type == 'audio':
                    #     audio_count += 1
                    #     _input_id += self.tokenizer(f'Audio {audio_count}: ').input_ids + self.tokenizer('<audio>').input_ids + self.tokenizer('<audio_pad>').input_ids + self.tokenizer('</audio>').input_ids + nl_tokens
                    #     input_text += f'Audio {audio_count}: <audio><audio_pad></audio>\n' 
                    # elif _type == 'video':
                    #     video_count += 1
                    #     _input_id += self.tokenizer(f'Video: {video_count}:').input_ids + self.tokenizer('<video>').input_ids + self.tokenizer('<video_pad>').input_ids + self.tokenizer('</video>').input_ids + nl_tokens
                    #     input_text += f'Video {video_count}: <video><video_pad></video>\n' 
                    # elif _type == 'av':
                    #     av_count += 1
                    #     _input_id += self.tokenizer(f'Audio Video: {av_count}: ').input_ids + self.tokenizer('<av>').input_ids + self.tokenizer('<av_pad>').input_ids + self.tokenizer('</av>').input_ids + nl_tokens
                    #     input_text += f'Audio Video {av_count}: <av><av_pad></av>\n' 
                    # elif _type == 'text':
                    #     _input_id += self.tokenizer(item['text']).input_ids
                    #     input_text += item['text']

                    if _type == 'text':
                        _input_id += self.tokenizer(item['text']).input_ids
                        input_text += item['text']
                    else:
                        av_count += 1
                        _input_id += self.tokenizer('<av>').input_ids + self.tokenizer('<av_pad>').input_ids + self.tokenizer('</av>').input_ids + nl_tokens
                        input_text += f'<av><av_pad></av>\n'
                
                _input_id += im_end + nl_tokens
                input_text += '[/INST]\n'
                input_ids += _input_id
                output_ids += [-100] * len(_input_id)


        assert len(input_ids) == len(output_ids)
        # print(input_ids)
        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # output_ids = torch.tensor(output_ids, dtype=torch.long)
        # attention_mask = torch.ones(input_ids.shape[0],dtype=torch.int32)
        
        return dict(
            input_ids = input_ids,
            output_ids = output_ids,
            # attention_mask = attention_mask, 
            input_text = input_text, 
            output_text = output_text
        )


    def tokenize(self,text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))


    def llama2_MM_tokenize_two_turns(self,conv, add_mm_idx = False):
        turn_nums = len(conv)
        assert turn_nums == 2
        # im_start = self.tokenizer('[INST]').input_ids
        # im_end = self.tokenizer('[/INST]').input_ids
        # nl_tokens = self.tokenizer('\n').input_ids
        # _system = self.tokenizer('system').input_ids + nl_tokens
        # _user = self.tokenizer('user').input_ids + nl_tokens
        # _assistant = self.tokenizer('assistant').input_ids + nl_tokens
        im_start = self.tokenize('[INST]')
        im_end = self.tokenize('[/INST]')
        nl_tokens = self.tokenize('\n')
        _system = self.tokenize('system') + nl_tokens
        _user = self.tokenize('user') + nl_tokens
        _assistant = self.tokenize('assistant') + nl_tokens
        
        system_message = "You are a helpful assistant."
        image_count, video_count, audio_count, av_count = 0, 0, 0, 0
        input_ids, output_ids = [], []
        input_text, output_text = '', ''
        
        ### system message
        system = im_start + _system + self.tokenize(system_message) + im_end + nl_tokens
        input_ids += [self.tokenizer.bos_token_id] + system
        input_text += '<s>[INST]system\nYou are a helpful assistant.[/INST]\n'
        # output_ids += [-100] * len(system)

        ### user content
        user_content = conv[0]['content']
        _input_id = im_start + _user
        input_text += f'[INST]user\n'
        for item in user_content:
            _type = item['type']
            if _type == 'text':
                _input_id += self.tokenize(item['text'])
                input_text += item['text']
            else:
                tag = 'av'
                if _type == 'image':
                    image_count += 1
                    prefix = f'Image {image_count}: '
                    tag = 'image'
                elif _type == 'av':
                    tag = 'av'
                    av_count += 1
                    prefix = f'Audio Video {av_count}: '
                elif _type == 'audio':
                    tag = 'audio'
                    audio_count += 1
                    prefix = f'Audio {audio_count}: '
                elif _type == 'video':
                    tag = 'video'
                    video_count += 1
                    prefix = f'Video {video_count}: '
                
                if add_mm_idx:
                    _input_id += self.tokenize(prefix) + self.tokenize(f'<{tag}>') + self.tokenize(f'<{tag}_pad>') + self.tokenize(f'</{tag}>')
                    input_text += f'{prefix}<{tag}><{tag}_pad></{tag}>'
                else:
                    _input_id += self.tokenize(f'<{tag}>') + self.tokenize(f'<{tag}_pad>') + self.tokenize(f'</{tag}>')
                    input_text += f'<{tag}><{tag}_pad></{tag}>'
        
        _input_id += im_end + nl_tokens + im_start + _assistant
        input_text += '[/INST]\n[INST]assistant\n'
        input_ids += _input_id
        
        ### gpt output
        gpt_output = conv[1]['content']
        output_ids += self.tokenize(gpt_output) + im_end + nl_tokens + [self.tokenizer.eos_token_id]
        output_text += gpt_output + '[/INST]\n</s>'

        return dict(
            input_ids = input_ids,
            output_ids = output_ids, 
            input_text = input_text, 
            output_text = output_text
        )


    def pad_resize_pil_image(self,image,size=(224,224)):
        width, height = image.size
        max_size = max(width, height)
        pad_width = (max_size - width) // 2
        pad_height = (max_size - height) // 2
        padded_image = ImageOps.expand(image, (pad_width, pad_height, pad_width, pad_height), fill="black")
        resized_image = padded_image.resize(size)
        return resized_image
    

    def conv_to_instruction(self,conv, add_mm_idx = False):
        turn_nums = len(conv)
        assert turn_nums == 2

        input_text = ''
        image_count, audio_count, video_count, av_count = 0, 0, 0, 0
        user_content = conv[0]['content']
        for item in user_content:
            _type = item['type']
            if _type == 'text':
                input_text += item['text']
            else:
                tag = 'av'
                if _type == 'image':
                    image_count += 1
                    prefix = f'Image {image_count}: '
                    tag = 'image'
                elif _type == 'av':
                    tag = 'av'
                    av_count += 1
                    prefix = f'Audio Video {av_count}: '
                elif _type == 'audio':
                    tag = 'audio'
                    audio_count += 1
                    prefix = f'Audio {audio_count}: '
                elif _type == 'video':
                    tag = 'video'
                    video_count += 1
                    prefix = f'Video {video_count}: '
                
                if add_mm_idx:
                    input_text += f'{prefix}<{tag}><{tag}_pad></{tag}>'
                else:
                    input_text += f'<{tag}><{tag}_pad></{tag}>'
        
        ### gpt output
        gpt_output = conv[1]['content']
        output_text = gpt_output

        return input_text, output_text


    def apply_qwen2_5_vl_chat_template(self, conv, image_token_nums, video_token_nums, audio_token_nums,
                                       add_generation_prompt = False, add_ids = False,):
        image_count, video_count, audio_count = 0, 0, 0
        input_ids = []
        output_ids = []
        input_text = ''
        output_text = ''
        turn_nums = len(conv)
        assert turn_nums == 2
        ### system prompt
        system_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
        input_text += system_text
        input_ids += self.tokenize(system_text)
        ### user input
        assert conv[0]['role'] == 'user'
        user_content = conv[0]['content']
        if isinstance(user_content,str):
            user_text = user_content
            input_text += user_text
            input_ids += self.tokenize(user_text)
        elif isinstance(user_content,list):
            for ele in user_content:
                if ele['type'] == 'text':
                    input_text += ele['text']
                    input_ids += self.tokenize(ele['text'])
                elif ele['type'] == 'audio':
                    audio_pad = ['<|audio_pad|>'] * audio_token_nums[audio_count]
                    audio_pad = ''.join(audio_pad)
                    user_text = f'<|audio_start|>{audio_pad}<|audio_end|>'
                    input_text += f'<|audio_start|>{audio_token_nums[audio_count]}*<|audio_pad|><|audio_end|>'
                    input_ids += self.tokenize(user_text)
                    audio_count += 1
                elif ele['type'] == 'video':
                    video_pad = ['<|video_pad|>'] * video_token_nums[video_count]
                    video_pad = ''.join(video_pad)
                    user_text = f'<|vision_start|>{video_pad}<|vision_end|>'
                    input_text += f'<|vision_start|>{video_token_nums[video_count]}*<|video_pad|><|vision_end|>'
                    input_ids += self.tokenize(user_text)
                    video_count += 1
                elif ele['type'] == 'image':
                    image_pad = ['<|image_pad|>'] * image_token_nums[image_count]
                    image_pad = ''.join(image_pad)
                    user_text = f'<|vision_start|>{image_pad}<|vision_end|>'
                    input_text += f'<|vision_start|>{image_token_nums[image_count]}*<|image_pad|><|vision_end|>'
                    input_ids += self.tokenize(user_text)
                    image_count += 1
        
        
        user_text = '<|im_start|>assistant\n'
        input_text += user_text
        input_ids += self.tokenize(user_text)

        ### assistant output
        assert conv[1]['role'] == 'assistant'
        assistant_content = conv[1]['content']
        label_text = assistant_content + '<|im_end|>\n'
        output_text += label_text
        output_ids += self.tokenize(label_text)

        return {
            'input_text':input_text,
            'output_text':output_text,
            'input_ids':input_ids,
            'output_ids':output_ids
        }


    def __getitem__(self,idx):
        sample = self.samples[idx]
        conv = sample['conv']
        metadata = sample.get('metadata',{})
        # print(conv)
        image_inputs, audio_inputs, video_inputs = self.get_mm_info(conv)
        image_token_nums, video_token_nums, audio_token_nums = 0, 0, 0
        if len(image_inputs) > 0:
            image_inputs = self.image_processor(images=image_inputs, videos=None, return_tensors = 'pt')
            image_grid_thw = image_inputs["image_grid_thw"]
            image_token_nums = [image_grid_thw[i].prod() // 4 for i in range(image_grid_thw.shape[0])]
        if len(video_inputs) > 0:
            video_inputs = self.image_processor(images=None, videos=video_inputs, return_tensors ='pt')
            video_grid_thw = video_inputs["video_grid_thw"]
            video_token_nums = [video_grid_thw[i].prod() // 4 for i in range(video_grid_thw.shape[0])]
        if len(audio_inputs) > 0:
            # spec, fbank, audio_token_nums = self.audio_processor(audio_inputs)
            # audio_inputs = [spec[0],fbank[0]]
            processed_audio = self.audio_processor(audio_inputs)
            spec_list = processed_audio['spec_list'] # [ [seg1, seg2,..], [seg1, seg2, ...] ]
            fbank_list = processed_audio['fbank_list']
            audio_token_nums = processed_audio['audio_token_nums']
            audio_inputs = [spec_list,fbank_list]
        
        output = self.apply_qwen2_5_vl_chat_template(conv,image_token_nums,video_token_nums,audio_token_nums)
        input_text = output['input_text']
        output_text = output['output_text']
        input_ids = output['input_ids']
        output_ids = output['output_ids']
        metadata.update({
            'input_text':input_text,
            'output_text':output_text
        })
        if self.mode == 'train':
            instruction_ids = input_ids + output_ids
            label_ids = [-100] * len(input_ids) + output_ids
        else:
            instruction_ids = input_ids
            label_ids = [-100] * len(input_ids)

        instruction_ids = torch.tensor(instruction_ids,dtype=torch.long)
        label_ids = torch.tensor(label_ids,dtype=torch.long)

        data = {
            'instruction_ids':instruction_ids,
            'label_ids':label_ids,
            'metadata':metadata
        }

        if len(image_inputs) > 0:
            data['image_inputs'] = image_inputs
        if len(video_inputs) > 0:
            data['video_inputs'] = video_inputs
        if len(audio_inputs) > 0:
            data['audio_inputs'] = audio_inputs
        
        return data

        # tokenize_output = self.llama2_MM_tokenize_two_turns(conv, add_mm_idx = False)
        
        # input_text, output_text = self.conv_to_instruction(conv,add_mm_idx=False)
        ### llama2 tokenizer template
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": input_text}
        # ]
        # instruction = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # output = output_text + '</s>'

        # metadata.update({
        #     'input_text':tokenize_output['input_text'],
        #     'output_text':tokenize_output['output_text']
        # })

        # instruction_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instruction))
        # output_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(output))
        
        # sample = {
        #     'input_ids':tokenize_output['input_ids'],
        #     'output_ids':tokenize_output['output_ids'],
        #     'image_inputs':image_inputs,
        #     'audio_inputs':audio_inputs,
        #     'video_inputs':video_inputs,
        #     'metadata':metadata,
        # }
        # return sample



# @dataclass
# class DataCollatorForPretrainDataset(object):
#     tokenzer: transformers.PreTrainedTokenizer
#     mode: str = 'train'
#     def __call__(self, instances: Sequence[Dict]):
        
#         tokenizer = self.tokenzer
#         mode = self.mode

#         batch_conv = [instance['conv'] for instance in instances]
#         batch_metadata = [instance['metadata'] for instance in instances]

#         batch_input_ids=[]
#         batch_labels=[]
#         batch_X_data=[]
#         batch_metadata = []
#         for instance in instances:
#             instruction_ids = instance['input_ids']
#             output_ids = instance['output_ids']
            
#             ### tokenize two turns
#             if self.mode == 'train':
#                 input_ids = instruction_ids + output_ids
#                 output_ids = [-100] * len(instruction_ids) + output_ids
#             else:
#                 input_ids = instruction_ids
#                 output_ids = [-100] * len(instruction_ids)
#             # if self.mode == 'train':
#             #     input_ids = instruction_ids + output_ids
#             #     labels = [-100] * len(instruction_ids) + output_ids
#             #     assert len(input_ids) == len(labels)
#             # else:
#             #     input_ids = instruction_ids
#             #     labels = [-100] * len(input_ids)
#             batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
#             batch_labels.append(torch.tensor(output_ids,dtype=torch.long))

#             image_inputs = instance['image_inputs']
#             audio_inputs = instance['audio_inputs']
#             video_inputs = instance['video_inputs']
            
#             batch_X_data.append(
#                 {
#                     'image':image_inputs,
#                     'audio':audio_inputs,
#                     'video':video_inputs
#                 }
#             )
#             batch_metadata.append(instance['metadata'])
        
#         data = {
#             'batch_input_ids':batch_input_ids,
#             'batch_labels':batch_labels,
#             'batch_X_data':batch_X_data,
#         }
#         if self.mode == 'test':
#             data['batch_metadata'] = batch_metadata
        
#         return data



@dataclass
class DataCollatorForPretrainDataset(object):
    tokenzer: Qwen2Tokenizer
    mode: str = 'train'
    def __call__(self, instances: Sequence[Dict]):
        tokenizer = self.tokenzer
        batch_instruction_ids = []
        batch_label_ids = []
        batch_metadata = []
        batch_attention_mask = []
        batch_image_grid_thw = []
        batch_video_grid_thw = []
        batch_pixel_values = []
        batch_pixel_values_videos = []
        batch_audio = []
        batch_X_data = []
        for instance in instances:
            instruction_ids = instance['instruction_ids']
            label_ids = instance['label_ids']
            batch_instruction_ids.append(instruction_ids)
            batch_label_ids.append(label_ids)
            
            attention_mask = torch.ones(instruction_ids.shape[0],dtype=torch.int32)
            batch_attention_mask.append(attention_mask)

            image_inputs = instance.get('image_inputs',None)
            audio_inputs = instance.get('audio_inputs',None)
            video_inputs = instance.get('video_inputs',None)
            
            # if image_inputs is not None:
            #     batch_pixel_values.append(image_inputs['pixel_values'])
            #     batch_image_grid_thw.append(image_inputs['image_grid_thw'])
            # if video_inputs is not None:
            #     batch_pixel_values_videos.append(video_inputs['pixel_values_videos'])
            #     batch_video_grid_thw.append(video_inputs['video_grid_thw'])
            # if audio_inputs is not None:
            #     batch_audio.append(audio_inputs)
            
            batch_X_data.append({
                'image':image_inputs,
                'video':video_inputs,
                'audio':audio_inputs
            })


            batch_metadata.append(instance['metadata'])
        
        input_ids = pad_sequence(batch_instruction_ids,batch_first=True,padding_value = tokenizer.pad_token_id)
        labels = pad_sequence(batch_label_ids,batch_first=True,padding_value=-100)
        attention_mask = pad_sequence(batch_attention_mask,batch_first=True,padding_value=0)
        
        # if len(batch_image_grid_thw) == 0:
        #     pixel_values = None
        #     image_grid_thw = None
        # else:
        #     pixel_values = torch.cat(batch_pixel_values,dim=0)
        #     image_grid_thw = torch.cat(batch_image_grid_thw,dim=0)
        
        # if len(batch_video_grid_thw) == 0:
        #     pixel_values_videos = None
        #     video_grid_thw = None
        # else:
        #     pixel_values_videos = torch.cat(batch_pixel_values_videos, dim=0)
        #     video_grid_thw = torch.cat(batch_video_grid_thw,dim=0)

        
        data = {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask':attention_mask,
            # 'pixel_values':pixel_values,
            # 'pixel_values_videos':pixel_values_videos,
            # 'image_grid_thw':image_grid_thw,
            # 'video_grid_thw':video_grid_thw,
            # 'batch_audio':batch_audio,
            'batch_X_data':batch_X_data,
        }
        if self.mode == 'test':
            data['batch_metadata'] = batch_metadata
        
        return data


