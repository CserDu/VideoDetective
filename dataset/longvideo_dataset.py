import json,os,random
random.seed(42)
from os.path import join,exists
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Sequence,Dict
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2_5_VLProcessor
from datetime import datetime, timedelta


from dataset.data_utils2 import (
    fetch_image, 
    fetch_audio,
    fetch_video
)

char_list = [chr(ord('A') + i) for i in range(26)]


class LongVideoDataset(Dataset):
    def __init__(
        self,
        processor,
        audio_processor,
        mode='train',
        shot_size = None,
        shot_nums = None,
        min_pixels = 64 * 28 * 28,
        max_pixels = 64 * 28 * 28,
        fps = 1,
        stage1_min_duration = 2 * 60,
        stage1_max_duration = 10 * 60,
        stage2_min_duration = 10 * 60,
        stage2_max_duration = 30 * 60,
        max_frames = 512,
        training_stage = 'stage1',
    ):
        super().__init__()

        self.samples = []
        self.mode = mode
        self.tot = 0
        self.shot_size = shot_size
        self.shot_nums = shot_nums
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.stage1_min_duration = stage1_min_duration
        self.stage1_max_duration = stage1_max_duration
        self.stage2_max_duration = stage2_max_duration
        self.stage2_min_duration = stage2_min_duration
        self.image_pocessor = processor.image_processor
        self.video_processor = processor.image_processor
        self.audio_processor = audio_processor
        self.tokenizer = processor.tokenizer

        self.training_stage = training_stage

        self.vision_kwargs = {
            ### image and video
            'min_pixels': min_pixels,
            'max_pixels': max_pixels,
            ### video
            'shot_size': shot_size,
            'shot_nums': shot_nums,
            'fps': fps,
            'max_frames': max_frames
        }

        if self.mode == 'train':
            if self.training_stage == 'stage1':
                self.add_vico_data()
                self.add_cinepile_data()
                self.add_nextqa_data()
                # self.add_moviechat_data()
                self.add_longvideo_reason_data()
                self.add_perception_test_data()
                self.add_movie_wo_reason_data()
                # self.add_time_data()
                # self.add_moment_10m_data(alpha=0.2)
            elif self.training_stage == 'stage2':
                # self.add_longvideo_reason_data()
                # self.add_moviechat_data()
                # self.add_moment_10m_data()

                ### movie dataset
                self.add_movie_data(use_script=False)
    
        else:
            # self.add_longvideo_bench_data()
            self.add_LMVU_data()
            # self.add_VNBench_data()
            # self.add_nextqa_test_data()
            # self.add_test_samples()
            # self.add_video_mme_data(use_subtitles = True)
            # self.add_movie_test_data()
            # self.add_cinepile_test_data()
            # self.add_charades_sta_test_data()
            # self.add_test_samples()

            # self.add_vista_data()
            
            # self.add_mvbench_data()

        print(f'mode: {self.mode} sample nums: {self.tot}')


    ### stage1 training dataset
    def add_vico_data(self):
        cnt = 0
        data_root = ''
        with open(join(data_root, 'vico_new.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            duration = sample['duration']
            if self.training_stage == 'stage1':
                if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

            id = sample['id']
            video = sample['video']
            conversations = sample['conversations']
            video_path = join('', video.split('/')[-1])
            question = conversations[0]['value']
            output = conversations[1]['value']
            if '<image>\n' in question: question = question.replace('<image>\n','')
            elif '<image>' in question: question = question.replace('<image>','')
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question}
                ]},
                {'role':'assistant', 'content': output}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': question, 'label': output}
            })
            cnt += 1
            self.tot += 1

        print(f'vico data nums: {cnt}')


    def add_cinepile_data(self):
        cnt = 0
        data_root = ''
        with open(join(data_root, 'cinepine_30k_new.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            duration = sample['duration']
            # if duration < self.min_duration or duration > self.max_duration: continue
            # if self.training_stage == 'stage1':
            #     if duration < self.min_duration or duration > self.max_duration: continue
            # elif self.training_stage == 'stage2':
            #     if duration < 10 * 60 or duration > 30 * 60: continue
            if self.training_stage == 'stage1':
                if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

                
            id = sample['id']
            video = sample['video']
            conversations = sample['conversations']
            video_path = join(data_root, 'cinepile', video.split('/')[-1])
            
            question = conversations[0]['value']
            output = conversations[1]['value']
            if '<image>\n' in question: question = question.replace('<image>\n','')
            elif '<image>' in question: question = question.replace('<image>','')
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question}
                ]},
                {'role':'assistant', 'content': output}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': question, 'label': output}
            })
            self.tot += 1
            cnt += 1
        
        print(f'cinepile data nums: {cnt}')
    
    
    def add_nextqa_data(self):
        cnt = 0
        data_root = ''
        with open(join(data_root, 'nextqa_new.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            duration = sample['duration']
            # if duration <= 30: continue
            # if self.training_stage == 'stage1':
            #     if duration <= 30: continue
            # elif self.training_stage == 'stage2':
            #     if duration < 10 * 60 or duration > 30 * 60: continue
            if self.training_stage == 'stage1':
                if duration <= 30 or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

                
            id = sample['id']
            video = sample['video']
            conversations = sample['conversations']
            video_path = join(data_root, video.replace('reasoning/next_qa/', ''))
            question = conversations[0]['value']
            output = conversations[1]['value']
            if '<image>\n' in question: question = question.replace('<image>\n','')
            elif '<image>' in question: question = question.replace('<image>','')
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question}
                ]},
                {'role':'assistant', 'content': output}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': question, 'label': output}
            })
            self.tot += 1
            cnt += 1
        
        print(f'nextqa data nums: {cnt}')


    def add_moviechat_data(self):
        cnt = 0
        invalid_path = [
        ]
        data_root = ''
        with open(join(data_root, 'training_data.json'), 'r') as f:
            data = json.load(f)
        # print('movie-chat ori data nums: ', len(data))
        for sample in data:
            duration = sample['duration']
            
            if self.training_stage == 'stage1':
                if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

            video_path = sample['video_path']
            if not exists(video_path): continue
            if video_path in invalid_path: continue
            
            caption = sample['caption']
            global_qa = sample['global']
            breakpoint_qa = sample['breakpoint']
            flag = random.random()
            if flag > 0.3:
                selected_qa = random.sample(global_qa, k = 2)
                # selected_qa = global_qa
                for qa in selected_qa:
                    question = qa['question']
                    answer = qa['answer']
                    conv = [
                        {'role':'user', 'content': [
                            {'type': 'video', 'video': video_path},
                            {'type': 'text', 'text': question}
                        ]},
                        {'role': 'assistant', 'content': answer}
                    ]
                    self.samples.append({'conv': conv, 'metadata': {'video_path': video_path, 'question': question, 'label': answer}})
                    self.tot += 1
                    cnt += 1
            else:
                question = 'Describe this video content in detail.'
                conv = [
                    {'role':'user', 'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': question}
                    ]},
                    {'role': 'assistant', 'content': caption}
                ]
                self.samples.append({'conv': conv, 'metadata': {'video_path': video_path, 'question': question, 'label': caption}})
                self.tot += 1
                cnt += 1
    
        print(f'movie-chat data nums: {cnt}')


    def add_longvideo_reason_data(self):
        qa_cnt = 0
        open_cnt = 0
        data_root = ''
        with open(join(data_root, 'video_info.json'), 'r') as f:
            video_info = json.load(f)
        
        with open(join(data_root, 'training_data.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            problem = sample['problem']
            reasoning = sample['reasoning']
            video_path = sample['video_path']
            answer = sample['answer']
            duration = video_info.get(video_path, None)
            if duration is None: continue
            if self.training_stage == 'stage1':
                if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

            label = f'({answer})'
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': problem}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': problem, 'label': label}
            })
            self.tot += 1
            qa_cnt += 1
        
        print(f'long video reason qa data nums: {qa_cnt}')
        

        with open(join(data_root, 'openended_data.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            problem = sample['problem']
            reasoning = sample['reasoning']
            video_path = sample['video_path']
            duration = video_info.get(video_path, None)
            if duration is None: continue
            # if duration is None or duration > self.max_duration or duration < self.min_duration: continue
            # if self.training_stage == 'stage1':
            #     if duration < self.min_duration or duration > self.max_duration: continue
            # elif self.training_stage == 'stage2':
            #     if duration < self.stage2_min or duration > self.stage2_max: continue
            if self.training_stage == 'stage1':
                if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue


            answer = sample['answer']

            label = answer
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': problem}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': problem, 'label': label}
            })
            self.tot += 1
            open_cnt += 1
        
        print(f'long video reason open data nums: {open_cnt}')


    def add_perception_test_data(self):
        cnt = 0
        data_root = ''
        with open(join(data_root, 'PerceptionTest_6348_new.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            duration = sample['duration']
            # if duration <= 30: continue
            if self.training_stage == 'stage1':
                if duration <= 30 or duration > self.stage1_max_duration: continue
            elif self.training_stage == 'stage2':
                if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

            problem = sample['problem']
            options = sample['options']
            solution = sample['solution']
            video_path = sample['path']
            duration = sample['duration']
            
            options = '\n'.join(options)
            question = f'{problem}\n{options}'
            matches = re.findall(r'<answer>(.*?)</answer>', solution)
            label = matches[0].strip()
            label = f'({label})'

            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': question, 'label': label}
            })
            self.tot += 1
            cnt += 1
        
        print(f'preception test data nums: {cnt}')


    def add_movie_wo_reason_data(self):
        cnt = 0
        flag = False
        use_subtitle = True
        data_root = ''
        with open('', 'r') as f:
            data = json.load(f)
        for cid_vid, items in data.items():
            for sample in items:
                question = sample['question']
                options = sample['options']
                # answer = sample.get('answer', None)
                # if answer is None:
                #     continue
                answer = sample['answer']
                reason = sample['reason']
                clues = sample['clues']
                video_path = sample['video_path']
                start_time = sample['start_time']
                end_time = sample['end_time']
                duration = (int(end_time) - int(start_time)) / 1000
                
                if self.training_stage == 'stage1':
                    if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
                elif self.training_stage == 'stage2':
                    if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

                script_path_list = sample['script_path']
                script_data = []
                for script_path in script_path_list:
                    with open(script_path, 'r') as f:
                        script = json.load(f)
                    script_data.extend(script)
                
                options = '\n'.join(options)
                qa = f'{question}\n{options}\nYou must select the correct option, provide a detailed reasoning for your answer, and specify the time clues in seconds from the video that support your reasoning.\n'
                
                # give the correct option, the reason for the correct option, and the timestamps of crucial clues in the reason.
                # qa += 'Choose the correct option, and then output the time period where the crucial clues are located.'
                
                clues_info = []
                for clue in clues:
                    if len(clue) == 1: clues_info.append(f'{clue[0]}s')
                    else: clues_info.append(f'{clue[0]}s~{clue[1]}s')
                label = f'{answer}\nReasoning: {reason}\nClues: {clues_info}'
                # label = answer

                conv = [
                    {'role': 'user', 'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': qa}
                    ]},
                    {'role':'assistant', 'content': label}
                ]
                self.samples.append({
                    'conv': conv,
                    'metadata': {'video_path': video_path, 'question': qa, 'label': label,
                                 'start_time': start_time, 'end_time': end_time, 'script_data': script_data,
                                 'dataset_name': 'movie', 'cid_vid': cid_vid}
                })
                self.tot += 1
                cnt += 1
            
        print(f'movie data nums: {cnt}')


    ### movie video dataset
    def add_movie_data(self, use_script = True):
        cnt = 0
        use_subtitle = True
        data_root = ''
        with open('', 'r') as f:
            data = json.load(f)
        for cid_vid, items in data.items():
            for sample in items:
                question = sample['question']
                options = sample['options']
                answer = sample['answer']
                # answer = sample.get('answer', None)
                # if answer is None:
                #     continue
                reason = sample['reason']
                clues = sample['clues']
                video_path = sample['video_path']
                start_time = sample['start_time']
                end_time = sample['end_time']
                duration = int(end_time) - int(start_time) / 1000
                
                # if self.training_stage == 'stage1':
                #     if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
                # elif self.training_stage == 'stage2':
                #     if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

                script_path_list = sample['script_path']
                script_data = []
                if use_script:
                    for script_path in script_path_list:
                        with open(script_path, 'r') as f:
                            script = json.load(f)
                        script_data.extend(script)
               
                # '''
                # "Based on the provided video frames (sampled at 1 frame per second) and the accompanying dialogues, "
                # "please answer the following multiple-choice question. "
                # "You must select the correct option, provide a detailed reason for your answer, "
                # "and specify the time clues in seconds from the video that support your reasoning."
                # '''
                options = '\n'.join(options)
                qa = f'{question}\n{options}\nYou must select the correct option, provide a detailed reasoning for your answer, and specify the time clues in seconds from the video that support your reasoning.\n'
                clues_info = []
                for clue in clues:
                    if len(clue) == 1: clues_info.append(f'{clue[0]}s')
                    else: clues_info.append(f'{clue[0]}s~{clue[1]}s')
                label = f'Answer: {answer}\nReason: {reason}\nTime Clues: {clues_info}'

                conv = [
                    {'role': 'user', 'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': qa}
                    ]},
                    {'role':'assistant', 'content': label}
                ]
                self.samples.append({
                    'conv': conv,
                    'metadata': {'video_path': video_path, 'question': qa, 'label': label,
                                 'start_time': start_time, 'end_time': end_time, 'script_data': script_data,
                                 'dataset_name': 'movie', 'cid_vid': cid_vid, 'use_script': use_script}
                })
                self.tot += 1
                cnt += 1
        print(f'movie data nums: {cnt}')


    def add_time_data(self):
        cnt = 0
        with open('longVideo/ICLR26/data/time_train_set.json', 'r') as f:
            data = json.load(f)
        for sample in data:
            video = sample['video']
            qa = sample['QA']
            source = sample['source']
            for ele in qa:
                question = ele['q']
                answer = ele['a']
                conv = [
                    {'role': 'user', 'content': [
                        {'type': 'video', 'video': video},
                        {'type': 'text', 'text': question}
                    ]},
                    {'role':'assistant', 'content': answer}
                ]
                self.samples.append({
                    'conv': conv,
                    'metadata': {'video_path': video, 'question': question, 'label': answer,
                                 'source': source}
                })
                self.tot += 1
                cnt += 1

        print(f'time data nums: {cnt}')

    ### long video data
    def add_moment_10m_data(self, alpha = 0.3, filepath = None):
        cnt = 0
        data_root = '/data/temporal-grounding/Moment-10m'
        if self.training_stage == 'stage1':
            filepath = '/longVideo/ICLR26/data/moment-10m_2min_10min.json'
        else: 
            filepath = '/longVideo/ICLR26/data/moment-10m_10min_30min.json'
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        for data_type, data_list in data.items():
            random_data_list = random.sample(data_list, int(len(data_list) * alpha)) # 30% for every data type
            for sample in random_data_list:
                video_path = sample['video_path']
                duration = sample['duration']
                q_id = sample['q_id']
                data_type = sample['data_type']
                conversation = sample['conversation']
                if len(conversation) > 1: continue
                question = conversation[0]['User']
                answer = conversation[0]['Assistant']
                
                if self.training_stage == 'stage1':
                    if duration < self.stage1_min_duration or duration > self.stage1_max_duration: continue
                elif self.training_stage == 'stage2':
                    if duration < self.stage2_min_duration or duration > self.stage2_max_duration: continue

                conv = [
                    {'role': 'user', 'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': question}
                    ]},
                    {'role':'assistant', 'content': answer}
                ]
                self.samples.append({
                    'conv': conv,
                    'metadata': {'video_path': video_path, 'question': question, 'label': answer}
                })
                self.tot += 1
                cnt += 1
        
        print(f'moment-10m data nums: {cnt}')


    ### evaluate dataset ###
    def add_video_mme_data(self, use_subtitles = False):
        data_root = '/data/video-mme'
        with open(join(data_root,'test.json'),'r') as f:
            data = json.load(f)
        nums = 0
        for sample in data:
            video_id = sample['video_id']
            duration = sample['duration']
            if duration != 'long': continue

            # nums += 1
            # if nums <= 740: continue

            domain = sample['domain']
            url = sample['url']
            videoID = sample['videoID']
            # if videoID != 'sxrx7oCrb3A':
            #     continue
            task_type = sample['task_type']
            question = sample['question']
            sub_category = sample['sub_category']
            options = sample['options']
            answer = sample['answer']
            video_path = join(data_root,'video_data','data',videoID + '.mp4')
            # audio_path = join(data_root,'audio_data_2',videoID + '.mp3')
            if not exists(video_path):
                continue
            
            dialogue = None
            if use_subtitles:
                subtitle_path = join(data_root, 'subtitles_parsed', videoID + '.json')
                if not exists(subtitle_path): continue
                dialogue = ''
                with open(subtitle_path, 'r') as f:
                    subtitles_data = json.load(f)
                for ele in subtitles_data:
                    dialogue += ele['content'] + '\n'

            options = '\n'.join(options)
            # if use_subtitles:
            #     qa = f'Subtitles:\n{dialogue}\n{question}\n{options}'
            # else:
            #     qa = f'{question}\n{options}'
            qa = f'{question}\n{options}'
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': qa}
                ]},
                {'role':'assistant', 'content': answer}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': qa, 'label': answer,
                             'duration': duration, 'sub_category': sub_category, 'domain': domain,
                             'video_id': video_id, 'task_type': task_type, 'dialogue': dialogue,
                             'dataset_name': 'video_mme'}
            })
            self.tot += 1


    def add_longvideo_bench_data(self):

        def time_string_to_seconds(time_str):
            """
            将 "HH:MM:SS.mmm" 格式的时间字符串转换为秒数
            """
            # 解析时间字符串
            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
            # 转换为 timedelta 对象
            td = timedelta(
                hours=time_obj.hour,
                minutes=time_obj.minute,
                seconds=time_obj.second,
                microseconds=time_obj.microsecond
            )
            # 返回总秒数
            return td.total_seconds()
        
        data_root = '/data/longvideo_bench'
        with open(join(data_root, 'lvb_val.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            video_id = sample['video_id']
            question = sample['question']
            question_wo_referring_query = sample['question_wo_referring_query']
            candidates = sample['candidates']
            correct_choice = sample['correct_choice']
            position = sample['position']
            topic_category = sample['topic_category']
            question_category = sample['question_category']
            level = sample['level']
            id = sample['id']
            video_path = sample['video_path']
            video_path = join(data_root, 'videos', video_path)
            if not exists(video_path): continue
            subtitle_path = sample['subtitle_path']
            subtitle_path = join(data_root, 'subtitles', subtitle_path)
            if not exists(subtitle_path): continue
            duration_group = sample['duration_group']
            start_timestamp_for_subtitles = sample['starting_timestamp_for_subtitles']
            duration = sample['duration']
            subtitles = []
            with open(subtitle_path, 'r') as f:
                subtitle_data = json.load(f)
            for ele in subtitle_data:
                if 'timestamp' in ele:
                    st = ele['timestamp'][0]
                    et = ele['timestamp'][1]
                    if not isinstance(et, float):
                        et = duration
                    # if st is None or et is None:
                    #     print(subtitle_path)
                    #     exit(0)
                    # if start_timestamp_for_subtitles is None:
                    #     print('start is None.', sample)
                    #     exit(0)
                    if st >= start_timestamp_for_subtitles:
                        subtitles.append([st - start_timestamp_for_subtitles, et - start_timestamp_for_subtitles, ele['text']])
                elif 'start' in ele:
                    st = time_string_to_seconds(ele['start'])
                    et = time_string_to_seconds(ele['end'])
                    if not isinstance(et, float):
                        et = duration
                    
                    if st >= start_timestamp_for_subtitles:
                        subtitles.append([st - start_timestamp_for_subtitles, et - start_timestamp_for_subtitles, ele['line']])
            
            qa = f'{question}\n'
            for i, op in enumerate(candidates):
                qa += f'({char_list[i]}) {op}\n'
            
            label = char_list[correct_choice]
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': qa}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': qa, 'label': label,
                            'question_wo_referring_query': question_wo_referring_query,
                            'position': position, 'topic_category': topic_category,
                            'question_category': question_category, 'level': level, 'id': id,
                            'duration_group': duration_group, 'subtitles': subtitles,
                             'duration': duration,'video_id': video_id, 'dataset_name': 'longvideo_bench'}
            })
            self.tot += 1


    def add_cinepile_test_data(self):
        data_root = ''
        with open(join('/data/CinePile', 'cinepile_v1_test.json'),'r') as f:
            samples = json.load(f)
        for idx, sample in enumerate(samples):
            movie_name = sample['movie_name']
            subtitles = sample['subtitles']
            question = sample['question']
            choices = sample['choices']
            answer_key = sample['answer_key']
            answer_key_position = sample['answer_key_position']
            question_category = sample['question_category']
            yt_clip_link = sample['yt_clip_link']
            vid = yt_clip_link.split('=')[-1]

            video_path = join(data_root,'0221',vid+'.mp4')
            if not exists(video_path): continue

            qa = f'{subtitles}\n{question}\n'
            for idx, op in enumerate(choices):
                qa += f'({char_list[idx]}) {op}\n'
            
            answer = char_list[answer_key_position]

            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': qa}
                ]},
                {'role':'assistant', 'content': answer}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': qa, 'label': answer,
                             'answer_key': answer_key, 'question_category': question_category}
            })
            self.tot += 1


    def add_nextqa_test_data(self):
        data_root = '/data/Next-QA/test-video'
        with open(join(data_root, 'nextqa_test.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            id = sample['id']
            video = sample['video']
            conversations = sample['conversations']
            video_path = join(data_root, 'nextqa-test-video', video.replace('reasoning/next_qa/split_videos/', ''))
            if not exists(video_path): continue
            question = conversations[0]['value']
            output = conversations[1]['value']
            if '<image>\n' in question: question = question.replace('<image>\n','')
            elif '<image>' in question: question = question.replace('<image>','')
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question}
                ]},
                {'role':'assistant', 'content': output}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': question, 'label': output, 
                             'id': id, 'dataset_name': 'nextqa'}
            })
            self.tot += 1


    def add_VNBench_data(self):
        data_root = '/data/VNBench'
        with open(join(data_root, 'VNBench-main-4try.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            video = sample['video']
            question = sample['question']
            options = sample['options']
            needle_time = sample['needle_time']
            gt = sample['gt']
            gt_option = sample['gt_option']
            length = sample['length']
            type = sample['type']
            _try = sample['try']

            qa = f'{question}\n'
            for i, op in enumerate(options):
                qa += f'({char_list[i]}) {op}\n'

            video_path = join(data_root, 'VNBench_new', video.split('/')[-1])
            if not exists(video_path): continue

            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': qa}
                ]},
                {'role':'assistant', 'content': gt_option}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': qa, 'label': gt_option, 
                             'needle_time': needle_time, 'dataset_name': 'vnbench', 'length': length,
                            'type': type, 'try': _try, 'gt': gt}
            })
            self.tot += 1


    def add_LMVU_data(self):
        data_root = '/data/LMVU'
        with open(join(data_root, 'gt', 'test_mcq_gt.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            video = sample['video']
            duration = sample['duration']
            question = sample['question']
            candidates = sample['candidates']
            answer = sample['answer']
            question_type = sample['question_type']
            question_id = sample['question_id']

            qa = f'{question}\n'
            for i, op in enumerate(candidates):
                qa += f'({char_list[i]}) {op}\n'

            video_path = join(data_root, 'video_data/video', video)
            if not exists(video_path): continue

            label = char_list[candidates.index(answer)]

            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': qa}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': qa, 'label': label, 
                             'dataset_name': 'lmvu', 'duration': duration, 'question_type': question_type,
                             'question_id': question_id}
            })
            self.tot += 1


    def add_movie_test_data(self):
        use_subtitle = True
        data_root = '/data/MovieDataset/new'
        with open('/longVideo/ICLR26/data/qag_filtered_test_0.2.json', 'r') as f:
            data = json.load(f)
        for cid_vid, items in data.items():
            for sample in items:
                question = sample['question']
                options = sample['options']
                answer = sample.get('answer', None)
                if answer is None:
                    continue
                reason = sample['reason']
                clues = sample['clues']
                video_path = sample['video_path']
                start_time = sample['start_time']
                end_time = sample['end_time']
                script_path_list = sample['script_path']
                script_data = []
                for script_path in script_path_list:
                    with open(script_path, 'r') as f:
                        script = json.load(f)
                    script_data.extend(script)
                
                options = '\n'.join(options)
                # qa = f'{question}\n{options}\n'
                # label = f'Answer: {answer}\nReason: {reason}\nClues: {clues}'

                qa = f'{question}\n{options}\nYou must select the correct option, provide a detailed reasoning for your answer, and specify the time clues in seconds from the video that support your reasoning.\n'
                clues_info = []
                for clue in clues:
                    if len(clue) == 1: clues_info.append(f'{clue[0]}s')
                    else: clues_info.append(f'{clue[0]}s~{clue[1]}s')
                label = f'Answer: {answer}\nReason: {reason}\nTime Clues: {clues_info}'


                conv = [
                    {'role': 'user', 'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': qa}
                    ]},
                    {'role':'assistant', 'content': label}
                ]
                self.samples.append({
                    'conv': conv,
                    'metadata': {'video_path': video_path, 'question': qa, 'label': label,
                                 'start_time': start_time, 'end_time': end_time, 'script_data': script_data,
                                 'dataset_name': 'movie', 'cid_vid': cid_vid, 'answer': answer, 'reason': reason,
                                 'clues': clues}
                })
                self.tot += 1
    

    def add_charades_sta_test_data(self):
        data_root = '/data/charades-sta'
        with open(join(data_root, 'test.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            video_path = sample['video_path']
            caption = sample['caption']
            timestamp = sample['timestamp']

            label = f'The given query happens in {timestamp[0]:.1f} - {timestamp[1]:.1f} seconds.'

            question = f"Find the video segment that corresponds to the given textual query {caption} and determine its start and end seconds."
            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': question, 'label': label, 
                            'timestamp': timestamp, 'caption': caption}
            })
            self.tot += 1


    def add_vista_data(self):
        data_root = '/data/VideoVista'
        with open(join(data_root,'VideoVista.json'),'r') as f:
            samples = json.load(f)
        for sample in samples:
            question = sample['Question']
            choices = sample['Answer_Choices']
            answer = sample['Answer']
            type = sample['Type']
            if type == 'Relation Reasoning-Image':
                continue
                # image_name = sample['image_name']
                # image_path = join(data_root,'images',image_name)
                # if not exists(image_path):
                #     continue
            time = sample['time']
            category = sample['category']
            video_name = sample['video_name']
            dirs = video_name.split('.')
            video_path = join(data_root,'merged',category,dirs[1],video_name)
            if not exists(video_path):
                continue
            
            qa = f'{question}\n'
            for choice_id, choice in enumerate(choices):
                qa += f'({char_list[choice_id]}) {choice}\n'
            label = f'{char_list[answer]}'

            conv = [
                {'role': 'user', 'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': qa}
                ]},
                {'role':'assistant', 'content': label}
            ]
            self.samples.append({
                'conv': conv,
                'metadata': {'video_path': video_path, 'question': qa, 'label': label, 
                             'dataset_name': 'vista', 'time': time, 'category': category, 'type': type,
                             }
            })
            self.tot += 1


    def add_youcook2_val_data(self):
        data_root = '/data/youcook'
        with open(join(data_root, 'val.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            video_path = sample['video_path']
            sentence = sample['sentence']
            

    def add_perception_test_val_data(self):
        data_root = '/data/reflect-reasoning/video-r1'
        with open(join(data_root, 'PerceptionTest_6348_new.json'), 'r') as f:
            data = json.load(f)
        for sample in data:
            duration = sample['duration']
            if duration <= 30: continue
            problem = sample['problem']
            options = sample['options']
            solution = sample['solution']
            video_path = sample['path']
            duration = sample['duration']
            
            options = '\n'.join(options)
            question = f'{problem}\n{options}'
            matches = re.findall(r'<answer>(.*?)</answer>', solution)
            label = matches[0].strip()
            label = f'({label})'


    def add_mvbench_data(self):
        task2folder = {
            "action_sequence": "star/Charades_v1_480", # has start & end
            "action_prediction": "star/Charades_v1_480", # has start & end
            "action_antonym": "ssv2_video",
            "fine_grained_action": "Moments_in_Time_Raw/videos",
            "unexpected_action": "FunQA_test/test",
            "object_existence": "clevrer/video_validation",
            "object_interaction": "star/Charades_v1_480", # has start & end
            "object_shuffle": "perception/videos",
            "moving_direction": "clevrer/video_validation",
            "action_localization": "sta/sta_video",  # has start & end
            "scene_transition": "scene_qa/video",
            "action_count": "perception/videos",
            "moving_count": "clevrer/video_validation",
            "moving_attribute": "clevrer/video_validation",
            "state_change": "perception/videos",
            # "fine_grained_pose": "nturgbd",
            "character_order": "perception/videos",
            "egocentric_navigation": "vlnqa",
            # "episodic_reasoning": "tvqa/frames_fps3_hq",  # has start & end, read frame
            "counterfactual_inference": "clevrer/video_validation",
        }

        data_root = '/data/mvbench'
        for task, folder in task2folder.items():
            with open(join(data_root, 'json', f'{task}.json'), 'r') as f:
                data = json.load(f)
            for sample in data:
                video = sample['video']
                question = sample['question']
                candidates = sample['candidates']
                answer = sample['answer']
                video_path = join(data_root, 'video', folder, video)
                if not exists(video_path): continue
                if answer in candidates:
                    label = candidates.index(answer)
                    label = char_list[label]
                else:
                    label = 'None'
                
                qa = f'{question}\n'
                for idx, op in enumerate(candidates):
                    qa += f'({char_list[idx]}) {op}\n'
                
                conv = [
                    {'role': 'user', 'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': qa}
                    ]},
                    {'role':'assistant', 'content': label}
                ]
                self.samples.append({
                    'conv': conv,
                    'metadata': {'video_path': video_path, 'question': qa, 'label': label, 
                                'dataset_name': 'mvbench', 'answer': answer, 'candidates': candidates,
                                'task': task}
                })
                self.tot += 1


    def add_test_samples(self):
        video_path = 'data/demo_split_24_80.mp4'
        prompt = '''Why didn't the small kitten catch the mouse?
        (A) It is afraid of the mouse.
        (B) The blue cartoon cat forbids it to catch the mouse.
        (C) It thinks the mouse is its friend.
        (D) Because the mouse is its teacher.
        '''
        output = 'none'
        conv = [
            {'role': 'user', 'content': [
                {'type': 'video', 'video': video_path},
                {'type': 'text', 'text': prompt}
            ]},
            {'role':'assistant', 'content': output}
        ]
        self.samples.append({
            'conv': conv,
            'metadata': {'video_path': video_path, 'question': prompt, 'label': output, 
                            }
        })
        self.tot += 1
        

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
        video_sample_fps_list = []
        shot_frame_nums_input = []
        for mm_info in mm_infos:
            mm_info.update(self.vision_kwargs)
            if "image" in mm_info:
                image_inputs.append(fetch_image(mm_info))
            
            elif "video" in mm_info:
                video_shot_inputs, shot_frame_nums, video_sample_fps = fetch_video(mm_info)
                video_inputs.extend(video_shot_inputs)
                video_sample_fps_list.extend(video_sample_fps)
                shot_frame_nums_input.extend(shot_frame_nums)
            
            elif 'audio' in mm_info:
                shot_audio_list = fetch_audio(mm_info)
                audio_inputs = shot_audio_list

            else:
                raise ValueError("image, audio or video should in content.")
        
        # if len(image_inputs) == 0: image_inputs = None
        # if len(video_inputs) == 0: video_inputs = None
        # if len(audio_inputs) == 0: audio_inputs = None

        return image_inputs, audio_inputs, video_inputs, {'fps': video_sample_fps_list, 'shot_frame_nums': shot_frame_nums_input}



    def tokenize(self,text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    

    def apply_qwen2_5_vl_chat_template(self, conv, image_token_nums, video_token_nums, audio_token_nums):
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



    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]
        conv = sample['conv']
        metadata = sample.get('metadata',{})

        image_inputs, audio_inputs, video_inputs, video_info = self.get_mm_info(conv)

        image_token_nums = []
        if len(image_inputs) > 0:
            image_inputs = self.image_processor(images=image_inputs, videos=None, return_tensors = 'pt')
            image_grid_thw = image_inputs["image_grid_thw"]
            image_token_nums = [image_grid_thw[i].prod() // 4 for i in range(image_grid_thw.shape[0])]
        
        video_token_nums = []
        if len(video_inputs) > 0:
            video_inputs = self.video_processor(images=None, videos=video_inputs, return_tensors ='pt')
            video_grid_thw = video_inputs["video_grid_thw"]

            fps = self.fps
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            video_inputs.update({"second_per_grid_ts": torch.tensor(second_per_grid_ts)})
            video_token_nums = [video_grid_thw[i].prod() // 4 for i in range(video_grid_thw.shape[0])]
        
        audio_token_nums = []
        if len(audio_inputs) > 0:
            processed_audio = self.audio_processor(audio_inputs)
            spec = processed_audio['spec'] # [seg1, seg2, ...]
            fbank = processed_audio['fbank']
            audio_grid_thw = processed_audio['audio_grid_thw'] # [[1, 88], [1, 88], .. ]
            audio_token_nums = [audio_grid_thw[i].prod() // 4 for i in range(audio_grid_thw.shape[0])]
            
        question = metadata['question']
        video_path = metadata['video_path']
        label = metadata['label']
        shot_nums = len(video_grid_thw)

        if self.shot_size is not None:  # use memory, update conv

            if 'dataset_name' in metadata and metadata['dataset_name'] == 'longvideo_bench':
                shot_frame_nums = video_info['shot_frame_nums']
                fps = video_info['fps']
                video_st = 0
                right_idx = 0
                content = []
                for i in range(shot_nums):
                    content.append({'type': 'video', 'video': video_path})
                    content.append({'type': 'text', 'text': question})
                    content.append({'type': 'text', 'text': '<split>'})
                
                subtitles = metadata['subtitles']
                subtitles_str = ''
                for ele in subtitles:
                    subtitles_str += ele[2] + '\n'
                
                content.append({'type': 'text', 'text': subtitles_str})
                content.append({'type': 'text', 'text': question})

            elif 'dataset_name' in metadata and metadata['dataset_name'] == 'movie':
                script_start_time = metadata['start_time']
                script_end_time = metadata['end_time']
                script_data = metadata['script_data']
                shot_frame_nums = video_info['shot_frame_nums']
                fps = video_info['fps']
                video_st = 0
                right_idx = 0
                content = []

                if 'use_script' in metadata and metadata['use_script']:    
                    for i in range(shot_nums):
                        shot_duration = shot_frame_nums[i] / fps[i]
                        video_et = video_st + shot_duration
                        j = right_idx
                        shot_script = ''
                        
                        while j < len(script_data):
                            if 'scene' in script_data[j]: shot_script += script_data[j]['scene'] + '\n'
                            elif 'description' in script_data[j]: shot_script += script_data[j]['description'] + '\n'
                            elif 'role' in script_data[j]:
                                role = script_data[j]['role']
                                dialogue = script_data[j]['dialogue']
                                st = script_data[j].get('start_time', None)
                                et = script_data[j].get('end_time', None)
                                if st is None or et is None:
                                    # shot_script += f'{role}: {dialogue}' + '\n'
                                    shot_script += f'{dialogue}\n'
                                else:
                                    st = (int(st) - int(script_start_time)) / 1000
                                    et = (int(et) - int(script_start_time)) / 1000
                                    if video_et >= et:
                                        # shot_script += f'{st:.1f}s~{et:.1f}s, {role}: {dialogue}' + '\n'
                                        shot_script += f'{dialogue}\n'
                                    else:
                                        break

                            j += 1

                        right_idx = j
                        video_st = video_et

                        content.append({'type': 'video', 'video': video_path})
                        content.append({'type': 'text', 'text': shot_script})
                        content.append({'type': 'text', 'text': question})
                        content.append({'type': 'text', 'text': '<split>'})

                    content.append({'type': 'text', 'text': question})

                else:

                    content = []
                    for i in range(shot_nums):
                        content.append({'type': 'video', 'video': video_path})
                        content.append({'type': 'text', 'text': question})
                        content.append({'type': 'text', 'text': '<split>'})
                    content.append({'type': 'text', 'text': question})

            else:
                content = []
                for i in range(shot_nums):
                    content.append({'type': 'video', 'video': video_path})
                    content.append({'type': 'text', 'text': question})
                    content.append({'type': 'text', 'text': '<split>'})
                
                if 'dataset_name' in metadata and metadata['dataset_name'] == 'video_mme':
                    dialogue = metadata['dialogue']
                    if dialogue is not None: content.append({'type': 'text', 'text': f'Dialogue:\n{dialogue}\n{question}'})
                    else: content.append({'type': 'text', 'text': question})
                
                else:
                    content.append({'type': 'text', 'text': question})
            

            conv = [
                {'role': 'user', 'content': content},
                {'role': 'assistant', 'content': label}
            ]
            
        output = self.apply_qwen2_5_vl_chat_template(
            conv=conv,
            image_token_nums=image_token_nums,
            video_token_nums=video_token_nums,
            audio_token_nums=audio_token_nums
        )
        
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
            'metadata':metadata,
        }
        if len(image_inputs) > 0:
            data['image_inputs'] = image_inputs
        if len(video_inputs) > 0:
            data['video_inputs'] = video_inputs
        if len(audio_inputs) > 0:
            data['audio_inputs'] = audio_inputs
        
        return data
        

@dataclass
class DataCollatorForLongVideoDataset(object):
    processor: Qwen2_5_VLProcessor
    mode: str = 'train'
    def __call__(self, instances: Sequence[Dict]):
        tokenizer = self.processor.tokenizer
        batch_instruction_ids = []
        batch_label_ids = []
        batch_metadata = []
        batch_attention_mask = []
        batch_X_data = []
        image_grid_thw = []
        video_grid_thw = []
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
            batch_X_data.append({
                'image':image_inputs,
                'video':video_inputs,
                'audio':audio_inputs
            })
            if image_inputs is not None:
                image_grid_thw.append(image_inputs['image_grid_thw'])
            if video_inputs is not None:
                video_grid_thw.append(video_inputs['video_grid_thw'])
            batch_metadata.append(instance['metadata'])
        
        input_ids = pad_sequence(batch_instruction_ids,batch_first=True,padding_value = tokenizer.pad_token_id)
        labels = pad_sequence(batch_label_ids,batch_first=True,padding_value=-100)
        attention_mask = pad_sequence(batch_attention_mask,batch_first=True,padding_value=0)
        
        if len(image_grid_thw) > 0:
            image_grid_thw = torch.cat(image_grid_thw,dim=0)
        if len(video_grid_thw) > 0:
            video_grid_thw = torch.cat(video_grid_thw,dim=0)
        data = {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask': attention_mask,
            'batch_X_data': batch_X_data,
            'image_grid_thw':image_grid_thw,
            'video_grid_thw':video_grid_thw,
        }
        # print(batch_metadata)
        if self.mode == 'test':
            data['batch_metadata'] = batch_metadata
        return data





