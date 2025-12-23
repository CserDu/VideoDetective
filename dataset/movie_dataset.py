import json,os,random
from os.path import join,exists
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
from typing import Sequence,Dict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizer,Qwen2Tokenizer
from transformers import Qwen2_5_VLImageProcessor

from dataset.video_processor import VideoProcessor,ImageProcessor
from dataset.audio_processor import AudioProcessor
from dataset.data_utils import (
    fetch_image, 
    fetch_audio, 
    fetch_video_quick, 
    fetch_video,
    process_mm_info,
    # apply_qwen2_5_vl_chat_template
)

FPS = 1.0

test_cids = [
    '1o29ui77e85grdr',
    '31082i4u5ovkrl0',
    'f6quz8ps5lkn728',
    'r6hc2kqgvnmiejn',
    'c2seabnsfozypl8',
]

qtype_list = [
    'story timeline',
    'character behavior',
    'psychological state',
    'plot development'
]

class MovieDataset(Dataset):

    def __init__(
        self,
        image_processor: Qwen2_5_VLImageProcessor,
        audio_processor: AudioProcessor,
        video_processor: Qwen2_5_VLImageProcessor,
        tokenizer: Qwen2Tokenizer,
        mode='train',
        use_memory = False,
        use_caption = False,
        question_after_shot = False, # question-aware: insert question after every shot data.
    ):
        super().__init__()

        self.samples = []
        self.mode = mode
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.tot = 0
        self.use_memory = use_memory
        self.use_caption= use_caption
        self.question_after_shot = question_after_shot
        print('use memory: ',self.use_memory,' use caption: ',self.use_caption,' question-aware: ',question_after_shot)
        if use_memory:
            # self.add_movie_samples_with_memory()
            # self.add_cinepile_samples()
            self.add_video_mme_samples()
            # self.add_longvideo_bench_samples()
            # self.add_videovista_samples()
        else:
            self.add_movie_samples()


    def add_movie_samples(self):
        data_root = ''
        with open('data/imdb_movie_raw_info.json','r') as f:
            raw_info_samples = json.load(f)
        cid2role_dir = {}
        for sample in raw_info_samples:
            cid = sample['cid']
            cid2role_dir[cid] = []
            role_names = sample.get('role_names',[])
            role_dir = join(data_root, cid, 'role_image')
            for role_name in role_names:
                role_image_dir = join(role_dir, role_name)
                if exists(role_image_dir):
                    cid2role_dir[cid].append(role_image_dir)

        tot = 0
        anno_file = 'qa_samples.json' if not self.use_caption else 'qa_caption_samples.json'
        annotation_path = join(data_root,'annotations',anno_file)
        with open(annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            cid = sample['cid']
            story_id = sample['story_id']
            type = sample['type']

            if cid in test_cids and self.mode == 'train':
                continue
            if cid not in test_cids and self.mode == 'test':
                continue
            
            if type == 'qa':
                question = sample['question']
                qtype = sample['qtype']
                a = sample['a']
                b = sample['b']
                c = sample['c']
                d = sample['d']
                answer = sample['answer']
                reason = sample['reason']
                if question == 'none':
                    continue
            
            else:
                caption = sample['caption']
                if caption == 'none':
                    continue

            role_dir = cid2role_dir[cid]
            if len(role_dir) < 5:
                print('role nums < 5, cid: ',cid)
                continue

            ### audio & video path
            story_video_path = join(data_root, cid, 'story_video', f'story_{story_id}.mp4')
            story_audio_path = join(data_root, cid, 'story_audio', f'story_{story_id}.mp3')

            user_content_list = []
            user_content_list.append({
                'type':'text',
                'text':'I will give you some images of roles, keep in mind what they look like and what they are wearing. These are role images:\n'
            })
            ### set role_image_dir for random sample one image during training stage.
            for dir in role_dir:
                role_name = dir.split('/')[-1]
                user_content_list.append({'type':'image','image':dir})
                user_content_list.append({'type':'text','text':f'The role name in this image is {role_name}.\n'})
              
            user_content_list.append({
                'type':'text',
                'text':'Then I will provide you with a video clip and corresponding speech.\n',
            })            
            user_content_list.append({'type':'video','video':story_video_path})
            user_content_list.append({'type':'text','text':' Look this video clip.\n'})
            user_content_list.append({'type':'audio','audio':story_audio_path})
            user_content_list.append({'type':'text','text':' Listen to corresponding speech.\n'})

            if type == 'qa':
                # user_content_list.append({
                #     'type':'text',
                #     'text':'Please answer following question based on above information.\n'
                # })
                qa = f'Question: {question}\n{a}\n{b}\n{c}\n{d}\n'
                user_content_list.append({
                    'type':'text',
                    'text':qa
                })
                gpt_output = f'{answer}\n{reason}'
                conv = [
                    {'role':'user','content':user_content_list},
                    {'role':'assistant','content':gpt_output}
                ]
            
            else:
                user_content_list.append({
                    'type':'text',
                    'text':'Please describe the storyline of this movie video and speech based on above information.\n'
                })
                gpt_output = caption
                conv = [
                    {'role':'user','content':user_content_list},
                    {'role':'assistant','content':gpt_output}
                ]

            metadata = {
                'type': type,
                'cid':cid,
                'story_id':story_id
            }
            if type == 'qa':
                metadata['qtype'] = qtype
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1

            
        self.tot += tot
        print(f'{self.mode}, movie sample nums: {tot}')


    def add_movie_samples_with_memory(self):
        data_root = ''
        with open('data/imdb_movie_raw_info.json','r') as f:
            raw_info_samples = json.load(f)
        cid2role_dir = {}
        for sample in raw_info_samples:
            cid = sample['cid']
            cid2role_dir[cid] = []
            role_names = sample.get('role_names',[])
            role_dir = join(data_root, cid, 'role_image')
            for role_name in role_names:
                role_image_dir = join(role_dir, role_name)
                if exists(role_image_dir):
                    cid2role_dir[cid].append(role_image_dir)
        
        tot = 0
        # exist_set = set()
        anno_file = 'qa_samples.json' if not self.use_caption else 'qa_caption_samples.json'
        annotation_path = join(data_root,'annotations',anno_file)
        with open(annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            cid = sample['cid']
            story_id = sample['story_id']
            type = sample['type']
            if cid in test_cids and self.mode == 'train':
                continue
            if cid not in test_cids and self.mode == 'test':
                continue 

            if type == 'qa':
                question = sample['question']
                qtype = sample['qtype']
                a = sample['a']
                b = sample['b']
                c = sample['c']
                d = sample['d']
                answer = sample['answer']
                reason = sample['reason']
                if question == 'none':
                    continue
            else:
                question = 'Please describe the storyline of this movie video and speech based on above information.'
                caption = sample['caption']
                if caption == 'none':
                    continue

            role_dir = cid2role_dir[cid]
            if len(role_dir) < 5:
                continue

            ### audio & video path
            story_video_path = join(data_root, cid, 'story_video', f'story_{story_id}.mp4')
            story_audio_path = join(data_root, cid, 'story_audio', f'story_{story_id}.mp3')

            user_content_list = []
            user_content_list.append({
                'type':'text',
                'text':'I will give you some images of roles, keep in mind what they look like and what they are wearing. These are role images:\n'
            })
            for dir in role_dir:
                role_name = dir.split('/')[-1]
                user_content_list.append({'type':'image','image':dir})
                user_content_list.append({'type':'text','text':f'The role name in this image is {role_name}.\n'})
            user_content_list.append({'type':'text','text':'<split>'})

            user_content_list.append({
                'type':'text',
                'text':'Then I will provide you with some video clips and corresponding speech.\n',
            })
            shot_results = join(data_root, cid, 'shot_results_12_18.json')
            with open(shot_results,'r') as f2:
                shot_results = json.load(f2)
            result = shot_results[int(story_id)]
            assert result['story_id'] == story_id, cid
            merged_shot_list = result['merged_shot_list']
            shot_nums = len(merged_shot_list)

            ######## ablation: use fixed window_size as one shot ########
            # duration = result['duration']
            # window_size = 15
            # shot_nums = int(duration // window_size)
            # if duration % window_size != 0:
            #     shot_nums += 1
            #############################################################

            # fps = FPS
            # duration = result['duration']
            # total_frames = int(fps * duration)
            # window_size = 15
            # shot_nums = int(total_frames // window_size)
            # if total_frames % window_size != 0:
            #     shot_nums += 1
            
            if shot_nums == 0:
                continue
            for i in range(shot_nums):
                # user_content_list.append({'type':'video','video':story_video_path,'merged_shot_list':merged_shot_list})
                user_content_list.append({'type':'video','video':story_video_path,'shot_nums':shot_nums})
                user_content_list.append({'type':'text','text':' Look this video clip.\n'})

                # user_content_list.append({'type':'audio','audio':story_audio_path,'merged_shot_list':merged_shot_list})
                user_content_list.append({'type':'audio','audio':story_audio_path,'shot_nums':shot_nums})
                user_content_list.append({'type':'text','text':' Listen to corresponding speech.\n'})

                ### question-aware
                if self.question_after_shot:
                    user_content_list.append({
                        'type':'text',
                        'text':question,
                    })
                user_content_list.append({'type':'text','text':'<split>'})
            
            if type == 'qa':
                # user_content_list.append({
                #     'type':'text',
                #     'text':'Please answer following question based on above information.\n'
                # })
                qa = f'Question: {question}\n{a}\n{b}\n{c}\n{d}\n'
                user_content_list.append({
                    'type':'text',
                    'text':qa
                })
                gpt_output = f'{answer}\n{reason}'
                conv = [
                    {'role':'user','content':user_content_list},
                    {'role':'assistant','content':gpt_output}
                ]
            else:
                user_content_list.append({
                    'type':'text',
                    'text':'Please describe the storyline of this movie video and speech based on above information.\n'
                })
                gpt_output = caption
                conv = [
                    {'role':'user','content':user_content_list},
                    {'role':'assistant','content':gpt_output}
                ]
            
            metadata = {
                'type': type,
                'cid':cid,
                'story_id':story_id
            }
            if type == 'qa':
                metadata['qtype'] = qtype
            self.samples.append(
                {
                    'conv':conv,
                    'metadata':metadata
                }
            )
            tot += 1
            if tot >= 1:
                break
            
        self.tot += tot
        print(f'{self.mode}, movie sample nums: {tot}')


    def add_video_mme_samples(self):
        data_root = ''
        with open(join(data_root,'test.json'),'r') as f:
            samples = json.load(f)
        char_list = ['(A)','(B)','(C)','(D)','(E)','(F)','(G)']
        tot = 0
        for sample in samples:
            video_id = sample['video_id']
            duration = sample['duration']
            domain = sample['domain']
            url = sample['url']
            videoID = sample['videoID']
            if videoID != 'sxrx7oCrb3A':
                continue
            task_type = sample['task_type']
            question = sample['question']
            choices = sample['options']
            answer = sample['answer']
            video_path = join(data_root,'video_data','data',videoID + '.mp4')
            audio_path = join(data_root,'audio_data_2',videoID + '.mp3')
            if not exists(video_path) or not exists(audio_path):
                continue
            user_content_list = []
            user_content_list.append({
                'type':'text',
                'text':'I will provide you with some video clips and corresponding speech.\n'
            })
            shot_result_path = join(data_root,'shot_results',videoID+'.json')
            if not exists(shot_result_path):
                continue
            with open(shot_result_path,'r') as f2:
                shot_results = json.load(f2)
            shot_nums = len(shot_results)
            if shot_nums == 0:
                continue

            # fps = FPS
            # duration = result['duration']
            # total_frames = int(fps * duration)
            # window_size = 15
            # shot_nums = int(total_frames // window_size)
            # if total_frames % window_size != 0:
            #     shot_nums += 1

            for i in range(shot_nums):
                user_content_list.append({'type':'video','video':video_path,'merged_shot_list':shot_results})
                user_content_list.append({'type':'text','text':' Look this video clip.\n'})

                user_content_list.append({'type':'audio','audio':audio_path,'merged_shot_list':shot_results})
                user_content_list.append({'type':'text','text':' Listen to corresponding speech.\n'})

                ### question-aware
                if self.question_after_shot:
                    user_content_list.append({
                        'type':'text',
                        'text':question,
                    })
                user_content_list.append({'type':'text','text':'<split>'})
            
            qa = f'Question: {question}\n'
            for i in range(len(choices)):
                qa += f'{choices[i]}\n'
            user_content_list.append({
                'type':'text',
                'text':qa
            })
            gpt_output = f'({answer})'
            conv = [
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':gpt_output}
            ]

            self.samples.append({
                'conv':conv,
                'metadata':{'vid':videoID,'task_type':task_type,'qa':qa,'answer':answer}
            })
            tot += 1

            if tot >= 1:
                break

        self.tot += tot
        print(f'{self.mode}, video-mme sample nums: {tot}')


    def add_cinepile_samples(self):
        data_root = ''
        with open(join('', 'cinepile_v1_test.json'),'r') as f:
            samples = json.load(f)
        char_list = ['(A)','(B)','(C)','(D)','(E)','(F)','(G)']
        tot = 0
        print('ori tot samples: ',len(samples))
        for idx, sample in enumerate(samples):
            # if idx <= 309:
            #     continue
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
            audio_path = join(data_root,'audio_data_2',vid+'.mp3')
            if not exists(audio_path) or not exists(video_path):
                continue

            subtitle_list = subtitles.split('\n<subtitle>')
            subtitles = '\n'.join(subtitle_list)

            user_content_list = []
            # user_content_list.append({
            #     'type':'text',
            #     'text':'I will provide you with some video clips and corresponding speech.\n'
            # })
            user_content_list.append({
                'type':'text',
                'text':'I will provide you with some video clips, corresponding speech and dialogues.\n'
            })

            shot_result_path = join('',vid+'.json')
            if not exists(shot_result_path):
                continue
            with open(shot_result_path,'r') as f2:
                shot_results = json.load(f2)
            shot_nums = len(shot_results)
            if shot_nums == 0:
                continue
            for i in range(shot_nums):
                user_content_list.append({'type':'video','video':video_path,'merged_shot_list':shot_results})
                user_content_list.append({'type':'text','text':' Look this video clip.\n'})

                user_content_list.append({'type':'audio','audio':audio_path,'merged_shot_list':shot_results})
                user_content_list.append({'type':'text','text':' Listen to corresponding speech.\n'})

                ### question-aware
                if self.question_after_shot:
                    user_content_list.append({
                        'type':'text',
                        'text':question,
                    })
                user_content_list.append({'type':'text','text':'<split>'})
            
            user_content_list.append({
                'type':'text',
                'text':f'These are corresponding dialogues:\n{subtitles}\n'
            })
            
            qa = f'Question: {question}\n'
            for i in range(len(choices)):
                qa += f'{char_list[i]}: {choices[i]}\n'
            user_content_list.append({
                'type':'text',
                'text':qa
            })
            gpt_output = f'{char_list[int(answer_key_position)]}'
            conv = [
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':gpt_output}
            ]

            self.samples.append({
                'conv':conv,
                'metadata':{'vid':vid,'answer_key':answer_key,'answer_key_position':answer_key_position,'choice':char_list[int(answer_key_position)],'category':question_category}
            })
            tot += 1
        
        self.tot += tot
        print(f'{self.mode}, cinepile sample nums: {tot}')


    def add_longvideo_bench_samples(self):
        data_root = ''
        with open(join(data_root,'lvb_val.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        for sample in samples:
            video_id = sample['video_id']
            question = sample['question']
            candidates = sample['candidates']
            correct_choice = sample.get('correct_choice',-1)
            top_category = sample['topic_category']
            question_category = sample['question_category']
            level = sample['level']
            video_path = sample['video_path']
            file = video_path[:-4]
            duration_group = sample['duration_group']
            duration = sample['duration']
            view_count = sample['view_count']

            user_content_list = []
            user_content_list.append({
                'type':'text',
                'text':'I will provide you with some video clips and dialogues.\n'
            })
            video_path = join(data_root,'videos',video_path)
            if not exists(video_path):
                continue
            merged_shot_list_path = join(data_root,'shot_results_with_subtitles',file+'.json')
            if not exists(merged_shot_list_path):
                continue
            with open(merged_shot_list_path,'r') as f2:
                shot_results = json.load(f2)
            shot_nums = len(shot_results)
            if shot_nums == 0:
                continue
            subtitles = []
            for i in range(shot_nums):
                user_content_list.append({'type':'video','video':video_path,'merged_shot_list':shot_results})
                user_content_list.append({'type':'text','text':' Look this video clip.\n'})

                # subtitle_data = shot_results[i]['subtitle_data']
                # subtitles.extend(subtitle_data)
                
                # subtitle_data = '\n'.join(subtitle_data)
                # user_content_list.append({'type':'text','text':' These are corresponding dialogues:\n'})
                # user_content_list.append({'type':'text','text':subtitle_data})

                # user_content_list.append({'type':'audio','audio':audio_path,'merged_shot_list':shot_results})
                # user_content_list.append({'type':'text','text':' Listen to corresponding speech.\n'})

                ### question-aware
                if self.question_after_shot:
                    user_content_list.append({
                        'type':'text',
                        'text':question,
                    })
                user_content_list.append({'type':'text','text':'<split>'})
            
            # subtitles = '\n'.join(subtitles)
            # user_content_list.append({'type':'text','text':'These are corresponding dialogues:\n'})
            # user_content_list.append({'type':'text','text':f'{subtitles}\n'})

            qa = f'Question: {question}\n'
            for choice_id, choice in enumerate(candidates):
                qa += f'({chr(ord("A")+choice_id)}): {choice}\n'
            user_content_list.append({
                'type':'text',
                'text':qa
            })
            gpt_output = f'({chr(ord("A")+correct_choice)})'
            conv = [
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':gpt_output}
            ]

            self.samples.append({
                'conv':conv,
                'metadata':{
                    'video_id':video_id,
                    'question':question,
                    'correct_choice':correct_choice,
                    'topic_category':top_category,
                    'question_category':question_category,
                    'lavel':level,
                    'video_path':video_path,
                    'duration_group':duration_group,
                    'duration':duration,
                    'view_count':view_count
                }
            })
            tot += 1

        print(f'longvideo_bench sample nums: {tot}')
        self.tot += tot


    def add_videovista_samples(self):
        data_root = ''
        with open(join(data_root,'VideoVista.json'),'r') as f:
            samples = json.load(f)
        tot = 0
        pbar = tqdm(total=len(samples) + 1,desc='video vista')
        for sample in samples:
            pbar.update(1)
            question = sample['Question']
            choices = sample['Answer_Choices']
            answer = sample['Answer']
            type = sample['Type']
            if type == 'Relation Reasoning-Image':
                # continue
                image_name = sample['image_name']
                image_path = join(data_root,'images',image_name)
                if not exists(image_path):
                    continue
            time = sample['time']
            category = sample['category']
            video_name = sample['video_name']
            dirs = video_name.split('.')
            video_path = join(data_root,'merged',category,dirs[1],video_name)
            if not exists(video_path):
                continue

            user_content_list = []

            if type == 'Relation Reasoning-Image':
                user_content_list.append({'type':'text','text':'I will give you a reference image, keep it in mind. This is image:\n'})
                user_content_list.append({'type':'image','image':image_path})
                user_content_list.append({'type':'text','text':'<split>'})

            user_content_list.append({
                'type':'text',
                'text':'I will provide you with some video clips.\n'
            })
            window_size = 15
            shot_nums = int(time // window_size)
            if time % window_size != 0:
                shot_nums += 1
            if shot_nums == 0:
                continue
            for i in range(shot_nums):
                user_content_list.append({'type':'video','video':video_path,'shot_nums':shot_nums})
                user_content_list.append({'type':'text','text':' Look this video clip.\n'})
                user_content_list.append({'type':'text','text':'<split>'})
            
            qa = f'Question: {question}\n'
            for choice_id, choice in enumerate(choices):
                qa += f'({chr(ord("A")+choice_id)}): {choice}\n'
            user_content_list.append({
                'type':'text',
                'text':qa
            })
            gpt_output = f'({chr(ord("A")+answer)})'
            conv = [
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':gpt_output}
            ]
            self.samples.append(
                {
                    'conv':conv,
                    'metadata':{
                        'type':type,
                        'category':category,
                        'time':time
                    }
                }
            )
            tot += 1
        
        pbar.close()
        print(f'video_vista samples: {tot}')
        self.tot += tot


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
        # print(mm_infos)
        ## Read images or videos or audio
        image_inputs = []
        video_inputs = []
        audio_inputs = []
        for mm_info in mm_infos:
            if "image" in mm_info:
                ### random sample role image.
                files = os.listdir(mm_info['image'])
                file = random.sample(files,1)[0]
                if 'cropped' in file:
                    file = file.split('_')[0] + '.jpg'
                mm_info['image'] = join(mm_info['image'],file) # image_path

                mm_info.update({'resized_height':224,'resized_width':224})
                image_inputs.append(fetch_image(mm_info))
            
            elif "video" in mm_info:
                if self.use_memory:
                    if len(video_inputs) == 0: # only process at first time.
                        # print('start process video')
                        assert 'merged_shot_list' in mm_info or 'shot_nums' in mm_info, mm_info
                        mm_info.update({'resized_height':224,'resized_width':224,'fps':FPS})
                        # mm_info.update({'resized_height':224,'resized_width':224,'nframes': N_FRAMES})
                        shot_video_list = fetch_video(mm_info)
                        video_inputs = shot_video_list
                        # print('process video finished...')
                else:
                    mm_info.update({'resized_height':224,'resized_width':224,'fps':1.0})
                    # mm_info.update({'resized_height':224,'resized_width':224,'nframes':540,'window_size':30})
                    video_inputs = fetch_video(mm_info)
            
            elif 'audio' in mm_info:
                if self.use_memory:
                    if len(audio_inputs) == 0:
                        # print('==== start process audio....')
                        shot_audio_list = fetch_audio(mm_info)
                        audio_inputs = shot_audio_list
                        # print('====process audio finishedd...')
                else:
                    audio_inputs.append(fetch_audio(mm_info))
            
            else:
                raise ValueError("image, audio or video should in content.")
        return image_inputs, audio_inputs, video_inputs


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
                    _input_id += self.tokenize(prefix) + self.tokenize(f'<{tag}>') + self.tokenize(f'<{tag}_pad>') + self.tokenize(f'</{tag}>') + nl_tokens
                    input_text += f'{prefix}<{tag}><{tag}_pad></{tag}>\n'
                else:
                    _input_id += self.tokenize(f'<{tag}>') + self.tokenize(f'<{tag}_pad>') + self.tokenize(f'</{tag}>') + nl_tokens
                    input_text += f'<{tag}><{tag}_pad></{tag}>\n'
        
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


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]
        conv = sample['conv']
        metadata = sample.get('metadata',{})
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
            if not self.use_memory:
                video_token_nums = [sum(video_token_nums)]
        if len(audio_inputs) > 0:
            processed_audio = self.audio_processor(audio_inputs)
            spec_list = processed_audio['spec_list'] # [ [seg1, seg2,..], [seg1, seg2, ...] ]
            fbank_list = processed_audio['fbank_list']
            audio_token_nums = processed_audio['audio_token_nums'] # [88 * seg_nums, ... ]
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



@dataclass
class DataCollatorForMovieDataset(object):
    tokenzer: Qwen2Tokenizer
    mode: str = 'train'
    def __call__(self, instances: Sequence[Dict]):
        tokenizer = self.tokenzer
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
        # print(image_grid_thw)
        if len(image_grid_thw) > 0:
            image_grid_thw = torch.cat(image_grid_thw,dim=0)
        if len(video_grid_thw) > 0:
            video_grid_thw = torch.cat(video_grid_thw,dim=0)
        data = {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask':attention_mask,
            'batch_X_data':batch_X_data,
            'image_grid_thw':image_grid_thw,
            'video_grid_thw':video_grid_thw,
        }
        if self.mode == 'test':
            data['batch_metadata'] = batch_metadata
        # data['batch_metadata'] = batch_metadata
        return data



