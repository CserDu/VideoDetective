import torch
from torch import nn
from abc import ABC, abstractmethod

from models.multimodel_encoder import (
    AuidoEncoder,
    AudioWindowQFormer,
)

class MovieMetaModel:
    def __init__(self, config):
        super(MovieMetaModel, self).__init__(config)
        self.config = config


    def init_multimodal_modules(self,d_model = 3584, audio_branch = True, visual_branch = True):
        if audio_branch:
            self.audio_encoder = AuidoEncoder()
            self.audio_qformer = AudioWindowQFormer(d_model = d_model)

    
    def encode_audio(self, spec, fbank):
        speech_embeds, beats_features = self.audio_encoder(spec, fbank)
        audio_embeds = self.audio_qformer(speech_embeds,beats_features)
        return audio_embeds


class MovieMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> MovieMetaModel:
        pass
    
    @abstractmethod
    def encode_image(self, image_inputs):
        pass

    @abstractmethod
    def encode_video(self, video_inputs):
        pass

    def encode_audio(self, audio_inputs, batch_first=False):
        spec, fbank = audio_inputs[0], audio_inputs[1]
        if not batch_first:
            spec = [item.unsqueeze(0) for item in spec]
            fbank = [item.unsqueeze(0) for item in fbank]
        total_embeds = []
        for spec_seg, fbank_seg in zip(spec,fbank):
            embeds = self.get_model().encode_audio(spec_seg,fbank_seg)
            total_embeds.append(embeds)
        total_embeds = torch.cat(total_embeds,dim=1)
        if not batch_first:
            total_embeds = total_embeds.squeeze(0)
        return total_embeds


    def encode_ids(self,ids):
        return self.get_model().embed_tokens(ids)
    

    def encode_X_data(self, batch_X_data):
        bs = len(batch_X_data)
        batch_image_embeds = []
        batch_video_embeds = []
        batch_audio_embeds = []
        for i in range(bs):
            X_data = batch_X_data[i]
            for key, value in X_data.items():
                if value is None:
                    continue
                if key == 'image':
                    # print('image grid thw: ',value['image_grid_thw'])
                    embeds = self.encode_image(value)
                    # print('image embeds: ',embeds.shape)
                    batch_image_embeds.append(embeds)
                elif key == 'video':
                    # print('video grid thw: ',value['video_grid_thw'])
                    video_grid_thw = value['video_grid_thw']
                    shot_nums = video_grid_thw.shape[0]
                    pixel_values_videos = value['pixel_values_videos']
                    st = 0
                    for i in range(shot_nums):
                        length = video_grid_thw[i].prod().item()
                        et = st + length
                        shot_data = {
                            'pixel_values_videos':pixel_values_videos[st:et],
                            'video_grid_thw':video_grid_thw[i].unsqueeze(0)
                        }
                        embeds = self.encode_video(shot_data)
                        # print('video: ',embeds.shape)
                        batch_video_embeds.append(embeds)
                        st = et

                    # embeds = self.encode_video(value)
                    # batch_video_embeds.append(embeds)
                elif key == 'audio':
                    spec_list, fbank_list = value[0], value[1]
                    for spec, fbank in zip(spec_list, fbank_list):
                        embeds = self.encode_audio([spec,fbank])
                        batch_audio_embeds.append(embeds)
                        # print('audio embeds: ',embeds.shape)
        
        batch_X_embeds = {}
        if len(batch_image_embeds) > 0:
            batch_image_embeds = torch.cat(batch_image_embeds,dim=0)
            batch_X_embeds['image'] = batch_image_embeds
            # print('batch_image_embeds: ', batch_image_embeds.shape)
        if len(batch_video_embeds) > 0:
            batch_video_embeds = torch.cat(batch_video_embeds,dim=0)
            batch_X_embeds['video'] = batch_video_embeds
            # print('batch_video: ',batch_video_embeds.shape)
        if len(batch_audio_embeds) > 0:
            batch_audio_embeds = torch.cat(batch_audio_embeds,dim=0)
            batch_X_embeds['audio'] = batch_audio_embeds
        
        return batch_X_embeds


    @property
    def device(self):
        return list(self.parameters())[0].device

    
    def full_token_ids(self,shape, value, device, dtype = torch.long):
        return torch.full(shape,value,dtype=dtype,device=device)


