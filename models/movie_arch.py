import torch
from torch import nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

from utils.util import nested_to_dtype


from models.multimodel_encoder import (
    EVAEncoder,
    AuidoEncoder,
    AudioVisualAlignmentModule,
    AudioFormer,
    VisualFormer,
    AudioWindowQFormer,
    VisualWindowQFormer
)

class MovieMetaModel:
    def __init__(self, config):
        super(MovieMetaModel, self).__init__(config)
        self.config = config


    def init_multimodal_modules(self,d_model = 4096, use_align = True, 
                                audio_branch = True, visual_branch=True):
        visual_dim = 768
        speech_dim = 1280
        beats_dim = 768
        if visual_branch:
            self.visual_encoder = EVAEncoder()
        if audio_branch:
            self.audio_encoder = AuidoEncoder()
            # self.speech_pre_qformer_proj = nn.Linear(speech_dim,visual_dim)
            # self.beats_pre_qformer_proj = nn.Linear(beats_dim,visual_dim)

        self.use_align = use_align
        if use_align:
            self.alignment_model = AudioVisualAlignmentModule(groupsize=10,low_groupsize=1,visual_dim=visual_dim,
                                                              d_model=d_model,num_video_query_token=30,
                                                              num_hidden_layers=2)
        else:
            if visual_branch:
                # self.visual_qformer = VisualFormer(groupsize=10,low_groupsize=1,visual_dim=768,
                #                                 hidden_size=768,d_model=d_model,
                #                                 num_query_token=30,
                #                                 num_hidden_layers=2)
                self.visual_qformer = VisualWindowQFormer()
            if audio_branch:
                # self.audio_qformer = AudioFormer(groupsize=10,low_groupsize=1,speech_dim=1280,
                #                                 beats_dim=768,hidden_size=768,d_model=d_model,
                #                                 num_query_token=30,num_hidden_layers=2)
                self.audio_qformer = AudioWindowQFormer()


    def encode_video(self,video):
        embeds = self.visual_encoder(video) # b, t * 32, dim
        if self.use_align:
            audio_len = embeds.size(1) // 32 * 25
            audio_embeds = embeds.new_zeros(embeds.size(0), audio_len, embeds.size(2) * 2)
            inputmasks = torch.tensor([1, 0]).unsqueeze(0).repeat(embeds.size(0), 1).to(embeds.device)
            embeds = self.alignment_model(embeds, audio_embeds, inputmasks = inputmasks)
        else:
            embeds = self.visual_qformer(embeds)
        return embeds
  
    
    def encode_audio(self, spec, fbank):
        speech_embeds, beats_features = self.audio_encoder(spec, fbank)
        # speech_embeds = self.speech_pre_qformer_proj(speech_embeds)
        # if beats_features.size(1) < speech_embeds.size(1):
        #     beats_features = F.pad(beats_features, (0, 0, 0, speech_embeds.size(1) - beats_features.size(1)), 'constant', 0).to(speech_embeds.device)
        # beats_features = self.beats_pre_qformer_proj(beats_features)
        # audio_embeds = torch.cat([speech_embeds, beats_features], dim=-1) # b, t * 50, dim
        if self.use_align:
            video_len = audio_embeds.size(1) // 25 * 32
            video_embeds = audio_embeds.new_zeros(audio_embeds.size(0), video_len, audio_embeds.size(2) // 2)
            inputmasks = torch.tensor([0, 1]).unsqueeze(0).repeat(audio_embeds.size(0), 1).to(audio_embeds.device)
            audio_embeds = self.alignment_model(video_embeds, audio_embeds, inputmasks = inputmasks)
        else:
            # audio_embeds = self.audio_qformer(audio_embeds)
            audio_embeds = self.audio_qformer(speech_embeds,beats_features)
        return audio_embeds
        
    

    def encode_image(self,image):
        image = image.unsqueeze(1) # b,1,c,h,w
        embeds = self.visual_encoder(image) # b, 1 * 32, dim
        embeds = self.visual_qformer(embeds)
        return embeds
    

    def encode_audio_video(self, spec, fbank, video):
        video_embeds = self.visual_encoder(video) # b, t * 32, dim
        speech_embeds, beats_features = self.audio_encoder(spec, fbank)
        speech_embeds = self.speech_pre_qformer_proj(speech_embeds)
        if beats_features.size(1) < speech_embeds.size(1):
            beats_features = F.pad(beats_features, (0, 0, 0, speech_embeds.size(1) - beats_features.size(1)), 'constant', 0).to(speech_embeds.device)
        beats_features = self.beats_pre_qformer_proj(beats_features)
        audio_embeds = torch.cat([speech_embeds, beats_features], dim=-1) # b, t * 50, dim
        if self.use_align:
            av_embeds = self.alignment_model(video_embeds, audio_embeds)
        else:
            raise ValueError('encode av must use align.')
        return av_embeds


    def align_av(self, audio_embeds, video_embeds, inputmasks = None):
        return self.alignment_model(video_embeds,audio_embeds,inputmasks = inputmasks)


class MovieMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> MovieMetaModel:
        pass
    
    def encode_audio(self, spec, fbank, batch_first=False):
        if not batch_first:
            spec = [item.unsqueeze(0) for item in spec]
            fbank = [item.unsqueeze(0) for item in fbank]
        # if not self.get_model().use_align:
        #     total_embeds = []
        #     total_video_embeds = []
        #     for spec_seg, fbank_seg in zip(spec,fbank):
        #         audio_embeds = self.get_model().encode_audio(spec_seg, fbank_seg)
        #         video_seg = torch.zeros([1,10,3,224,224],dtype=spec_seg.dtype,device=spec_seg.device)
        #         video_embeds = self.get_model().encode_video(video_seg)
        #         total_embeds.append(audio_embeds)
        #         total_video_embeds.append(video_embeds)
        #     # total_av_embeds = torch.cat(total_av_embeds,dim=1)
        #     total_embeds = torch.cat(total_embeds,dim=1)
        #     total_video_embeds = torch.cat(total_video_embeds,dim=1)
        #     total_embeds = total_embeds + total_video_embeds.sum() * 0
        #     if not batch_first:
        #         total_embeds = total_embeds.squeeze(0)
        # else:

        total_embeds = []
        for spec_seg, fbank_seg in zip(spec,fbank):
            embeds = self.get_model().encode_audio(spec_seg,fbank_seg)
            total_embeds.append(embeds)
        total_embeds = torch.cat(total_embeds,dim=1)
        if not batch_first:
            total_embeds = total_embeds.squeeze(0)
        return total_embeds


    def encode_video(self,video,batch_first=False):
        if not batch_first:
            video = [item.unsqueeze(0) for item in video]
        # total_embeds = []
        # total_audio_embeds = []
        # for video_seg in video:
        #     embeds = self.get_model().encode_video(video_seg)
        #     spec = torch.zeros([1,80, 3000],dtype=video_seg.dtype,device=video_seg.device)
        #     fbank = torch.zeros([1,100,128],dtype=video_seg.dtype,device = video_seg.device)
        #     audio_embeds = self.get_model().encode_audio(spec,fbank)
        #     total_embeds.append(embeds)
        #     total_audio_embeds.append(audio_embeds)
        # total_embeds = torch.cat(total_embeds, dim=1)
        # total_audio_embeds = torch.cat(total_audio_embeds,dim=1)
        # total_embeds = total_embeds + total_audio_embeds.sum() * 0
        # if not batch_first:
        #     # embeds = embeds.squeeze(0)
        #     total_embeds = total_embeds.squeeze(0)
        # # return embeds
        # return total_embeds

        total_embeds = []
        for video_seg in video:
            embeds = self.get_model().encode_video(video_seg)
            total_embeds.append(embeds)
        total_embeds = torch.cat(total_embeds,dim=1)
        if not batch_first:
            total_embeds = total_embeds.squeeze(0)
        return total_embeds


    def encode_image(self,image,batch_first=False):
        if not batch_first: # c, h, w
            image = image.unsqueeze(0).unsqueeze(0)
        else: # b, c, h, w
            image = image.unsqueeze(1)
        embeds = self.get_model().encode_video(image)
        if not batch_first:
            embeds = embeds.squeeze(0)
        return embeds


    def encode_audio_video(self, spec, fbank, video, batch_first = False):
        if not batch_first:
            spec = [item.unsqueeze(0) for item in spec]
            fbank = [item.unsqueeze(0) for item in fbank]
            video = [item.unsqueeze(0) for item in video]
        total_embeds = []
        for spec_seg, fbank_seg, video_seg in zip(spec, fbank, video):
            embeds = self.get_model().encode_audio_video(spec_seg,fbank_seg,video_seg)
            total_embeds.append(embeds)
        total_embeds = torch.cat(total_embeds,dim=1)
        if not batch_first:
            total_embeds = total_embeds.squeeze(0)
        return total_embeds
    

    def encode_av(self,spec, fbank ,video,batch_first= False):
        # assert audio is not None or video is not None
        # inputmasks = None
        # if audio is None:
        #     if not batch_first:
        #         video = [item.unsqueeze(0) for item in video]
        #     total_av_embeds = []
        #     for video_seg in video:
        #         video_embeds = self.encode_video(video_seg, batch_first=True)    
        #         audio_len = video_embeds.size(1) // 32 * 25
        #         audio_embeds = video_embeds.new_zeros(video_embeds.size(0), audio_len, video_embeds.size(2) * 2)
        #         inputmasks = torch.tensor([1, 0]).unsqueeze(0).repeat(video_embeds.size(0), 1).to(video_embeds.device)
        #         av_embeds = self.get_model().align_av(audio_embeds, video_embeds, inputmasks = inputmasks)
        #         total_av_embeds.append(av_embeds)
        #     total_av_embeds = torch.cat(total_av_embeds,dim=1)
        #     if not batch_first:
        #         total_av_embeds = total_av_embeds.squeeze(0)
        #     return total_av_embeds
        
        # elif video is None:
        #     spec = audio[0]
        #     fbank = audio[1]
        #     if not batch_first:
        #         spec = [item.unsqueeze(0) for item in spec]
        #         fbank = [item.unsqueeze(0) for item in fbank]
        #     total_av_embeds = []
        #     for spec_seg, fbank_seg in zip(spec,fbank):
        #         audio_embeds = self.encode_audio(spec_seg,fbank_seg, batch_first=True)
        #         video_len = audio_embeds.size(1) // 25 * 32
        #         video_embeds = audio_embeds.new_zeros(audio_embeds.size(0), video_len, audio_embeds.size(2) // 2)
        #         inputmasks = torch.tensor([0, 1]).unsqueeze(0).repeat(audio_embeds.size(0), 1).to(audio_embeds.device)
        #         av_embeds = self.get_model().align_av(audio_embeds, video_embeds, inputmasks = inputmasks)
        #         total_av_embeds.append(av_embeds)
        #     total_av_embeds = torch.cat(total_av_embeds,dim=1)
        #     if not batch_first:
        #         total_av_embeds = total_av_embeds.squeeze(0)
        #     return total_av_embeds
        # else:
        #     if not batch_first:
        #         spec, fbank = audio[0], audio[1]
        #         spec = [item.unsqueeze(0) for item in spec]
        #         fbank = [item.unsqueeze(0) for item in fbank]
        #         video = [item.unsqueeze(0) for item in video]
        #     total_av_embeds = []
        #     for spec_seg, fbank_seg, video_seg in zip(spec, fbank, video):
        #         video_embeds = self.encode_video(video_seg, batch_first=True)
        #         audio_embeds = self.encode_audio(spec_seg,fbank_seg, batch_first=True)
        #         av_embeds = self.get_model().align_av(audio_embeds, video_embeds, inputmasks = inputmasks)
        #         total_av_embeds.append(av_embeds)
        #     total_av_embeds = torch.cat(total_av_embeds,dim=1)
        #     if not batch_first:
        #         total_av_embeds = total_av_embeds.squeeze(0)
        #     return total_av_embeds

        if not batch_first:
            # spec, fbank = audio[0], audio[1]
            spec = [item.unsqueeze(0) for item in spec]
            fbank = [item.unsqueeze(0) for item in fbank]
            video = [item.unsqueeze(0) for item in video]
        total_audio_embeds = []
        total_video_embeds = []
        for spec_seg, fbank_seg, video_seg in zip(spec, fbank, video):
            # video_embeds = self.encode_video(video_seg, batch_first=True)
            # audio_embeds = self.encode_audio(spec_seg,fbank_seg, batch_first=True)
            video_embeds = self.get_model().encode_video(video_seg)
            audio_embeds = self.get_model().encode_audio(spec, fbank)
            # av_embeds = self.get_model().align_av(audio_embeds, video_embeds, inputmasks = inputmasks)
            # total_av_embeds.append(av_embeds)
            total_audio_embeds.append(audio_embeds)
            total_video_embeds.append(video_embeds)
        # total_av_embeds = torch.cat(total_av_embeds,dim=1)
        total_audio_embeds = torch.cat(total_audio_embeds,dim=1)
        total_video_embeds = torch.cat(total_video_embeds, dim=1)
        total_embeds = torch.cat([total_audio_embeds, total_video_embeds], dim=1)
        
        if not batch_first:
            # total_av_embeds = total_av_embeds.squeeze(0)
            total_video_embeds = total_video_embeds.squeeze(0)
            total_audio_embeds = total_audio_embeds.squeeze(0)
        
        return total_av_embeds


    def encode_ids(self,ids):
        return self.get_model().embed_tokens(ids)
    
    
    def prepare_multimodal_inputs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_data,
        key_tokens = None,
    ):
        device = self.device
        max_length = 0
        bs = len(batch_input_ids)
        new_batch_inputs_embeds = []
        new_batch_attention_mask = []
        new_batch_labels = []
        ## key_tokens = ['<audio>','<image>','<video>','<av>']
        key_tokens = self.get_model().key_tokens
        SPECIAL_TOKEN_2_IDS = self.get_model().SPECIAL_TOKEN_2_IDS
        IDS_2_SPECIAL_TOKEN = self.get_model().IDS_2_SPECIAL_TOKEN

        for i in range(bs):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]
            
            X_token_indices = torch.where(torch.any(torch.stack([input_ids == SPECIAL_TOKEN_2_IDS[key_token] for key_token in key_tokens]), dim=0))[0]
            X_token_indices = X_token_indices.tolist()

            inputs_embeds_seg = []
            labels_seg = []
            # attention_mask_seg = []
            pre_indice = 0
            a_idx, v_idx, i_idx = 0, 0, 0

            for idx, indice in enumerate(X_token_indices):
                inputs_embeds_seg.append(self.encode_ids(input_ids[pre_indice:indice]))
                labels_seg.append(labels[pre_indice:indice])

                special_token = IDS_2_SPECIAL_TOKEN[input_ids[indice].item()]
                if special_token == '<audio_pad>':
                    audio = batch_X_data[i]['audio'][a_idx]
                    spec, fbank = audio[0], audio[1]
                    embeds = self.encode_audio(spec, fbank)
                    inputs_embeds_seg.append(embeds)
                    labels_seg.append(torch.full((embeds.shape[0],),-100,dtype=torch.long,device=device))
                    a_idx += 1
                elif special_token == '<video_pad>':
                    video = batch_X_data[i]['video'][v_idx]
                    # embeds = self.encode_av(None,video)
                    embeds = self.encode_video(video)
                    inputs_embeds_seg.append(embeds)
                    labels_seg.append(torch.full((embeds.shape[0],),-100,dtype=torch.long,device=device))
                    v_idx += 1
                elif special_token == '<image_pad>':
                    image = batch_X_data[i]['image'][i_idx] # c,h,w
                    video = image.unsqueeze(0)
                    embeds = self.encode_video([video])
                    inputs_embeds_seg.append(embeds)
                    labels_seg.append(torch.full((embeds.shape[0],),-100,dtype=torch.long,device=device))
                    i_idx += 1
                elif special_token == '<av_pad>':
                    audio = batch_X_data[i]['audio'][a_idx]
                    spec, fbank = audio[0], audio[1]
                    a_idx += 1
                    video = batch_X_data[i]['video'][v_idx]
                    v_idx += 1
                    # embeds = self.encode_av(audio,video)
                    embeds = self.encode_audio_video(spec, fbank, video)
                    inputs_embeds_seg.append(embeds)
                    labels_seg.append(torch.full((embeds.shape[0],),-100,dtype=torch.long,device=device))

                pre_indice = indice + 1 # +1, skip special token

            # add last tokens
            inputs_embeds_seg.append(self.encode_ids(input_ids[pre_indice:]))
            labels_seg.append(labels[pre_indice:])
            # attention_mask_seg.append(attention_mask[pre_indice:])

            if a_idx > 0:
                assert a_idx == len(batch_X_data[i]['audio'])
            if v_idx > 0:
                assert v_idx == len(batch_X_data[i]['video'])
            if i_idx > 0:
                assert i_idx == len(batch_X_data[i]['image'])

            inputs_embeds_seg=torch.cat(inputs_embeds_seg,dim=0)
            attention_mask_seg=torch.ones((inputs_embeds_seg.shape[0]),dtype=torch.int32,device=device)
            labels_seg=torch.cat(labels_seg,dim=0)

            new_batch_inputs_embeds.append(inputs_embeds_seg)
            new_batch_attention_mask.append(attention_mask_seg)
            new_batch_labels.append(labels_seg)

            max_length=max(max_length,inputs_embeds_seg.shape[0])

        ### left padding
        padding_inputs_embeds = []
        padding_attention_mask = []
        padding_labels = []
        for i in range(bs):
            embeds = new_batch_inputs_embeds[i]
            mask = new_batch_attention_mask[i]
            labels = new_batch_labels[i]
            L,d = embeds.shape
            pad_embeds = self.encode_ids(torch.full((max_length-L,),self.get_model().pad_token_id,dtype=torch.long,device=device))
            padding_inputs_embeds.append(torch.cat([pad_embeds,embeds],dim=0))
            padding_attention_mask.append(torch.cat([torch.zeros((max_length-L),dtype=torch.int32,device=device),mask],dim=0))
            padding_labels.append(torch.cat([torch.full((max_length-L,),-100,dtype=torch.long,device=device),labels],dim=0))

        padding_inputs_embeds = torch.stack(padding_inputs_embeds,dim=0)
        padding_attention_mask = torch.stack(padding_attention_mask,dim=0)
        padding_labels = torch.stack(padding_labels,dim=0)

        position_ids = torch.cumsum(padding_attention_mask,dim=-1) - 1
        position_ids[position_ids==-1] = 0

        ### right padding ###
        # padding_inputs_embeds = []
        # for i in range(bs):
        #     emb = new_batch_inputs_embeds[i]
        #     L,d = emb.shape
        #     pad_embeds = self.encode_ids(torch.full((max_length-L,),self.get_model().pad_token_id,dtype=torch.long,device=device))
        #     padding_inputs_embeds.append(torch.cat([emb,pad_embeds],dim=0))
        # padding_inputs_embeds = torch.stack(padding_inputs_embeds,dim=0)
        # padding_attention_mask = pad_sequence(new_batch_attention_mask,batch_first=True,padding_value=0)
        # padding_labels = pad_sequence(new_batch_labels,batch_first=True,padding_value=-100)
        # position_ids = None

        # print('inputs_embeds: ',padding_inputs_embeds.shape,'  labels: ',labels.shape)
        
        return {
            'input_ids':None,
            'inputs_embeds':padding_inputs_embeds,
            'attention_mask':padding_attention_mask,
            'labels':padding_labels,
            'position_ids':position_ids,
        }
    
    '''
        Convert single <image_pad>,<audio_pad>, etc. multimodal tokens into multiple 
        tokens according to their embeds.
        Return full input_ids,... and X_embeds.
    '''
    def prepare_multimodal_full_input_ids(self,batch_input_ids, batch_labels, batch_X_data):
        device = self.device
        max_length = 0
        bs = batch_input_ids.shape[0]
        batch_full_inputs_ids = []
        batch_full_attention_mask = []
        batch_full_labels = []
        batch_X_embeds = []
        ## key_tokens = ['<audio>','<image>','<video>','<av>']
        key_tokens = self.get_model().key_tokens
        SPECIAL_TOKEN_2_IDS = self.get_model().SPECIAL_TOKEN_2_IDS
        IDS_2_SPECIAL_TOKEN = self.get_model().IDS_2_SPECIAL_TOKEN
        image_token_ids = SPECIAL_TOKEN_2_IDS['<image_pad>']
        audio_token_ids = SPECIAL_TOKEN_2_IDS['<audio_pad>']
        video_token_ids = SPECIAL_TOKEN_2_IDS['<video_pad>']
        av_token_ids = SPECIAL_TOKEN_2_IDS['<av_pad>']
        pad_token_id = self.get_model().pad_token_id

        for i in range(bs):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]
            
            X_token_indices = torch.where(torch.any(torch.stack([input_ids == SPECIAL_TOKEN_2_IDS[key_token] for key_token in key_tokens]), dim=0))[0]
            X_token_indices = X_token_indices.tolist()

            full_input_ids = []
            full_labels = []
            # X_embeds = {'image':[],'audio':[],'video':[],'av':[]}
            X_embeds = []
            pre_indice = 0
            a_idx, v_idx, i_idx = 0, 0, 0

            for idx, indice in enumerate(X_token_indices):
                full_input_ids.append(input_ids[pre_indice:indice])
                full_labels.append(labels[pre_indice:indice])
                special_token = IDS_2_SPECIAL_TOKEN[input_ids[indice].item()]

                if special_token == '<audio_pad>':
                    audio = batch_X_data[i]['audio'][a_idx]
                    spec, fbank = audio[0], audio[1]
                    embeds = self.encode_audio(spec, fbank)
                    X_embeds.append(embeds)
                    full_input_ids.append(self.full_token_ids((embeds.shape[0],),audio_token_ids,self.device))
                    full_labels.append(self.full_token_ids((embeds.shape[0],),-100,self.device))
                    a_idx += 1
                elif special_token == '<video_pad>':
                    video = batch_X_data[i]['video'][v_idx]
                    embeds = self.encode_video(video)
                    X_embeds.append(embeds)
                    full_input_ids.append(self.full_token_ids((embeds.shape[0],),video_token_ids,self.device))
                    full_labels.append(self.full_token_ids((embeds.shape[0],),-100,self.device))
                    v_idx += 1
                elif special_token == '<image_pad>':
                    image = batch_X_data[i]['image'][i_idx]
                    video = image.unsqueeze(0)
                    embeds = self.encode_video([video])
                    X_embeds.append(embeds)
                    full_input_ids.append(self.full_token_ids((embeds.shape[0],),image_token_ids,self.device))
                    full_labels.append(self.full_token_ids((embeds.shape[0],),-100,self.device))
                    i_idx += 1
                elif special_token == '<av_pad>':
                    audio = batch_X_data[i]['audio'][a_idx]
                    spec, fbank = audio[0], audio[1]
                    a_idx += 1
                    video = batch_X_data[i]['video'][v_idx]
                    v_idx += 1
                    # print('video: ',video[0].shape)
                    # embeds = self.encode_av(audio,video)
                    embeds = self.encode_audio_video(spec, fbank, video)
                    X_embeds.append(embeds)
                    full_input_ids.append(self.full_token_ids((embeds.shape[0],),av_token_ids,self.device))
                    full_labels.append(self.full_token_ids((embeds.shape[0],),-100,self.device))
                    
                pre_indice = indice + 1 # +1, skip special token

            # add last tokens
            full_input_ids.append(input_ids[pre_indice:])
            full_labels.append(labels[pre_indice:])

            if a_idx > 0:
                assert a_idx == len(batch_X_data[i]['audio'])
            if v_idx > 0:
                assert v_idx == len(batch_X_data[i]['video'])
            if i_idx > 0:
                assert i_idx == len(batch_X_data[i]['image'])

            full_input_ids = torch.cat(full_input_ids,dim=0)
            full_attention_mask = torch.ones((full_input_ids.shape[0]),dtype=torch.int32,device=device)
            full_labels = torch.cat(full_labels,dim=0)
            X_embeds = torch.cat(X_embeds, dim=0)

            max_length=max(max_length,full_input_ids.shape[0])

            batch_full_inputs_ids.append(full_input_ids)
            batch_full_attention_mask.append(full_attention_mask)
            batch_full_labels.append(full_labels)
            batch_X_embeds.append(X_embeds)

        ### left padding
        padding_input_ids = []
        padding_attention_mask = []
        padding_labels = []
        for i in range(bs):
            input_ids = batch_full_inputs_ids[i]
            attention_mask = batch_full_attention_mask[i]
            labels = batch_full_labels[i]
            L = input_ids.shape[0]
            pad_length = max_length - L
            padding_input_ids.append(torch.cat([self.full_token_ids((pad_length,),pad_token_id,self.device),input_ids],dim=0))
            padding_attention_mask.append(torch.cat([torch.zeros((max_length-L),dtype=torch.int32,device=device),attention_mask],dim=0))
            padding_labels.append(torch.cat([self.full_token_ids((pad_length,),-100,self.device),labels],dim=0))

        padding_input_ids = torch.stack(padding_input_ids,dim=0)
        padding_attention_mask = torch.stack(padding_attention_mask,dim=0)
        padding_labels = torch.stack(padding_labels,dim=0)

        position_ids = torch.cumsum(padding_attention_mask,dim=-1) - 1
        position_ids[position_ids==-1] = 0

        batch_X_embeds = torch.cat(batch_X_embeds,dim=0)

        return {
            'input_ids':padding_input_ids,
            'attention_mask':padding_attention_mask,
            'labels':padding_labels,
            'position_ids':position_ids,
            'batch_X_embeds':batch_X_embeds,
        }



    @property
    def device(self):
        return list(self.parameters())[0].device

    
    def full_token_ids(self,shape, value, device, dtype = torch.long):
        return torch.full(shape,value,dtype=dtype,device=device)

