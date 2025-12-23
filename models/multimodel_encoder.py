import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import normal_
from transformers import CLIPImageProcessor,CLIPVisionModel

from models.beats.BEATs import BEATs,BEATsConfig
from models.modeling_whisper import WhisperModel

from models.eva_vit import create_eva_vit_g
from models.Qformer import BertConfig,BertLMHeadModel


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class EVAEncoder(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        vit_model='eva_clip_g'
        image_size=224
        drop_path_rate=0.0
        use_grad_checkpoint=False
        vit_precision='fp32'
        vit_ckpt_path=''
        # vit_ckpt_path=None
        num_query_token=32
        # vit
        self.visual_encoder = self.init_vit(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision, vit_ckpt_path
        )
        self.visual_encoder.requires_grad_(False)
        self.ln_vision = nn.LayerNorm(self.visual_encoder.num_features) # 1408
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.visual_encoder.eval()

        # qformer
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        qformer_ckpt_path=''
        ckpt=torch.load(qformer_ckpt_path, map_location='cpu')
        self.load_state_dict(ckpt['model'],strict=False)
        
        self.Qformer.requires_grad_(False)
        self.Qformer.eval()
        self.query_tokens.requires_grad = False
        # self.Qformer.train = disabled_train


    def init_vit(self,model_name,img_size,drop_path_rate,use_grad_checkpoint,precision,ckpt_path):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
        return visual_encoder


    def init_Qformer(self, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("",local_files_only=True)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    

    def encode_video(self,video):
        with torch.no_grad():
            video_embeds = self.visual_encoder(video) # bt,n,d
        video_embeds = self.ln_vision(video_embeds)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video_embeds.device)
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        with torch.no_grad():
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                return_dict=True,
            )
        q_hidden_state = query_output.last_hidden_state # bt,q,h
        return q_hidden_state


    def encode_image(self,image):
        # b,c,1,h,w
        b,c,t,h,w = image.shape
        device = image.device
        image = rearrange(image,'b c t h w -> (b t) c h w')
        image = image.contiguous()
        with torch.no_grad():
            image_embeds = self.visual_encoder(image) # bt,n,d
        image_embeds = self.ln_vision(image_embeds)
        n = image_embeds.shape[1]
        image_embeds = rearrange(image_embeds,'(b t) n d -> b (t n) d',b=b,t=t)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.int32).to(device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        with torch.no_grad():
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        q_hidden_state = query_output.last_hidden_state # b,q,h
        return q_hidden_state
    

    @torch.no_grad()
    def forward(self,video):
        b,t,c,h,w = video.size()
        device = video.device
        video = video.reshape(b * t, c, h, w)
        video_embeds = self.visual_encoder(video) # bt,n,d
        video_embeds = self.ln_vision(video_embeds)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.int32).to(device)
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        q_hidden_state = query_output.last_hidden_state # bt,q,h
        q = q_hidden_state.shape[1]
        q_hidden_state = q_hidden_state.reshape(b,t * q,-1).contiguous()
        return q_hidden_state


class ViTEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path = '',
        select_layer = -1,
        select_feature = 'patch',
    ) -> None:
        super().__init__()
        
        self.select_layer = select_layer
        self.select_feature = select_feature

        self.image_processor = CLIPImageProcessor.from_pretrained(model_name_or_path,local_files_only=True)
        self.vision_tower = CLIPVisionModel.from_pretrained(model_name_or_path,local_files_only=True)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()


    def feature_select(self, image_forward_outs, select_feature=None, select_layer=None):
        select_layer = select_layer if select_layer is not None else self.select_layer
        select_feature = select_feature if select_feature is not None else self.select_feature
        image_features = image_forward_outs.hidden_states[select_layer]
        if select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {select_feature}')
        
        return image_features
    

    def encode_video(self,video,select_layer=None,select_feature=None):
        b,t,c,h,w = video.shape
        video = video.reshape(b*t,c,h,w)
        video_forward_outs = self.vision_tower(video, output_hidden_states=True)
        video_feature = self.feature_select(video_forward_outs, select_layer=select_layer, select_feature=select_feature)
        return video_feature


    @torch.no_grad()
    def forward(self,video,select_layer=None,select_feature=None):
        b,t,c,h,w = video.shape
        feature = self.encode_video(video, select_layer=select_layer, select_feature=select_feature) # bt,n,d
        n = feature.shape[1]
        feature = feature.reshape(b,t,n,-1)
        return feature



class AuidoEncoder(nn.Module):

    def __init__(self,) -> None:
        super().__init__()
        
        BEATs_ckpt_path = ''
        ckpt = torch.load(BEATs_ckpt_path,map_location='cpu')
        beats_cfg = BEATsConfig(ckpt['cfg'])
        beats_cfg.encoder_layerdrop = 0.  # not training, layerdrop = 0.
        self.beats_encoder = BEATs(cfg=beats_cfg)
        self.beats_encoder.load_state_dict(ckpt['model'],strict=True)
        self.beats_encoder.requires_grad_(False)
        self.beats_encoder.eval()
        self.beats_encoder.training = False
        
        # Whisper audio encoder
        whisper_ckpt_path=''
        self.whisper_encoder = WhisperModel.from_pretrained(whisper_ckpt_path,local_files_only=True).encoder
        self.whisper_encoder.requires_grad_(False)
        self.whisper_encoder.eval()


    @torch.no_grad()
    def forward(self,spec, fbank):
        # spec: (1,80,3000)  fbank: (1,len,128)
        speech_embeds = self.whisper_encoder(spec, return_dict=True).last_hidden_state
        beats_feature_mask = torch.zeros(fbank.shape[:-1],device=fbank.device).bool()
        beats_features = self.beats_encoder.extract_features(fbank,padding_mask=beats_feature_mask,feature_only=True)[0]
        
        return speech_embeds, beats_features



class AudioWhisperEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # Whisper audio encoder
        whisper_ckpt_path=''
        self.whisper_encoder = WhisperModel.from_pretrained(whisper_ckpt_path,local_files_only=True).encoder
        self.whisper_encoder.requires_grad_(False)
        self.whisper_encoder.eval()


    def encode_audio(self,spec,fbank):
        # sepc: bs, 80, len
        # fbank: bs, len, 128
        speech_embeds = self.whisper_encoder(spec,return_dict=True).last_hidden_state
        # audio_padding_mask = torch.zeros(fbank.shape[:-1],device=fbank.device).bool()
        # beats_embeds, _ = self.beats_encoder.extract_features(fbank, padding_mask=audio_padding_mask, feature_only=True)
        # return speech_embeds, beats_embeds
        return speech_embeds, None


    @torch.no_grad()
    def forward_whisper(self,audio):
        spec = audio['spec']  # list, [(80,len), (80,len), ...]
        speech_embeds = []
        for spec_seg in spec:
            spec_seg = spec_seg.unsqueeze(0)
            speech_embeds_seg = self.whisper_encoder(spec_seg,return_dict=True).last_hidden_state
            speech_embeds.append(speech_embeds_seg) # bs,L,1280
        speech_embeds = torch.cat(speech_embeds,dim=1)
        return speech_embeds


    @torch.no_grad()
    def forward(self,spec, fbank):
        # spec: (1,80,len)  fbank: (1,len,128)
        speech_embeds, beats_embeds = self.encode_audio(spec,fbank)
        return speech_embeds, beats_embeds


class AudioWindowQFormer(nn.Module):
    def __init__(
        self,
        speech_dim = 1280,
        beats_dim = 768,
        num_query_token = 1,
        second_per_window = 0.333333,
        second_stride = 0.333333,
        d_model = 4096,
    ):
        super().__init__()

        # from models.causal_qformer import BertConfig, BertLMHeadModel
        encoder_config = BertConfig.from_pretrained("",local_files_only=True)
        encoder_config.encoder_width = speech_dim + beats_dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        encoder_config.use_cache = False
        # encoder_config.gradient_checkpointing = True
        encoder_config.gradient_checkpointing = False
        self.Qformer = BertLMHeadModel(config=encoder_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None

        self.speech_ln = nn.LayerNorm(speech_dim)
        self.beats_ln = nn.LayerNorm(beats_dim)

        self.second_per_window = second_per_window
        self.second_stride = second_stride

        self.audio_proj = build_mlp(depth=2,hidden_size=768,output_hidden_size=d_model)
    

    def forward(self,speech_embeds, beats_embeds):
        speech_embeds = self.speech_ln(speech_embeds)
        beats_embeds = self.beats_ln(beats_embeds)
        
        if beats_embeds.size(1) < speech_embeds.size(1):
            beats_embeds = F.pad(beats_embeds, (0, 0, 0, speech_embeds.size(1) - beats_embeds.size(1)))
        elif beats_embeds.size(1) > speech_embeds.size(1):
            speech_embeds = F.pad(speech_embeds, (0, 0, 0, beats_embeds.size(1) - speech_embeds.size(1)))
        speech_embeds = torch.cat((speech_embeds, beats_embeds), dim=-1)
        # print('speech embeds: ',speech_embeds.shape)
        B, T, C = speech_embeds.shape
        kernel = round(1500 * self.second_per_window / 30.0)
        stride = round(1500 * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        query_tokens = self.query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.audio_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        return speech_embeds


class VisualWindowQFormer(nn.Module):
    def __init__(
        self,
        visual_dim = 768,
        num_query_token = 32,
        second_per_window = 10,
        second_stride = 10,
    ):
        super().__init__()

        # from models.causal_qformer import BertConfig, BertLMHeadModel
        encoder_config = BertConfig.from_pretrained("",local_files_only=True)
        encoder_config.encoder_width = visual_dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        encoder_config.use_cache = False
        # encoder_config.gradient_checkpointing = True
        encoder_config.gradient_checkpointing = False
        self.Qformer = BertLMHeadModel(config=encoder_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None

        # self.speech_ln = nn.LayerNorm(speech_dim)
        # self.beats_ln = nn.LayerNorm(beats_dim)

        self.visual_ln = nn.LayerNorm(visual_dim)

        self.second_per_window = second_per_window
        self.second_stride = second_stride

        self.visual_proj = build_mlp(depth=2,hidden_size=768,output_hidden_size=4096)
    

    def forward(self,visual_embeds):
        # visual_embeds: b, t * 32, dim
        B, seq_len, C = visual_embeds.shape
        visual_embeds = self.visual_ln(visual_embeds)

        visual_size = 32
        T = int(seq_len // visual_size)
        ngroups = T // self.second_per_window
        if T % self.second_per_window != 0:
            ngroups = ngroups + 1
            diff_T = ngroups * self.second_per_window - T
            concat_feature_paddings = visual_embeds.new_zeros(B, diff_T * visual_size, C)
            visual_embeds = torch.cat([visual_embeds, concat_feature_paddings], dim=1)
            T += diff_T

        kernel = self.second_per_window * visual_size
        stride = self.second_stride * visual_size
        kernel = (1, kernel)
        stride = (1, stride)
        visual_embeds_tr = visual_embeds.transpose(1, 2).unsqueeze(2) # b, dim, 1, seq_len
        visual_embeds_overlap = F.unfold(visual_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = visual_embeds_overlap.shape
        visual_embeds_overlap = visual_embeds_overlap.view(B, -1, kernel[1], L)
        visual_embeds_overlap = torch.permute(visual_embeds_overlap, [0, 3, 2, 1])
        visual_embeds = visual_embeds_overlap.reshape(-1, kernel[1], C)
        # print('ttt: ', visual_embeds.shape)
        visual_attns = torch.ones(visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
        query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=visual_attns,
            return_dict=True,
        )
        visual_embeds = self.visual_proj(query_output.last_hidden_state)
        visual_embeds = visual_embeds.view(B, -1, visual_embeds.size(2)).contiguous()
        return visual_embeds



'''
audio-viusal joint q-former
'''
class AudioVisualAlignmentModule(nn.Module):
    def __init__(
        self,
        groupsize = 10,
        low_groupsize = 1,
        visual_dim = 768,
        d_model = 4096,
        num_video_query_token = 30,
        num_hidden_layers = 2,
    ):
        super().__init__()

        self.groupsize = groupsize
        self.low_groupsize = low_groupsize

        self.ln_joint = nn.LayerNorm(visual_dim * 3)
        self.joint_frame_position_embedding = nn.Embedding(d_model, visual_dim * 3)
        self.num_video_query_token = num_video_query_token
        self.joint_Qformer, self.joint_query_tokens, self.low_query_tokens = self.init_video_Qformer(
            num_query_token = self.num_video_query_token,
            vision_width=visual_dim * 3,
            num_hidden_layers = num_hidden_layers,
            causal_encoder=True,
        )
        self.joint_Qformer.bert.embeddings.word_embeddings = None
        self.joint_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.joint_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.joint_Qformer.cls = None

        self.llama_proj_joint = nn.Linear(self.joint_Qformer.config.hidden_size * 2, d_model)

    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, 
                           num_hidden_layers=2, causal_encoder=False, cache_dir=""):
        
        from models.causal_qformer import BertConfig, BertLMHeadModel
        
        ckpt_path = ''
        encoder_config = BertConfig.from_pretrained(ckpt_path)
        if num_hidden_layers > 0:
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.cross_attention_freq = 1
            encoder_config.causal_encoder = causal_encoder
        else:
            encoder_config.cross_attention_freq = 2
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.query_length = num_query_token
        encoder_config.use_cache = False
        # encoder_config.gradient_checkpointing = True
        encoder_config.gradient_checkpointing = False
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        low_query_tokens = nn.Parameter(
            torch.zeros(1, 3, 768)
        )
        low_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        return Qformer, query_tokens, low_query_tokens


    def forward(self,video_embeds, audio_embeds, inputmasks = None):
        '''
            audio_embeds: b, t2 * 25, dim
            video embeds: b, t1 * 32, dim
        '''
        video_size = 32
        audio_size = 25
        bsize = video_embeds.size(0)
        vid_T = video_embeds.size(1) // video_size
        video_embeds = video_embeds.view(bsize, -1, video_size, video_embeds.size(-1))
        audio_embeds = audio_embeds.view(bsize, -1, audio_size, audio_embeds.size(-1))
        aud_T = audio_embeds.size(1)
        ### total time padding
        if aud_T > vid_T:
            diff_T = aud_T - vid_T
            video_pad = video_embeds.new_zeros(bsize, diff_T, video_size, video_embeds.size(-1))
            video_embeds = torch.cat([video_embeds, video_pad], dim=1)
        elif aud_T < vid_T:
            diff_T = vid_T - aud_T
            audio_pad = audio_embeds.new_zeros(bsize, diff_T, audio_size, audio_embeds.size(-1))
            audio_embeds = torch.cat([audio_embeds, audio_pad], dim=1)
        
        ### padding for every second
        audio_token_padding = audio_embeds.new_zeros(bsize, audio_embeds.size(1), video_size - audio_size, audio_embeds.size(-1))
        audio_embeds = torch.cat([audio_embeds, audio_token_padding], dim=2)
        
        # video_token_padding = video_embeds.new_zeros(bsize, video_embeds.size(1), audio_size - video_size, video_embeds.size(-1))
        # video_embeds = torch.cat([video_embeds, video_token_padding], dim=2)
        # video_size = audio_size
        
        if inputmasks is not None: # modality missing
            video_embeds = video_embeds * inputmasks[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            audio_embeds = audio_embeds * inputmasks[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        concat_features = torch.cat([video_embeds, audio_embeds], dim=3).view(
            bsize, video_embeds.size(1) * video_size, -1)
        
        total_mask = concat_features.new_ones(concat_features.size()[:-1])
        vid_T = video_embeds.size(1)

        ngroups = vid_T // self.groupsize
        if vid_T % self.groupsize != 0:
            ngroups = ngroups + 1
            diff_T = ngroups * self.groupsize - vid_T
            concat_feature_paddings = concat_features.new_zeros(bsize, diff_T * video_size, concat_features.size(-1))
            concat_features = torch.cat([concat_features, concat_feature_paddings], dim=1)
            total_mask = torch.cat([total_mask, total_mask.new_zeros(bsize, diff_T * video_size)], dim=1)
            vid_T += diff_T
        
        if self.low_groupsize is not None:
            lgroups = vid_T // self.low_groupsize
            low_features = concat_features.view(bsize * lgroups, self.low_groupsize * video_size, concat_features.size(-1))
            low_mask = total_mask.view(bsize * lgroups, self.low_groupsize * video_size)

            low_embeds = self.ln_joint(low_features)  # B x 32*T_max x D
            position_ids = torch.arange(low_embeds.size(1), dtype=torch.long, device=low_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(low_embeds.size(0), -1)
            frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
            low_hidden_state = frame_position_embeddings + low_embeds
            low_query_tokens = self.low_query_tokens.expand(low_embeds.shape[0], -1, -1)

            low_query_output = self.joint_Qformer.bert(
                query_embeds=low_query_tokens,
                encoder_hidden_states=low_hidden_state,
                encoder_attention_mask=low_mask,
                return_dict=True,
                segsize=0,
            )
            low_embeds = low_query_output.last_hidden_state
        
        concat_features = concat_features.view(bsize * ngroups, self.groupsize * video_size, concat_features.size(-1))
        total_mask = total_mask.view(bsize * ngroups, self.groupsize * video_size)
        # Forward Q-Former
        total_embeds = self.ln_joint(concat_features)  # B x 32*T_max x D
        position_ids = torch.arange(total_embeds.size(1), dtype=torch.long, device=total_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(total_embeds.size(0), -1)
        frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
        frame_hidden_state = frame_position_embeddings + total_embeds
        joint_query_tokens = self.joint_query_tokens.expand(total_embeds.shape[0], -1, -1)

        av_query_output = self.joint_Qformer.bert(
            query_embeds=joint_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=total_mask,
            return_dict=True,
            segsize=0,
        )
        total_embeds = av_query_output.last_hidden_state
        
        total_embeds = total_embeds.reshape(bsize, -1, total_embeds.size(-1))
        low_embeds = low_embeds.reshape(bsize, -1, low_embeds.size(-1))
        # print('concat feature: ',concat_features.shape,'  lgroups: ',lgroups, ' ngroups: ',ngroups, ' tot embed: ',total_embeds.shape,' low embed: ',low_embeds.shape)
        
        total_embeds = torch.cat((low_embeds, total_embeds), dim=-1)
        inputs_llama = self.llama_proj_joint(total_embeds)
        
        return inputs_llama

'''
audio-qformer
# '''
class AudioFormer(nn.Module):
    def __init__(
        self,
        groupsize = 10,
        low_groupsize = 1,
        speech_dim = 1280,
        beats_dim = 768,
        hidden_size = 768,
        d_model = 4096,
        num_query_token = 30,
        num_hidden_layers = 2,
    ):
        super().__init__()

        self.groupsize = groupsize
        self.low_groupsize = low_groupsize

        # self.speech_pre_qformer_proj = nn.Linear(speech_dim,hidden_size)
        # self.beats_pre_qformer_proj = nn.Linear(beats_dim,hidden_size)

        self.ln_joint = nn.LayerNorm(hidden_size * 2)
        self.joint_frame_position_embedding = nn.Embedding(4096, hidden_size * 2)
        self.num_query_token = num_query_token
        self.joint_Qformer, self.joint_query_tokens, self.low_query_tokens = self.init_video_Qformer(
            num_query_token = self.num_query_token,
            vision_width=hidden_size * 2,
            num_hidden_layers = num_hidden_layers,
            causal_encoder=True,
        )
        self.joint_Qformer.bert.embeddings.word_embeddings = None
        self.joint_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.joint_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.joint_Qformer.cls = None

        self.llama_proj_joint = nn.Linear(self.joint_Qformer.config.hidden_size * 2, d_model)
        # self.llama_proj_joint = build_mlp(depth=2,hidden_size=self.joint_Qformer.config.hidden_size * 2,output_hidden_size=d_model)

    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, 
                           num_hidden_layers=2, causal_encoder=False, cache_dir=""):
        
        from models.causal_qformer import BertConfig, BertLMHeadModel
        ckpt_path = '/group/40061/cserdu/pretrain/google-bert-base-uncased'
        encoder_config = BertConfig.from_pretrained(ckpt_path)
        if num_hidden_layers > 0:
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.cross_attention_freq = 1
            encoder_config.causal_encoder = causal_encoder
        else:
            encoder_config.cross_attention_freq = 2
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.query_length = num_query_token
        encoder_config.use_cache = False
        # encoder_config.gradient_checkpointing = True
        encoder_config.gradient_checkpointing = False
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        low_query_tokens = nn.Parameter(torch.zeros(1, 3, 768))
        low_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        return Qformer, query_tokens, low_query_tokens


    def forward(self, audio_embeds):
        '''
            audio_embeds: b, t2 * 50, dim
        '''
        audio_size = 50
        bsize = audio_embeds.shape[0]
        vid_T = int(audio_embeds.shape[1] // audio_size)
        total_mask = torch.ones(audio_embeds.shape[:-1],dtype=torch.int32,device=audio_embeds.device)
        ngroups = vid_T // self.groupsize
        if vid_T % self.groupsize != 0:
            ngroups = ngroups + 1
            diff_T = ngroups * self.groupsize - vid_T
            concat_feature_paddings = audio_embeds.new_zeros(bsize, diff_T * audio_size, audio_embeds.size(-1))
            audio_embeds = torch.cat([audio_embeds, concat_feature_paddings], dim=1)
            total_mask = torch.cat([total_mask, total_mask.new_zeros(bsize, diff_T * audio_size)], dim=1)
            vid_T += diff_T
        
        if self.low_groupsize is not None:
            lgroups = vid_T // self.low_groupsize
            low_features = audio_embeds.view(bsize * lgroups, self.low_groupsize * audio_size, audio_embeds.size(-1))
            low_mask = total_mask.view(bsize * lgroups, self.low_groupsize * audio_size)

            low_embeds = self.ln_joint(low_features)  # B x 32*T_max x D
            position_ids = torch.arange(low_embeds.size(1), dtype=torch.long, device=low_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(low_embeds.size(0), -1)
            frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
            low_hidden_state = frame_position_embeddings + low_embeds
            low_query_tokens = self.low_query_tokens.expand(low_embeds.shape[0], -1, -1)

            low_query_output = self.joint_Qformer.bert(
                query_embeds=low_query_tokens,
                encoder_hidden_states=low_hidden_state,
                encoder_attention_mask=low_mask,
                return_dict=True,
                segsize=0,
            )
            low_embeds = low_query_output.last_hidden_state
        
        concat_features = audio_embeds.view(bsize * ngroups, self.groupsize * audio_size, audio_embeds.size(-1))
        total_mask = total_mask.view(bsize * ngroups, self.groupsize * audio_size)
        # Forward Q-Former
        total_embeds = self.ln_joint(concat_features)  # B x 32*T_max x D
        position_ids = torch.arange(total_embeds.size(1), dtype=torch.long, device=total_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(total_embeds.size(0), -1)
        frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
        frame_hidden_state = frame_position_embeddings + total_embeds
        joint_query_tokens = self.joint_query_tokens.expand(total_embeds.shape[0], -1, -1)

        av_query_output = self.joint_Qformer.bert(
            query_embeds=joint_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=total_mask,
            return_dict=True,
            segsize=0,
        )
        total_embeds = av_query_output.last_hidden_state
        
        total_embeds = total_embeds.reshape(bsize, -1, total_embeds.size(-1))
        low_embeds = low_embeds.reshape(bsize, -1, low_embeds.size(-1))
        # print('concat feature: ',concat_features.shape,'  lgroups: ',lgroups, ' ngroups: ',ngroups, ' tot embed: ',total_embeds.shape,' low embed: ',low_embeds.shape)
        
        total_embeds = torch.cat((low_embeds, total_embeds), dim=-1)
        inputs_llama = self.llama_proj_joint(total_embeds)
        
        return inputs_llama


'''
audio-qformer
# '''
class VisualFormer(nn.Module):
    def __init__(
        self,
        groupsize = 10,
        low_groupsize = 1,
        visual_dim = 768,
        hidden_size = 768,
        d_model = 4096,
        num_query_token = 30,
        num_hidden_layers = 2,
    ):
        super().__init__()

        self.groupsize = groupsize
        self.low_groupsize = low_groupsize

        # self.speech_pre_qformer_proj = nn.Linear(speech_dim,hidden_size)
        # self.beats_pre_qformer_proj = nn.Linear(beats_dim,hidden_size)

        self.ln_joint = nn.LayerNorm(visual_dim)
        self.joint_frame_position_embedding = nn.Embedding(4096, hidden_size)
        self.num_query_token = num_query_token
        self.joint_Qformer, self.joint_query_tokens, self.low_query_tokens = self.init_video_Qformer(
            num_query_token = self.num_query_token,
            vision_width=hidden_size,
            num_hidden_layers = num_hidden_layers,
            causal_encoder=True,
        )
        self.joint_Qformer.bert.embeddings.word_embeddings = None
        self.joint_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.joint_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.joint_Qformer.cls = None

        self.llama_proj_joint = nn.Linear(self.joint_Qformer.config.hidden_size * 2, d_model)
        # self.llama_proj_joint = build_mlp(depth=2,hidden_size=self.joint_Qformer.config.hidden_size * 2,output_hidden_size=d_model)

    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, 
                           num_hidden_layers=2, causal_encoder=False, cache_dir=""):
        
        from models.causal_qformer import BertConfig, BertLMHeadModel
        ckpt_path = ''
        encoder_config = BertConfig.from_pretrained(ckpt_path)
        if num_hidden_layers > 0:
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.cross_attention_freq = 1
            encoder_config.causal_encoder = causal_encoder
        else:
            encoder_config.cross_attention_freq = 2
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.query_length = num_query_token
        encoder_config.use_cache = False
        # encoder_config.gradient_checkpointing = True
        encoder_config.gradient_checkpointing = False
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        low_query_tokens = nn.Parameter(torch.zeros(1, 3, 768))
        low_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        return Qformer, query_tokens, low_query_tokens


    def forward(self, visual_embeds):
        '''
            visual_embeds: b, t2 * 32, dim
        '''
        visual_size = 32
        bsize = visual_embeds.shape[0]
        vid_T = int(visual_embeds.shape[1] // visual_size)
        total_mask = torch.ones(visual_embeds.shape[:-1],dtype=torch.int32,device=visual_embeds.device)
        ngroups = vid_T // self.groupsize
        if vid_T % self.groupsize != 0:
            ngroups = ngroups + 1
            diff_T = ngroups * self.groupsize - vid_T
            concat_feature_paddings = visual_embeds.new_zeros(bsize, diff_T * visual_size, visual_embeds.size(-1))
            visual_embeds = torch.cat([visual_embeds, concat_feature_paddings], dim=1)
            total_mask = torch.cat([total_mask, total_mask.new_zeros(bsize, diff_T * visual_size)], dim=1)
            vid_T += diff_T
        
        if self.low_groupsize is not None:
            lgroups = vid_T // self.low_groupsize
            low_features = visual_embeds.view(bsize * lgroups, self.low_groupsize * visual_size, visual_embeds.size(-1))
            low_mask = total_mask.view(bsize * lgroups, self.low_groupsize * visual_size)

            low_embeds = self.ln_joint(low_features)  # B x 32*T_max x D
            position_ids = torch.arange(low_embeds.size(1), dtype=torch.long, device=low_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(low_embeds.size(0), -1)
            frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
            low_hidden_state = frame_position_embeddings + low_embeds
            low_query_tokens = self.low_query_tokens.expand(low_embeds.shape[0], -1, -1)

            low_query_output = self.joint_Qformer.bert(
                query_embeds=low_query_tokens,
                encoder_hidden_states=low_hidden_state,
                encoder_attention_mask=low_mask,
                return_dict=True,
                segsize=0,
            )
            low_embeds = low_query_output.last_hidden_state
        
        concat_features = visual_embeds.view(bsize * ngroups, self.groupsize * visual_size, visual_embeds.size(-1))
        total_mask = total_mask.view(bsize * ngroups, self.groupsize * visual_size)
        # Forward Q-Former
        total_embeds = self.ln_joint(concat_features)  # B x 32*T_max x D
        position_ids = torch.arange(total_embeds.size(1), dtype=torch.long, device=total_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(total_embeds.size(0), -1)
        frame_position_embeddings = self.joint_frame_position_embedding(position_ids)
        frame_hidden_state = frame_position_embeddings + total_embeds
        joint_query_tokens = self.joint_query_tokens.expand(total_embeds.shape[0], -1, -1)

        av_query_output = self.joint_Qformer.bert(
            query_embeds=joint_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=total_mask,
            return_dict=True,
            segsize=0,
        )
        total_embeds = av_query_output.last_hidden_state
        
        total_embeds = total_embeds.reshape(bsize, -1, total_embeds.size(-1))
        low_embeds = low_embeds.reshape(bsize, -1, low_embeds.size(-1))
        # print('concat feature: ',concat_features.shape,'  lgroups: ',lgroups, ' ngroups: ',ngroups, ' tot embed: ',total_embeds.shape,' low embed: ',low_embeds.shape)
        
        total_embeds = torch.cat((low_embeds, total_embeds), dim=-1)
        inputs_llama = self.llama_proj_joint(total_embeds)
        
        return inputs_llama


