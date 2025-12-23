import json
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange
import contextlib
from typing import Optional,List
from torch.nn.utils.rnn import pad_sequence
from packaging import version
from dataclasses import asdict 
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import is_torchdynamo_compiling

use_memory = True
infer = True
if use_memory:
    from models.qwen2_5_vl.configuration_qwen2_5_vl_memory import Qwen2_5_VLConfig
    from models.qwen2_5_vl.modeling_qwen2_5_vl_memory import (
        Qwen2_5_VLForConditionalGeneration, 
        Qwen2_5_VLModel,
        Qwen2_5_VisionTransformerPretrainedModel,
    )
else:
    from models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration, 
        Qwen2_5_VLModel,
        Qwen2_5_VisionTransformerPretrainedModel,
    )
from models.qwen2_5_vl.movie_arch import MovieMetaModel,MovieMetaForCausalLM


ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]

def get_size(x, dim = 2):
    size = x.shape[dim] if x is not None else 0
    return size


class MovieConfig(Qwen2_5_VLConfig):
    model_type = "moviellm"


class MovieModel(MovieMetaModel,Qwen2_5_VLModel):
    config_class = MovieConfig

    def __init__(self, config: MovieConfig):
        super(MovieModel, self).__init__(config)
        self.config = config


class MovieForCausalLM(MovieMetaForCausalLM,Qwen2_5_VLForConditionalGeneration):
    config_class = MovieConfig

    def __init__(self, config: MovieConfig):
        super().__init__(config)

        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        # self.model = Qwen2_5_VLModel(config)
        self.model = MovieModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()
    

    def _extract_past_from_model_output(self, outputs):
        # if not use_memory:
        #     return super()._extract_past_from_model_output(outputs)
        print('into extract_past_from_model_output...')
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
            cache = self.memory.beacon_activations
            print('past_key: ',past_key_values[0][0].shape,'  cache: ', cache[0][0].shape)
            new_cache = []
            for i in range(len(cache)):
                past_key, past_value, _, _  = past_key_values[i]
                cache_key, cache_value, *_ = cache[i]
                new_key = torch.cat([cache_key,past_key],dim=2)
                new_value = torch.cat([cache_value,past_value],dim=2)
                new_cache.append((new_key,new_value,0,None))
            past_key_values = new_cache
            self.memory.beacon_activations = new_cache
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"

        return cache_name, past_key_values


    def _get_initial_cache_position(self, input_ids, model_kwargs):
        if not use_memory:
            return super()._get_initial_cache_position(input_ids,model_kwargs)
        # print('into get initial cache position....')
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        # print('cache pos: ',cache_position)
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()
            # print('past_length: ',past_length)
            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]
        # print('cache pos: ',cache_position)
        model_kwargs["cache_position"] = cache_position
        return model_kwargs
    

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        if not use_memory:
            return super()._update_model_kwargs_for_generation(outputs,model_kwargs,
                                                               is_encoder_decoder=is_encoder_decoder,
                                                               num_new_tokens=num_new_tokens)
        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                    ### update past_key_values ###
                    past_key_values = outputs.past_key_values
                    cache = self.memory.beacon_activations
                    # print('past_key: ',past_key_values[0][0].shape,'  cache: ', cache[0][0].shape)
                    new_cache = []
                    for i in range(len(cache)):
                        past_key, past_value, _, _  = past_key_values[i]
                        cache_key, cache_value, *_ = cache[i]
                        new_key = torch.cat([cache_key,past_key],dim=2)
                        new_value = torch.cat([cache_value,past_value],dim=2)
                        new_cache.append((new_key,new_value,0,None))
                    past_key_values = new_cache
                    self.memory.beacon_activations = new_cache
                    model_kwargs[cache_name] = new_cache
                # model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs


    def get_model(self) -> MovieModel:
        return self.model
    

    def encode_image(self, image_inputs):
        pixel_values = image_inputs['pixel_values']
        grid_thw = image_inputs['image_grid_thw']
        # self.image_grid_thw = grid_thw
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        # print(f'image pixel values: ',pixel_values.shape,'  grid: ',grid_thw,'  image embeds: ',image_embeds.shape)
        return image_embeds


    def encode_video(self, video_inputs):
        pixel_values_videos = video_inputs['pixel_values_videos']
        video_grid_thw = video_inputs['video_grid_thw']
        # self.video_grid_thw = video_grid_thw
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds

    
    def forward(
        self,
        batch_X_data = None,
        ## infer
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        # print('input_ids: ', input_ids.shape)
        ### memory forward ###
        if use_memory:
            if infer :
                return self._native_forward(
                    batch_X_embeds = {},
                    ## infer
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_values = past_key_values,
                    inputs_embeds = inputs_embeds,
                    labels = labels,
                    use_cache = use_cache,
                    output_attentions =output_attentions,
                    output_hidden_states = output_hidden_states,
                    return_dict = return_dict,
                    pixel_values = None,
                    pixel_values_videos = None,
                    image_grid_thw = None,
                    video_grid_thw = None,
                    rope_deltas = rope_deltas,
                    cache_position = cache_position,
                    second_per_grid_ts = second_per_grid_ts,
                )

            return self._beacon_forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                labels = labels,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
                pixel_values = pixel_values,
                pixel_values_videos = pixel_values_videos,
                image_grid_thw = image_grid_thw,
                video_grid_thw = video_grid_thw,
                rope_deltas = rope_deltas,
                cache_position = cache_position,
                second_per_grid_ts = second_per_grid_ts,
                batch_X_data=batch_X_data,
            )
        

        ### lora forward ###
        if input_ids is not None and input_ids.shape[1]==1: ## infer
            # print('infer....')
            inputs_embeds = self.encode_ids(input_ids)
            input_ids = None
        
        else: # training or fisrt generation.
            batch_X_embeds = self.encode_X_data(batch_X_data) if batch_X_data is not None else {}
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # during training, get 3d position_ids. 
            # for infer, use official code.
            if not infer:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )

            if 'image' in batch_X_embeds:
                image_embeds = batch_X_embeds['image']
                n_image_tokens = (input_ids == self.get_model().image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                assert n_image_tokens == n_image_features, f'n_image_tokens: {n_image_tokens}, n_image_features: {n_image_features}'
                mask = input_ids == self.get_model().image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if 'video' in batch_X_embeds:
                video_embeds = batch_X_embeds['video']
                n_video_tokens = (input_ids == self.get_model().video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                assert n_video_tokens == n_video_features, f'n_video_tokens:{n_video_tokens}, n_video_features: {n_video_features}'
                mask = input_ids == self.get_model().video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if 'audio' in batch_X_embeds:
                audio_embeds = batch_X_embeds['audio']
                # print('audio embeds: ',audio_embeds.dtype)
                audio_embeds = audio_embeds.to(self.dtype)
                n_audio_tokens = (input_ids == self.get_model().audio_token_id).sum().item()
                n_audio_features = audio_embeds.shape[0]
                # print(f'n_audio_tokens: ',n_audio_tokens,'  n_audio_features: ',n_audio_features)
                assert n_audio_tokens == n_audio_features, f'n_audio_tokens: {n_audio_tokens}, n_audio_features: {n_audio_features}'
                mask = input_ids == self.get_model().audio_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                audio_mask = mask_expanded.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        
        return super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            pixel_values = pixel_values,
            pixel_values_videos = pixel_values_videos,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            rope_deltas = rope_deltas,
            cache_position = cache_position,
            second_per_grid_ts = second_per_grid_ts,
        )


    def _beacon_forward(
        self,
        batch_X_data = None,
        ## infer
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        tokens_2_ids = self.get_model().tokens_2_ids
        split_tokens = ['<split>']
        ### 1. get shot split indices
        split_indices = torch.where(torch.any(torch.stack([input_ids[0] == tokens_2_ids[key_token] for key_token in split_tokens]), dim=0))[0]
        split_indices = split_indices.tolist()
        split_indices = split_indices + [input_ids.shape[1]]

        ### 2. encode batch_X_data into batch_X_embeds
        with torch.no_grad():
            batch_X_embeds = self.encode_X_data(batch_X_data)
        
        ### 3. initialize cache
        self.memory.prepare(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            labels = labels,
            split_indices = split_indices,
        )

        ### 4. set self.model.X_start_idx = 0
        self.image_start = 0
        self.video_start = 0
        self.audio_start = 0
        
        step = 0
        ### 5. get shot input_ids every step.
        while not self.memory.finish:
            # input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()
            input_ids, past_key_values, attention_mask, position_ids, labels = self.memory.step()
            # n_image_tokens = (input_ids == self.get_model().image_token_id).sum().item()
            # n_video_tokens = (input_ids == self.get_model().video_token_id).sum().item()
            # n_audio_tokens = (input_ids == self.get_model().audio_token_id).sum().item()
            # print('step, n_video_tokens: ',n_video_tokens,' n_image_tokens: ',n_image_tokens,' n_audio_tokens: ',n_audio_tokens)
            ### prepare 3d position_ids
            past_key = past_key_values[0][0]
            past_len = past_key.shape[2] if past_key is not None else 0
            tmp_input_ids = torch.cat([
                input_ids.new_ones((input_ids.shape[0],past_len)),
                input_ids
            ],dim = 1)
            if step == 0:
                shot_image_grid_thw = image_grid_thw
                shot_video_grid_thw = None
            else:
                shot_image_grid_thw = None
                shot_video_grid_thw = video_grid_thw[step - 1 : step] if step <= video_grid_thw.shape[0] else None
            tmp_attention_mask = torch.ones_like(tmp_input_ids,dtype=torch.int32,device=input_ids.device)
            position_ids, rope_deltas = self.get_rope_index(
                tmp_input_ids,
                shot_image_grid_thw,
                shot_video_grid_thw,
                second_per_grid_ts,
                tmp_attention_mask,
            )
            # print('past_len: ',past_len, 'input_ids: ',input_ids.shape, 'pos: ',position_ids.shape)

            # print('step, input_ids: ', input_ids.shape,' attention_mask: ',attention_mask.shape,' label: ',labels.shape, ' pos: ',position_ids.shape)
            # beacon_size = past_key_values[0][2]
            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                # shift_labels=False,
                batch_X_embeds=batch_X_embeds,
            )
            # update past_key_values
            self.memory.update_memory(outputs.past_key_values)
            if labels is not None:
                # update loss
                self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)
            
            step += 1

        ### 6. assert all the X_embeds are used.
        if 'image' in batch_X_embeds:
            assert self.image_start == batch_X_embeds['image'].shape[0],(self.image_start,batch_X_embeds['image'].shape[0])
        if 'audio' in batch_X_embeds:
            assert self.audio_start == batch_X_embeds['audio'].shape[0],(self.audio_start,batch_X_embeds['audio'].shape[0])
        if 'video' in batch_X_embeds:
            assert self.video_start == batch_X_embeds['video'].shape[0],(self.video_start,batch_X_embeds['video'].shape[0])

        ### 7. output loss, past_key_values, and perplexity
        outputs = self.memory.output(outputs)

        ### 8. reset memory after process one sequence.
        self.memory.reset()

        return outputs


    def _native_forward(
        self,
        batch_X_embeds = None,
        ## infer
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        # print('native forward, cache pos: ',cache_position)
        past_key, past_value, beacon_size, beacon_indices = past_key_values[0]

        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-input_ids.shape[1]:]
            ordinal_input_ids = input_ids[:, cur_beacon_indices == 0]
            beacon_input_ids = input_ids[:, cur_beacon_indices > 0]
            beacon_input_embeds = self.model.beacon_embed_tokens(beacon_input_ids - self.config.vocab_size)
            ordinal_inputs_embeds = self.model.embed_tokens(ordinal_input_ids)

            if 'image' in batch_X_embeds:
                image_embeds = batch_X_embeds['image']
                n_image_tokens = (ordinal_input_ids == self.get_model().image_token_id).sum().item()
                if n_image_tokens > 0:
                    # n_image_features = image_embeds.shape[0]
                    # assert n_image_tokens == n_image_features
                    st = self.image_start
                    et = st + n_image_tokens
                    mask = ordinal_input_ids == self.get_model().image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(ordinal_inputs_embeds)
                    image_mask = mask_expanded.to(ordinal_inputs_embeds.device)
                    ordinal_inputs_embeds = ordinal_inputs_embeds.masked_scatter(image_mask, image_embeds[st:et])
                    self.image_start = et

            if 'video' in batch_X_embeds:
                video_embeds = batch_X_embeds['video']
                n_video_tokens = (ordinal_input_ids == self.get_model().video_token_id).sum().item()
                if n_video_tokens > 0:
                    # n_video_features = video_embeds.shape[0]
                    # assert n_video_tokens == n_video_features
                    st = self.video_start
                    et = st + n_video_tokens
                    mask = ordinal_input_ids == self.get_model().video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(ordinal_inputs_embeds)
                    video_mask = mask_expanded.to(ordinal_inputs_embeds.device)
                    ordinal_inputs_embeds = ordinal_inputs_embeds.masked_scatter(video_mask, video_embeds[st:et])
                    self.video_start = et

            if 'audio' in batch_X_embeds:
                audio_embeds = batch_X_embeds['audio']
                audio_embeds = audio_embeds.to(self.dtype)
                n_audio_tokens = (ordinal_input_ids == self.get_model().audio_token_id).sum().item()
                if n_audio_tokens > 0:
                    # n_audio_features = audio_embeds.shape[0]
                    # assert n_audio_tokens == n_audio_features
                    st = self.audio_start
                    et = st + n_audio_tokens
                    mask = ordinal_input_ids == self.get_model().audio_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(ordinal_inputs_embeds)
                    audio_mask = mask_expanded.to(ordinal_inputs_embeds.device)
                    ordinal_inputs_embeds = ordinal_inputs_embeds.masked_scatter(audio_mask, audio_embeds[st:et])
                    self.audio_start = et

            inputs_embeds = beacon_input_embeds.new_zeros(*input_ids.shape, beacon_input_embeds.shape[-1])
            inputs_embeds[:, cur_beacon_indices == 0] = ordinal_inputs_embeds
            inputs_embeds[:, cur_beacon_indices > 0] = beacon_input_embeds

        else:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if 'image' in batch_X_embeds:
                image_embeds = batch_X_embeds['image']
                n_image_tokens = (input_ids == self.get_model().image_token_id).sum().item()
                if n_image_tokens > 0:
                    # n_image_features = image_embeds.shape[0]
                    # assert n_image_tokens == n_image_features
                    st = self.image_start
                    et = st + n_image_tokens
                    mask = input_ids == self.get_model().image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds[st:et])

            if 'video' in batch_X_embeds:
                video_embeds = batch_X_embeds['video']
                n_video_tokens = (input_ids == self.get_model().video_token_id).sum().item()
                if n_video_tokens > 0:
                    # n_video_features = video_embeds.shape[0]
                    # assert n_video_tokens == n_video_features
                    st = self.video_start
                    et = st + n_video_tokens
                    mask = input_ids == self.get_model().video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds[st:et])

            if 'audio' in batch_X_embeds:
                audio_embeds = batch_X_embeds['audio']
                audio_embeds = audio_embeds.to(self.dtype)
                n_audio_tokens = (input_ids == self.get_model().audio_token_id).sum().item()
                if n_audio_tokens > 0:
                    # n_audio_features = audio_embeds.shape[0]
                    # assert n_audio_tokens == n_audio_features
                    st = self.audio_start
                    et = st + n_audio_tokens
                    mask = input_ids == self.get_model().audio_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    audio_mask = mask_expanded.to(inputs_embeds.device)
                    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds[st:et])

        # input_ids_with_mem = input_ids
        # mem_size = past_key.shape[2] if past_key is not None else 0
        # if mem_size > 0:
        #     mem_input_ids = input_ids.new_zeros((input_ids.shape[0],mem_size)) 
        #     input_ids_with_mem = torch.cat([mem_input_ids,input_ids],dim=1)
        
        # print('native forward: ',input_ids_with_mem.shape)
        return super().forward(
            input_ids = None,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            pixel_values = pixel_values,
            pixel_values_videos = pixel_values_videos,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            rope_deltas = rope_deltas,
            cache_position = cache_position,
            second_per_grid_ts = second_per_grid_ts,
        )


    @torch.no_grad()
    def generate(self,**kwargs):

        if use_memory:
            return self.beacon_generate(**kwargs)
        
        generated_ids = super().generate(**kwargs)
        return generated_ids, kwargs['input_ids']
    

    @torch.no_grad()
    def beacon_generate(
        self,
        batch_X_data = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        print('====== input_ids: ',input_ids.shape)
        self.token_nums = 0
        tokens_2_ids = self.get_model().tokens_2_ids
        split_tokens = ['<split>']
        ### 1. get shot split indices
        split_indices = torch.where(torch.any(torch.stack([input_ids[0] == tokens_2_ids[key_token] for key_token in split_tokens]), dim=0))[0]
        split_indices = split_indices.tolist()
        split_indices = split_indices + [input_ids.shape[1]]
        print('==== shot_nums: ', len(split_indices)-1)
        ### 2. encode batch_X_data into batch_X_embeds
        with torch.no_grad():
            batch_X_embeds = self.encode_X_data(batch_X_data)
        # print('grid thw: ',self.image_grid_thw,'  ',self.video_grid_thw)
        ### 3. initialize cache
        self.memory.prepare(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            labels = labels,
            split_indices = split_indices,
        )
        ### 4. set self.model.X_start_idx = 0
        self.image_start = 0
        self.video_start = 0
        self.audio_start = 0
        exist_images = True if isinstance(image_grid_thw, torch.Tensor) else False
        
        step = 0
        pbar = tqdm(total = len(split_indices) + 1, desc='infer')
        ### 5. get shot input_ids every step.
        while not self.memory.finish:
            pbar.update(1)
            # input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()
            input_ids, past_key_values, attention_mask, position_ids, labels = self.memory.step()
            # print('step, input_ids: ', input_ids.shape,' attention_mask: ',attention_mask.shape,' label: ',labels.shape, ' pos: ',position_ids.shape)
            beacon_size = past_key_values[0][2]
            # print('input_ids: ',input_ids.shape, ' beacon size: ', beacon_size)
            if beacon_size == 0: ## last shot, start generate
                # print('last shot, input_ids: ',input_ids,'  shape: ',input_ids.shape)
                all_past_key = self.memory.beacon_activations[0][0]
                all_past_beacon_size = all_past_key.shape[2] if all_past_key is not None else 0
                input_ids = torch.cat([
                    input_ids.new_zeros((input_ids.shape[0],all_past_beacon_size)),
                    input_ids
                ],dim=1)
                print('==== last shot, input_ids: ',input_ids.shape)
                attention_mask = torch.ones_like(input_ids,dtype=torch.int32,device=input_ids.device)
                output_ids = super().generate(
                    inputs = input_ids,
                    past_key_values = past_key_values,
                    attention_mask = attention_mask,
                    # position_ids = position_ids,
                    output_attentions = output_attentions,
                    **kwargs
                )
                # exit(-1)
            else:
                ### prepare 3d position_ids
                past_key = past_key_values[0][0]
                past_len = past_key.shape[2] if past_key is not None else 0
                tmp_input_ids = torch.cat([
                    input_ids.new_ones((input_ids.shape[0],past_len)),
                    input_ids
                ],dim = 1)
                ######## for qwen2_5_vl_memory, video_vista image relation task, infer #####
                if exist_images:
                    if step == 0:  # first step for role images.
                        shot_image_grid_thw = image_grid_thw
                        shot_video_grid_thw = None
                    else:
                        shot_image_grid_thw = None
                        shot_video_grid_thw = video_grid_thw[step - 1 : step] if step <= video_grid_thw.shape[0] else None
                else:
                    ###### for cinepile, longvideo_bench, video-mme infer #######
                    shot_image_grid_thw = None
                    shot_video_grid_thw = video_grid_thw[step: step + 1]
                
                # shot_image_grid_thw = None
                # shot_video_grid_thw = video_grid_thw[step: step + 1]

                tmp_attention_mask = torch.ones_like(tmp_input_ids,dtype=torch.int32,device=input_ids.device)
                # print('==== ',image_grid_thw,'  ',video_grid_thw)
                position_ids, rope_deltas = self.get_rope_index(
                    tmp_input_ids,
                    shot_image_grid_thw,
                    shot_video_grid_thw,
                    None,
                    tmp_attention_mask,
                )
                # print('beacon generate, input_ids: ',input_ids.shape, '  past_len: ',past_len, '  mask: ',attention_mask.shape,'  pos: ',position_ids)
                outputs = self._native_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    labels=labels,
                    # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                    # shift_labels=False,
                    batch_X_embeds=batch_X_embeds,
                )
            # update past_key_values
            # if beacon_size == 0:  # last shot
            #     print('beacon_size = 0, all_beacon_size: ', self.memory.beacon_activations[0][0].shape[2])
            #     exit(-1)
            #     new_past_key_values = []
            #     last_cache = outputs.past_key_values
            #     for i in range(len(last_cache)):
            #         last_key, last_value, _, _ = last_cache[i]
            #         beacon_key, beacon_value, _, _ = self.memory.beacon_activations[i]
            #         new_key = torch.cat([beacon_key,last_key], dim=2)
            #         new_value = torch.cat([beacon_value, last_value], dim=2)
            #         new_past_key_value = (new_key, new_value, 0, None)
            #         new_past_key_values.append(new_past_key_value)
            
            # else:
                self.memory.update_memory(outputs.past_key_values)
            # print('=======')
            # print('update, beacon_size: ', self.memory.beacon_activations[0][0].shape[2])
            # print('sink_size: ',get_size(self.memory.sink_activations[0][0]))
            # print('raw_size: ',get_size(self.memory.raw_activations[0][0]))
            
            # if labels is not None:
            #     # update loss
            #     self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)
            step += 1

        ### 6. assert all the X_embeds are used.
        if 'image' in batch_X_embeds:
            assert self.image_start == batch_X_embeds['image'].shape[0],(self.image_start,batch_X_embeds['image'].shape[0])
        if 'audio' in batch_X_embeds:
            assert self.audio_start == batch_X_embeds['audio'].shape[0],(self.audio_start,batch_X_embeds['audio'].shape[0])
        if 'video' in batch_X_embeds:
            assert self.video_start == batch_X_embeds['video'].shape[0],(self.video_start,batch_X_embeds['video'].shape[0])

        ### 7. output loss, past_key_values, and perplexity
        # outputs = self.memory.output(outputs)

        # output_ids = super().generate(
        #     inputs = input_ids,
        #     past_key_values = new_past_key_values,
        #     **kwargs
        # )

        ### 8. reset memory after process one sequence.
        self.memory.reset()

        return output_ids, input_ids


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        batch_X_data = None,
        **kwargs,
    ):
        if use_memory:
            return self.prepare_inputs_for_generation_with_memory(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                batch_X_data=batch_X_data,
                **kwargs
            )
        
        # print(' attention_mask: ',attention_mask,' cache_pos: ',cache_position,'  past_key_calues: ',past_key_values.get_seq_length(),'  inputs_embeds: ',inputs_embeds,'  pos: ',position_ids)
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "batch_X_data": batch_X_data,
            }
        )
        return model_inputs


    def prepare_inputs_for_generation_with_memory(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        batch_X_data = None,
        **kwargs,
    ):
        # print(f'into prepare inputs for generate with memory... input_ids: {input_ids.shape}, inputs_embeds: {inputs_embeds} pos: {position_ids} mask: {attention_mask.shape} cache_pos: {cache_position} past_key_values: ',past_key_values[0][0].shape)
        # self.token_nums += 1
        # if self.token_nums > 5:
        #     exit(-1)
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}
        
        # if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        # if attention_mask.ndim == 2:
        #     if model_inputs["inputs_embeds"] is not None:
        #         batch_size, sequence_length, _ = inputs_embeds.shape
        #         device = inputs_embeds.device
        #     else:
        #         batch_size, sequence_length = input_ids.shape
        #         device = input_ids.device

        #     target_length = past_key_values[0][0].shape[2] + sequence_length + 1
        #     attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
        #         attention_mask,
        #         sequence_length=sequence_length,
        #         target_length=target_length,
        #         dtype=self.lm_head.weight.dtype,
        #         device=device,
        #         cache_position=cache_position,
        #         batch_size=batch_size,
        #         config=self.config,
        #         past_key_values=past_key_values,
        #     )
        position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=attention_mask.device).repeat(attention_mask.shape[0], 1)

        past_key = past_key_values[0][0]
        mem_size = past_key.shape[2] if past_key is not None else 0
        tgt_size = attention_mask.size(-1) - mem_size
        dtype = torch.bfloat16
        min_value = torch.finfo(self.dtype).min
        device = attention_mask.device
        batch_size, src_size = attention_mask.size()

        # square for memory, and lower triangular for input_ids
        causal_mask = torch.full((tgt_size, tgt_size), min_value, device=device, dtype=dtype)
        mask_cond = torch.arange(causal_mask.size(-1), device=device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), -1), 0)
        causal_mask = torch.cat([torch.zeros(tgt_size, mem_size, dtype=dtype, device=device), causal_mask], dim=-1)
        causal_mask = causal_mask[None, None, ...].expand(batch_size, 1, tgt_size, src_size)
        # 1 for non-padding tokens
        expand_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_size, src_size)
        invert_mask = 1.0 - expand_mask
        ### add
        # invert_mask = ~ expand_mask
        invert_mask.masked_fill_(invert_mask.bool(), min_value)
        attention_mask = causal_mask.masked_fill(invert_mask.bool(), min_value)
        # print( 'mask dtype: ',attention_mask.dtype)

        # if position_ids is None:
        # position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=attention_mask.device).repeat(attention_mask.shape[0], 1)

        # print('prepare inputs: ',attention_mask)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "batch_X_data": batch_X_data,
            }
        )
        return model_inputs


    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    @property
    def device(self):
        return list(self.parameters())[0].device
    

AutoConfig.register("moviellm", MovieConfig)
AutoModelForCausalLM.register(MovieConfig, MovieForCausalLM)



