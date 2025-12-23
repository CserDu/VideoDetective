import os
import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    no_use_fast: bool = field(
        default=False,
        metadata={'help': 'Do not use fast tokenizer?'}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: Optional[str] = field(
        # default="flash_attention_2",
        default=None,
        metadata={'help': 'The implementation of attention.'}
    )

    # max_length: int = field(
    #     default=4096,
    #     metadata={'help': 'How many tokens at maximum for each input.'},
    # )
    chat_template: str = field(
        default="hf",
        metadata={'help': 'Instruction template name in fastchat.'}
    )

    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum position.'},
    )
    mistral_sliding_window: Optional[int] = field(
        default=None,
        metadata={'help': 'Sliding window size in Mistral models.'},
    )
    rope_theta: Optional[float] = field(
        default=None,
        metadata={'help': 'RoPE base (theta).'},
    )
    rope_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to scale RoPE? {linear, dynamic, yarn}'},
    )
    rope_factor: float = field(
        default=1.,
        metadata={'help': 'RoPE scaling factor.'},
    )

    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )
    load_in_4_bit: bool = field(
        default=False,
        metadata={'help': 'Load model in 4 bits?'},
    )

    dtype: str = field(
        default="fp16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )
    
    enable_vllm: bool = field(
        default=False,
        metadata={'help': 'Use vllm?'}
    )
    vllm_mem: float = field(
        default=0.9,
        metadata={'help': 'Vllm maximum GPU memory utilization.'}
    )
    vllm_tp: int = field(
        default=1,
        metadata={'help': 'Vllm tensor parallel degree.'}
    )
    vllm_len: Optional[int] = field(
        default=None,
        metadata={'help': 'Vllm maximum sequence length.'}
    )
    vllm_disable_ar: bool = field(
        default=False,
        metadata={'help': 'Disable custom all-reduce in vllm?'}
    )

    enable_beacon: bool = field(
        default=False,
        metadata={'help': 'Enable activation beacon?'}
    )
    beacon_window: Optional[int] = field(
        default=None,
        metadata={'help': 'The initial sliding window size.'}
    )
    beacon_stride: Optional[int] = field(
        default=None,
        metadata={'help': 'The stride of the sliding window.'}
    )
    beacon_attn: Optional[str] = field(
        default=None,
        metadata={'help': 'How to assign attention masks of beacon tokens? {segmentation, step-expansion, full-converage}'}
    )
    beacon_ratio: Optional[List[int]] = field(
        default=None,
        metadata={'help': 'Condensing ratios for beacons.'}
    )
    beacon_ratio_mix: Optional[str] = field(
        default=None,
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    beacon_param: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'The introduced parameters for beacon.'}
    )
    beacon_embed_init: str = field(
        default="eos",
        metadata={'help': 'Initialize beacon embedding from eos/bos embedding.'}
    )
    beacon_sink_size: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of activations that are always kept in the head of the sequence according to StreamingLLM.'}
    )
    beacon_attend_prev: Optional[bool] = field(
        default=None,
        metadata={'help': 'Can beacon tokens attend to previous beacon tokens?'}
    )
    beacon_pos: Optional[str] = field(
        default=None,
        metadata={'help': 'Where to put beacon tokens? {append, interleave}'}
    )
    beacon_parallel_window: Optional[int] = field(
        default=None,
        metadata={'help': 'How many windows to run in parallel?'}
    )
    retrieval_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to retrieve? {bm25}'}
    )
    retrieval_topk: Optional[int] = field(
        default=None,
        metadata={'help': 'How many windows to retrieve?'}
    )
    retrieval_key_length: Optional[int] = field(
        default=None,
        metadata={'help': 'The key sequence length in retrieval.'}
    )

    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    do_sample: Optional[bool] = field(
        default=None,
        metadata={'help': 'Do sampling when decoding?'},
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )

    llm_model: str = field(default='qwen')

    freeze_backbone: bool = field(default=True)

    ## role aware model args
    speech_dim: int = field(default=1280)
    visual_dim: int = field(default=1024)
    bind_model_num_layers: int = field(default=2)
    bind_model_cross_attention_freq: int = field(default=1)
    bind_model_hidden_size: int = field(default=768)
    bind_model_duration: int = field(default=5)
    role_model_hidden_size: int = field(default=768)
    role_model_num_layers: int = field(default=2)
    role_model_cross_attention_freq: int = field(default=1)
    role_model_num_query_token: int = field(default=128)
    role_model_duration: int = field(default=5)

    audio_window_duration: int = field(default=8)
    visual_window_duration: int = field(default=8)

    audio_branch: bool = field(default=True)
    visual_branch: bool = field(default=True)

    
    def get_generation_config(self):
        generation_config = {}
        if self.max_new_tokens is not None:
            generation_config["max_new_tokens"] = self.max_new_tokens
        if self.do_sample is not None:
            generation_config["do_sample"] = self.do_sample
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["top_p"] = self.top_p
        return generation_config

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)


@dataclass
class TrainingArgs(TrainingArguments):
    # ==============================
    # Common arguments
    # ==============================
    output_dir: str = field(default="results")

    per_device_train_batch_size: int = field(
        default=1,
        metadata={'help': 'Train batch size.'}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Remove columns in the dataset that are not registered in the forward function?'}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={'help': 'Find unusuable parameters?'}
    )
    # NOTE: essential to keep comuputation graph because we need gradients for beacon tokens
    use_reentrant: Optional[bool] = field(
        default=None,
        metadata={'help': "Use reetrant in gradient checkpointing?"}
    )
    report_to: str = field(
        default="none",
        metadata={'help': 'Log results by external tools?'}
    )

    # ==============================
    # Customized arguments
    # ==============================
    min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for training?'}
    )
    group_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Group the training data instances by the number of strides in the beacon model. {relaxed, strict}'}
    )
    sort_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Sort the training data instances by the number of strides in the beacon model. {ascend, descend}'}
    )
    only_train_beacon: bool = field(
        default=True,
        metadata={'help': 'Freeze LLM parameters when training beacon parameters?'}
    )
    
    eval_method: str = field(
        default="perplexity",
        metadata={'help': 'How to evaluate during training? {perplexity, generation}'}
    )
    eval_max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input in evaluation.'},
    )
    eval_min_length: int = field(
        default=512,
        metadata={'help': 'How many tokens at minimum for each input in evaluation.'},
    )
    eval_beacon_ratio: List[int] = field(
        default_factory=lambda: [32],
        metadata={'help': 'Condensing ratios for beacons in evaluation.'}
    )
    eval_beacon_ratio_mix: str = field(
        default="adapt-1024",
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    max_eval_num: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for validation?'}
    )

    lora_enable: bool = field(default=False)
    lora_tune: bool = field(
        default=False,
        metadata={"help": "Use LoRA fine-tuning?"},
    )
    lora_rank: int = field(
        default=32,
        metadata={'help': 'LoRA rank.'}
    )
    lora_alpha: int = field(
        default=16,
        metadata={'help': 'LoRA scaling factor.'}
    )
    lora_dropout: float = field(
        default=0.,
        metadata={'help': 'LoRA dropout p.'}
    )
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Module name patterns to add LoRA."},
    )
    lora_extra_params: List[str] = field(
        default_factory=lambda: ["embed_tokens", "norm"],
        metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    )

    metrics: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'List of metrics. {rouge, save_result}'}
    )
    log_path: str = field(
        default="results/metrics.log",
        metadata={'help': 'Log file path.'}
    )

    lora_r: int = field(
        default=32,
        metadata={'help': 'LoRA rank.'}
    )
    lora_bias: str = field(default='none')
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

    # train_visual_branch: bool = field(default=True)
    # train_audio_branch: bool = field(default=True)
    pretrain_audio: bool = field(default=False)
    pretrain_visual: bool = field(default=False)

    save_modules: str = field(default='')

    exp_desc: str = field(default='exp')
    training_stage: str = field(default='stage1') # stage1 stage2 stage3

    pretrain_ckpt_path: str = field(default='')

    audio_pretrain_ckpt_path: str = field(default='')
    visual_pretrain_ckpt_path: str = field(default='')

    use_memory: bool = field(default=False)
    use_caption: bool = field(default=False)

    question_after_shot: bool = field(default=False)


@dataclass
class DataArgs:
    # cid_dir: str = field(default='/cfs/cfs-bve8l01f/data/ip_data/cid')
    vids_path: str = field(default='data/vids.txt')
    video_dir: str = field(default='')
    audio_dir: str = field(default='')
    caption_dir : str = field(default='')
    # video
    image_size: int = field(default=224)
    n_frms: int = field(default=60)
    # audio
    sample_rate: int = field(default=16000)
    clip_duration: int = field(default=2)
    clips_per_video: int = field(default=60)

    image_caption_task: bool = field(default=True)
    video_caption_task: bool = field(default=True)
    audio_caption_task: bool = field(default=True)

    ## role aware dataset
    max_duration: int = field(default=3*60)
    max_role_nums: int = field(default=15)


@dataclass
class InferenceArgs:
    audio_qformer_ckpt_path: str = field(default='')
    ckpt_dir: str = field(default='')
    device: str = field(default='cuda:0')
    finetune_ckpt_dir: str = field(default='')


