# import soundfile as sf
import numpy as np
import torch
from torch import nn
import librosa
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi

from transformers import WhisperFeatureExtractor


class AudioProcessor:

    def __init__(
        self,
        sr = 16000,
        whisper_path='',
    ) -> None:
        
        self.sr = sr
        self.whisper_processor = WhisperFeatureExtractor.from_pretrained(whisper_path,local_files_only=True)


    def preprocess_for_beats(self,source: torch.Tensor,fbank_mean: float = 15.41663,fbank_std: float = 6.55582,) -> torch.Tensor:
        # source: bs,L
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0) # bs, len, 128
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank


    # def __call__(self, audio_path, shot_duration = None):
    #     if shot_duration is not None:
    #         return self.process_with_shot_duration(audio_path, shot_duration)
        
    #     return self.process_without_shot_duration(audio_path)

    def __call__(self,audio_list:list):
        shot_duration = 30
        sr = 16000
        audio_token_nums = []
        spec_list = []
        fbank_list = []
        for audio in audio_list:
            seg_nums = int(len(audio) // (shot_duration * sr))
            if len(audio) % (shot_duration * sr) != 0:
                seg_nums += 1
            spec = []
            fbank = []
            for i in range(seg_nums):
                st = i * sr * shot_duration
                et = (i + 1) * sr * shot_duration
                audio_seg = audio[st : et]
                if len(audio_seg) < sr:
                    pad_size = sr - len(audio_seg)
                    sil = np.zeros(pad_size, dtype=float)
                    audio_seg = np.concatenate((audio_seg,sil),axis=0)
                spec_seg = self.whisper_processor(audio_seg, sampling_rate=self.sr, return_tensors="pt")["input_features"]
                spec_seg = spec_seg.squeeze(0).to(torch.float32) # 80,len
                fbank_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank_seg = self.preprocess_for_beats(fbank_seg)
                fbank_seg = fbank_seg.squeeze(0).to(torch.float32) # len, 128
                spec.append(spec_seg)
                fbank.append(fbank_seg)
            
            spec_list.append(spec)
            fbank_list.append(fbank)
            audio_token_nums.append(88 * seg_nums)

        # return spec_list, fbank_list, audio_token_nums
        return {
            'spec_list':spec_list, # [ [seg1, seg2,..], [seg1, seg2, ...] ]
            'fbank_list':fbank_list,
            'audio_token_nums':audio_token_nums
        }


    def process_with_shot_duration(self, audio_path, shot_duration):
        try:
            audio, sr = librosa.load(audio_path, sr = 16000, mono = True)
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            # audio = audio[: 4 * 60 * sr] # max 4 min.
        except:
            print('====== audio processor error, audio path: ',audio_path)
            audio = np.zeros(int(shot_duration * sr), dtype=float)
        
        seg_nums = int(len(audio) // (shot_duration * sr))
        if len(audio) % (shot_duration * sr) != 0:
            seg_nums += 1
        
        spec = []
        fbank = []
        for i in range(seg_nums):
            st = i * sr * shot_duration
            et = (i + 1) * sr * shot_duration
            audio_seg = audio[st : et]

            if len(audio_seg) < sr:
                pad_size = sr - len(audio_seg)
                sil = np.zeros(pad_size, dtype=float)
                audio_seg = np.concatenate((audio_seg,sil),axis=0)

            spec_seg = self.whisper_processor(audio_seg, sampling_rate=self.sr, return_tensors="pt")["input_features"]
            spec_seg = spec_seg.squeeze(0).to(torch.float32) # 80,len

            fbank_seg = torch.from_numpy(audio_seg).unsqueeze(0) # 80, 1500
            fbank_seg = self.preprocess_for_beats(fbank_seg)
            fbank_seg = fbank_seg.squeeze(0).to(torch.float32) # len, 128

            spec.append(spec_seg)
            fbank.append(fbank_seg)
            
        return spec, fbank

    def process_without_shot_duration(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr = 16000, mono = True)
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            # audio = audio[: int(max_duration * sr)]
        except:
            print('====== audio processor error, audio path: ',audio_path)
            audio = np.zeros(int(15 * sr), dtype=float)
        
        seg_nums = 1
        
        spec = []
        fbank = []
        for i in range(seg_nums):
            audio_seg = audio
            if len(audio_seg) < sr:
                pad_size = sr - len(audio_seg)
                sil = np.zeros(pad_size, dtype=float)
                audio_seg = np.concatenate((audio_seg,sil),axis=0)

            spec_seg = self.whisper_processor(audio_seg, sampling_rate=self.sr, return_tensors="pt")["input_features"]
            spec_seg = spec_seg.squeeze(0).to(torch.float32) # 80,len

            fbank_seg = torch.from_numpy(audio_seg).unsqueeze(0) # 80, 1500
            fbank_seg = self.preprocess_for_beats(fbank_seg)
            fbank_seg = fbank_seg.squeeze(0).to(torch.float32) # len, 128

            spec.append(spec_seg)
            fbank.append(fbank_seg)
            
        return spec, fbank
    
