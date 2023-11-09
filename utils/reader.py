import json
import os
import sys
from typing import List
from utils.augment_eeg import RandomShapeMasker,shift_data
import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.binary import DatasetReader
from utils.utils import preprocess_eeg_data, lowpass_filter, add_gaussian_noise
import jsonlines
from utils.process_utils import torch_random_choices
def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts
def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)

class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 data_list_dir='/home/yyang/dataset/multi_media/',
                 modal='eeg',
                 modal_ch=66,
                 language=None,
                 timestamps=False,
                 sample_rate=1000,
                 min_duration=0.5,
                 max_duration=30,
                 augment_config_path=None):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            modal: eeg,speech
            language: 微调数据的语言
            timestamps: 微调时是否使用时间戳
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            augment_config_path: 数据增强配置参数文件路径
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        self.data_list_path = data_list_path
        self.mode=os.path.basename(data_list_path)[:-6]
        self.processor = processor
        self.signal_sample_rate = sample_rate
        self.language = language
        self.timestamps = timestamps
        self.data_list_dir = data_list_dir
        self.modal = modal
        self.modal_ch = modal_ch
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        self.nocaptions = self.vocab['<|nocaptions|>']
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()
        # 数据增强配置参数
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)

    # 加载数据列表
    def _load_data_list(self):
        # if self.mode.startswith('train'):
        #     # num=1000
        #     self.data_list = read_jsonlines(self.data_list_path)#[:num]
        # elif self.mode.startswith('val'):
        #     # num=10
        #     self.data_list = read_jsonlines(self.data_list_path)[500:600]
        # else:
        #     num=None
        #     self.data_list = read_jsonlines(self.data_list_path)[:num]
        self.data_list = read_jsonlines(self.data_list_path)
        print(f'num of data:{len(self.data_list)} mode:{self.mode}')

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file = os.path.join(self.data_list_dir,data_list[self.modal]['path'])
        assert audio_file is not None
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if self.modal=='eeg':
            sample=np.load(audio_file)

            assert sample.shape[0]<sample.shape[1],f'eeg shape should be [ch,len],now shape is {sample.shape},data idx{idx}'
            sample=sample[:self.modal_ch]
            sample=sample.T # convert to shape[len,ch]
            sample_rate=self.signal_sample_rate
        elif self.modal=='speech':
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            raise NotImplementedError

        sample = sample.T  # eeg:[ch, len]
        if self.modal=='eeg':
            # 先滤波
            sample = lowpass_filter(sample, cutoff_freq=50, sample_freq=self.signal_sample_rate)
            sample = self.resample(sample,self.signal_sample_rate,200)
            self.signal_sample_rate=200
            sample,clipped_ratio=preprocess_eeg_data(sample)
            assert clipped_ratio<0.2
        # 数据增强
        if self.augment_configs:
            # 只有训练的时候才会增强数据
            if self.mode.startswith('train'):
                # pass
                sample, sample_rate = self.augment_audio(sample, sample_rate)
        # 重采样

        # if self.modal=='eeg':
        #     if self.signal_sample_rate != 1000:
        #         sample = self.resample(sample, orig_sr=1000, target_sr=self.signal_sample_rate)
        # elif self.modal=='speech':
        #     if self.signal_sample_rate != sample_rate:
        #         sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.signal_sample_rate)
        # else:
        #     raise NotImplementedError
        # sample should be of shape [ch,len]
        return sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def shift_data_transcript(self,sample,transcript):
        assert self.modal=='eeg'
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        length=sample.shape[1]
        max_shift=int(self.max_duration*self.signal_sample_rate)-length
        now_shift=np.random.randint(max_shift,size=None)
        sample=shift_data(sample,now_shift)
        now_shift_time=now_shift/self.signal_sample_rate
        for t in transcript:
            # 将目标文本编码为标签ID
            t['start']=t['start']+now_shift_time
            t['end']=t['end']+now_shift_time
        return sample,transcript


    def __getitem__(self, idx):
        try:
            # 从数据列表里面获取音频数据、采样率和文本
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            # 将sample进行时间漂移，并将transcript的时间对齐。
            if self.mode=='train':
                sample,transcript=self.shift_data_transcript(sample,transcript)
            # 可以为单独数据设置语言
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # 加载带有时间戳的文本
                if self.timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    if self.modal=='speech':
                        data["input_features"] = self.processor(audio=sample,
                                                                sampling_rate=self.signal_sample_rate).input_features
                        # print(f'input_features:{data["input_features"][0].shape}')
                    else:
                        data["input_features"]=self.padding_sample(sample)
                else:
                    if self.modal=='speech':
                        data = self.processor(audio=sample, sampling_rate=self.signal_sample_rate, text=transcript)
                    else:
                        data={
                            'input_features':self.padding_sample(sample),
                            'labels':self.process_transcript(transcript),
                        }

            else:
                # 如果没有文本，则使用<|nocaptions|>标记
                print('没有文本')
                if self.modal=='speech':
                    data = self.processor(audio=sample, sampling_rate=self.signal_sample_rate)
                else:
                    data= {'input_features': self.padding_sample(sample),
                           'labels': [self.startoftranscript, self.nocaptions, self.endoftext]}
            # print(f'mode:{self.mode}   data:{idx}')
            return data

        except Exception as e:
            print(f'读取数据出错，序号：{idx}，错误信息：{e}', file=sys.stderr)
            return self.__getitem__(torch.randint(0, self.__len__(),(1,)).tolist()[0])

    def padding_sample(self,sample):
        assert self.modal == 'eeg'
        max_length=int(self.max_duration*self.signal_sample_rate)
        sample=sample[:,:max_length]
        # print(f'before pad eeg:{sample.shape}')
        sample=np.pad(sample,pad_width=((0,0),(0,max_length-sample.shape[-1])))
        # print(f'sample.shape:{sample.shape}')
        assert sample.shape==(self.modal_ch,int(self.signal_sample_rate*30)) ,' sample shape should be [self.modal_ch,int(self.signal_sample_rate*30))]'
        # print(f'after pad eeg:{sample.shape}')
        return [sample]

    def process_transcript(self,transcript):
        data = self.processor(text=transcript)
        return data['input_ids']

    def __len__(self):
        return len(self.data_list)

    # 分割读取音频
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # 数据增强

    def augment_audio(self, sample, sample_rate):
        for config in self.augment_configs:
            # if config['type'] == 'speed' and torch.rand(1).tolist()[0] < config['prob']:
            #     if self.modal=='speech':
            #         if self.speed_rates is None:
            #             min_speed_rate, max_speed_rate, num_rates = config['params']['min_speed_rate'], \
            #                 config['params']['max_speed_rate'], config['params']['num_rates']
            #             self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
            #         # rate = random.choice(self.speed_rates)
            #         rate=torch_random_choices(self.speed_rates,1)[0]
            #         sample = self.change_speed(sample, speed_rate=rate)
            #     elif self.modal=='eeg':
            #         pass
            #     else:
            #         raise NotImplementedError
            # if config['type'] == 'shift' and torch.rand(1).tolist()[0]< config['prob']:
            #     min_shift_ms, max_shift_ms = config['params']['min_shift_ms'], config['params']['max_shift_ms']
            #     shift_ms = torch.randint(min_shift_ms, max_shift_ms,(1,)).tolist()[0]
            #     if self.modal == 'speech':
            #         sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            #     elif self.modal == 'eeg':
            #         sample = self.shift(sample.T, sample_rate, shift_ms=shift_ms)
            #         sample = sample.T
            #     else:
            #         raise NotImplementedError
            # if config['type'] == 'volume' and torch.rand(1).tolist()[0] < config['prob']:
            #     min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
            #     gain = torch.randint(min_gain_dBFS, max_gain_dBFS,(1,)).tolist()[0]
            #     sample = self.volume(sample, gain=gain)
            # if config['type'] == 'resample' and torch.rand(1).tolist()[0] < config['prob']:
            #     if self.modal == 'speech':
            #         new_sample_rates = config['params']['new_sample_rates']
            #         # new_sample_rate = np.random.choice(new_sample_rates)
            #         new_sample_rate=torch_random_choices(new_sample_rates,1)[0]
            #         sample = self.resample(sample, orig_sr=sample_rate, target_sr=new_sample_rate)
            #         sample_rate = new_sample_rate
            #     elif self.modal == 'eeg':
            #         # eeg 不做重采样
            #         pass
            #     else:
            #         raise NotImplementedError

            # if config['type'] == 'filter' and torch.rand(1).tolist()[0] < config['prob']:
            #     if self.modal == 'eeg':
            #         cutoff_freq=np.random.uniform(config['params']['min_high_freq'],config['params']['max_high_freq'], size=None)
            #         sample=lowpass_filter(sample,cutoff_freq=cutoff_freq,sample_freq=self.signal_sample_rate)
            if config['type'] == 'noise' and torch.rand(1).tolist()[0] < config['prob']:
                if self.modal == 'eeg':
                    sample=add_gaussian_noise(sample,snr_range=(config['params']['min_snr_dB'],config['params']['max_snr_dB']))
            if config['type'] == 'mask' and torch.rand(1).tolist()[0] < config['prob']:
                if self.modal == 'speech':
                    pass
                elif self.modal == 'eeg':
                    # eeg 目前是做椒盐噪声，即随机掩码
                    # print('eeg mask')
                    time_mask_len=torch.randint(20, 40, (1,)).tolist()[0]
                    ch_mask_len=torch.randint(1, 4, (1,)).tolist()[0]
                    prob=torch.rand(1).tolist()[0]*0.3
                    augmentor=RandomShapeMasker(unit=(ch_mask_len,time_mask_len),mask_prob=prob,length_unit=20,
                                                length_prob=(0.1,0.2),channel_num=(1,10),random_types=(1,2,3))
                    mask=augmentor(sample.shape)
                    del augmentor
                    mask=np.array(mask)
                    sample=sample*mask
                else:
                    raise NotImplementedError
        return sample, sample_rate

    # 改变语速
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # 音频偏移
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # 改变音量
    @staticmethod
    def volume(sample, gain):
        sample = sample*10.**(gain / 20.)
        return sample

    # 声音重采样
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # 添加噪声
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # 标准化音频音量，保证噪声不会太大
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample = sample * 10. ** (gain / 20.)
        # 指定噪声音量
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample = noise_sample * 10. ** (noise_gain_db / 20.)
        # 固定噪声长度
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = torch.randint(0, noise_sample.shape[0] - sample.shape[0],(1,)).tolist()[0]
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
        