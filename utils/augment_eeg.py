import torch
import numpy as np


def random_prob(low_prob=0.2, high_prob=0.8):
    # 生成0到1之间的随机数
    rand_num = torch.rand(1).item()

    # 对随机数进行线性变换
    prob = low_prob + rand_num * (high_prob - low_prob)

    return prob


def random_discrete_only_mask(signal_shape, unit=(1, 40), prob=0.5):
    # signal: [64, 2000] tensor
    # unit:[1,40]
    length = int(np.ceil(signal_shape[1] / unit[1]))
    channel_num = int(np.ceil(signal_shape[0] / unit[0]))
    pre_mask = torch.rand(channel_num, length)
    pre_mask[pre_mask >= prob] = 1
    pre_mask[pre_mask < prob] = 0
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal_shape[0] / channel_num)), dim=0)
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal_shape[1] / length)), dim=1)[:signal_shape[0],
               :signal_shape[1]]
    return pre_mask


def random_discrete_mask(signal, unit=(1, 40), prob=0.5):
    # signal: [64, 2000] tensor
    # unit:[1,40]
    length = int(np.ceil(signal.shape[1] / unit[1]))
    channel_num = int(np.ceil(signal.shape[0] / unit[0]))
    pre_mask = torch.rand(channel_num, length)
    pre_mask[pre_mask >= prob] = 1
    pre_mask[pre_mask < prob] = 0
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal.shape[0] / channel_num)), dim=0)
    pre_mask = torch.repeat_interleave(pre_mask, int(np.ceil(signal.shape[1] / length)), dim=1)[:signal.shape[0],
               :signal.shape[1]]
    signal = signal * pre_mask
    return signal, pre_mask


def random_channel_mask(signal, low=1, high=32):
    # 随机选择掩码通道数量
    mask_size = torch.randint(low, high + 1, (1,)).item()

    # 随机选择要掩码的通道
    channels = torch.randperm(signal.shape[0])[:mask_size]
    mask = torch.ones_like(signal)
    # 对选择的通道进行掩码
    mask[channels, :] = 0

    return signal * mask, mask


def random_length_mask(signal, unit_length=40, low_prob=0.2, high_prob=0.8):
    prob = random_prob(low_prob, high_prob)
    length = int(np.ceil(signal.shape[1] / unit_length))
    pre_mask = torch.rand(1, length)
    pre_mask[pre_mask >= prob] = 1
    pre_mask[pre_mask < prob] = 0
    pre_mask = pre_mask.repeat_interleave(signal.shape[0], axis=0)
    pre_mask = pre_mask.repeat_interleave(unit_length, axis=1)
    pre_mask = pre_mask[:, :signal.shape[1]]
    return signal * pre_mask, pre_mask


class RandomShapeMasker:
    def __init__(self, unit=(1, 40), mask_prob=0.7, channel_num=(1, 32), length_prob=(0.2, 0.4), random_types=(1,)):
        self.unit = unit
        self.mask_prob = mask_prob
        self.channel_num = channel_num
        self.length_prob = length_prob
        self.random_types = random_types

    def __call__(self, signal_shape):
        random_type = np.random.choice(self.random_types, 1)[0]
        if random_type == 1:
            return random_discrete_only_mask(signal_shape, unit=self.unit, prob=self.mask_prob)
