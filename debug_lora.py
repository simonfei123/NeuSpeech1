import argparse
import functools
import os
import platform
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# import torch._dynamo as dynamo
import logging
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint,trainer_save_model,compute_accuracy,projection_module
from utils.load_model import WhisperForConditionalGeneration,match_modules,match_modules_string
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/train_data.jsonl",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/val_data.jsonl",        help="测试数据集的路径")
add_arg("base_model",    type=str, default="/home/yyang/dataset/multi_media/transformers_whisper_models/large_finetune",      help="Whisper的基础模型")
add_arg("lora_model",    type=str, default=None,      help="训练过的lora模型")
add_arg("output_dir",    type=str, default="output1/",                  help="训练保存模型的路径")
add_arg("warmup_steps",  type=int, default=10000,      help="训练预热步数")
add_arg("logging_steps", type=int, default=100,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=1000,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=1000,    help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=6,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=1e-3,  help="学习率大小")
add_arg("modal", type=str, default='speech',  help="输入的模态")
add_arg("sampling_rate", type=int, default=200,  help="输入信号期望采样率")
add_arg("orig_sample_rate", type=int, default=200,  help="输入信号采样率")
add_arg("eeg_ch", type=int, default=224,  help="输入信号通道数")
add_arg("lora_eeg_ch", type=int, default=None,  help="lora模型的输入信号通道数")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒")
add_arg("use_adalora",   type=bool,  default=True,  help="是否使用AdaLora而不是Lora")
add_arg("fp16",          type=bool,  default=False,  help="是否使用fp16训练模型")
add_arg("use_8bit",      type=bool,  default=False, help="是否将模型量化为8位")
add_arg("filter_dataset",      type=bool,  default=False, help="是否过滤数据集")
add_arg("timestamps",    type=bool,  default=True, help="训练时是否使用时间戳数据")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=30,      help="训练的轮数")
add_arg("language",      type=str, default="English", help="设置语言，可全称也可简写，如果为None则训练的是多语言")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("augment_config_path",         type=str, default='configs/augmentation.json', help="数据增强配置文件路径")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=2,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=2,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("fine_tune_layers", type=int, default=None,    help="微调base model的前多少层")
add_arg("device", type=str, default='auto',    help="device")
add_arg("config_name", type=str, default='base',    help="conv1 module")
add_arg("random_initialize_whisper", type=bool, default=False,    help="随机初始化whisper")
args = parser.parse_args()
print_arguments(args)


# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)


# 获取Whisper模型
device_map = args.device
if device_map == 'cpu':
    ddp=0
else:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# print(f'device_map:{device_map}, os env:{os.environ["CUDA_VISIBLE_DEVICES"]}')
# device_map = 'cpu'
# 获取模型
print(f'device map :{device_map}')
model=WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                    load_in_8bit=args.use_8bit,
                                                    device_map=device_map,
                                                    local_files_only=args.local_files_only,
                                                        )
print(f'model device {model.device}')
eeg_ch=args.eeg_ch
if args.lora_eeg_ch is not None:
    eeg_ch=args.lora_eeg_ch

device=model.device
kwargs={
    'meg_ch':eeg_ch,
    'd_model':model.model.encoder.conv2.in_channels,
}

conv1=projection_module(config_name=args.config_name,**kwargs)


# conv1 = nn.Conv1d(meg_ch, d_model, kernel_size=3, padding=1)
conv1 = conv1.to(device)
model.model.encoder.set_input_embeddings(conv1)
# model=model.to(device)
if args.lora_model is not None:
    # 之前的加载模型是把模型变成要加载的模型的形状，然后再加载参数。
    # 现在是变成要训练的模型。
    model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
    model = model.merge_and_unload()
    if args.lora_eeg_ch!=args.eeg_ch:
        kwargs={
            'meg_ch':args.eeg_ch,
            'd_model':model.model.encoder.conv2.in_channels,
        }

        conv1=projection_module(config_name=args.config_name,**kwargs)
        conv1 = conv1.to(device)
        model.model.encoder.set_input_embeddings(conv1)
if args.random_initialize_whisper:
    model.post_init() #todo 这个没用，还不会弄
    print('模型已被初始化')
# model.save_pretrained(save_directory=os.path.join(args.output_dir, "checkpoint-init"))
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# 量化模型
model = prepare_model_for_kbit_training(model)
# 注册forward，否则多卡训练会失败
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
# model.model.encoder.conv1[0].register_forward_hook(make_inputs_require_grad)

for param in model.parameters():
    param.requires_grad=False
# for param in model.model.model.decoder.parameters():
#     param.requires_grad=False

# print('加载LoRA模块...')
# if args.resume_from_checkpoint:
#     # 恢复训练时加载Lora参数
#     print("Loading adapters from checkpoint.")
#     model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
# else:
#     print(f'adding LoRA modules...')
#     if args.fine_tune_layers is not None:
#         prefixes = [f'model.encoder.layers.{i}.' for i in range(args.fine_tune_layers)]
#         if args.fine_tune_layers ==0:
#             prefixes=['model.encoder.conv1']
#     else:
#         prefixes = ['model.encoder']
#     suffixes = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
#     # model_named_modules=[]
#     # target_modules = []
#     target_modules = match_modules_string(model.named_modules(), prefixes, suffixes)
#     print(target_modules)
#     if args.fine_tune_layers ==0:
#         # target_modules='model.encoder.conv2'
#         target_modules=['conv1']
#     print('target_modules')
#     print(target_modules)
#     # modules_to_save= match_modules(model.named_modules(),[''],[''],[".*model.encoder.conv1",".*model.encoder.conv2"])
#     modules_to_save= ['model.encoder.conv1', 'model.encoder.conv2']
#     print('modules_to_save')
#     print(modules_to_save)
#     if args.use_adalora:
#         config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
#                                lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules,
#                                modules_to_save=modules_to_save)
#     else:
#         config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none",
#                             modules_to_save=modules_to_save)
#     print(config)
#     config.target_modules=False
#     model = get_peft_model(model, config)

if args.fine_tune_layers ==0:
    # 不用lora
    for name,param in model.named_parameters():
        if 'conv' not in name:
            param.requires_grad=False
        else:
            param.requires_grad=True

print(model)
# model.to('cpu')
print('trainable parameters')
print('=' * 90)

# for name,param in model.named_parameters():
#     if 'layers' in name:
#         model
for name,param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('=' * 90)