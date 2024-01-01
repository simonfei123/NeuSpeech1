import argparse
import functools
import gc
import json
import os

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.load_model import MyWhisperForConditionalGeneration
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="/home/yyang/dataset/multi_media/formal_dataset/cut/test_data.jsonl",            help="测试集的路径")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("lora_model",    type=str, default=None,      help="训练过的lora模型")
add_arg("modal", type=str, default='speech',  help="输入的模态")
add_arg("sampling_rate", type=int, default=1000,  help="输入信号采样率")
add_arg("eeg_ch", type=int, default=66,  help="输入信号通道数")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=True,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("metric",     type=str, default="fulleval",        choices=['cer', 'wer','fulleval'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"
# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids()
# 获取模型
if args.modal=='eeg':
    model = MyWhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only,
                                                        modal_ch=args.eeg_ch)
    if args.lora_model is not None:
        model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
        model = model.merge_and_unload()
        print(model)
elif args.modal=='speech':
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only,
                                                        )
else:
    raise NotImplementedError
# 因为保存的模型有问题，我们直接手动加载权重
# from collections import OrderedDict
# checkpoint = torch.load(os.path.join(args.model_path,'pytorch_model.bin'))
# new_cp=OrderedDict()
# msd=model.state_dict()
# for key in checkpoint.keys():
#     real_key=key[27:]
#     if real_key in msd.keys():
#         new_cp[real_key]=checkpoint[key]
#         print(real_key)
#     else:
#         # print(real_key)
#         pass
# model.load_state_dict(new_cp)
# print('models weights are replaced')
model.eval()

# 获取测试数据
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             timestamps=args.timestamps,
                             modal=args.modal,
                             mode='test',
                             modal_ch=args.eeg_ch,
                             sample_rate=args.sampling_rate,
                             language=args.language,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"测试数据：{len(test_dataset)}")

# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# 获取评估方法
metric = evaluate.load(f'metrics/{args.metric}.py')

# 开始评估
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if args.modal=='eeg':
                input_features = F.gelu(model.pre_conv1(batch["input_features"].cuda()).cuda())
                input_features = F.gelu(model.pre_conv2(input_features))
            elif args.modal=='speech':
                input_features = batch["input_features"].cuda()

            generated_tokens = (
                model.generate(
                    # encoder_outputs=model.model.encoder(model.pre_conv(batch["input_features"].cuda())),
                    input_features=input_features,
                    # condition_on_previous_text=0,
                    decoder_input_ids=batch["labels"][:, :4].cuda(),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255).cpu().numpy())
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # 将预测和实际的token转换为文本
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # 删除标点符号
            if args.remove_pun:
                decoded_preds = remove_punctuation(decoded_preds)
                decoded_labels = remove_punctuation(decoded_labels)
            # 将繁体中文总成简体中文
            if args.to_simple:
                decoded_preds = to_simple(decoded_preds)
                decoded_labels = to_simple(decoded_labels)
            print('decoded_preds')
            print(decoded_preds)
            print('decoded_labels')
            print(decoded_labels)
            print('end')
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    # 删除计算的记录
    del generated_tokens, labels, batch
    gc.collect()
# 计算评估结果
m = metric.compute()
print(f"评估结果：{m}")
json_file_path=os.path.join(args.lora_model,'eval.json')
with open(json_file_path,'w') as f:
    json.dump(m,f)
