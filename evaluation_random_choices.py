import argparse
import functools
import gc
import json
import os
import torch.nn as nn
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import random
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.model_utils import projection_module
from peft import PeftModel

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding,generate_random_string, remove_punctuation, to_simple,contains_valid_letters
from utils.process_str import filter_ascii_text,model_generate,convert_lower_text,list_operation
from utils.reader import CustomDataset,read_jsonlines
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default=None,            help="测试集的路径")
add_arg("select_data",   type=str, default=None,            help="测试集的路径")
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
add_arg("noise",  type=bool,  default=False, help="输入模型的是噪声")
add_arg("filter_dataset",      type=bool,  default=False, help="是否过滤数据集")
add_arg("random_choice",  type=bool,  default=False, help="随机选择标签中的文本,选用这个，等于模型无效，noise无效")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("random_initialize_whisper", type=bool, default=False,    help="随机初始化whisper")
add_arg("teacher_forcing", type=bool, default=False,    help="使用teacher forcing")
add_arg("extra_name", type=str, default=None,    help="result basename里面增加字符")
add_arg("post_processing", type=bool, default=False,    help="是否使用后处理")
add_arg("config_name", type=str, default='base',    help="使用的模型")

# add_arg("metric",     type=str, default="fulleval",        choices=['cer', 'wer','fulleval'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"
# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
print('loading')
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

print('loading done')
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=args.language,
    task=args.task,
    no_timestamps=not args.timestamps,)

# 获取测试数据
test_jsonlines=read_jsonlines(args.test_data)
select_jsonlines=read_jsonlines(args.select_data)
test_sentences=[d["sentence"] for d in test_jsonlines]
select_sentences=[d["sentence"] for d in select_jsonlines]
# 获取评估方法
metrics = []
# metric_files = ['bert_score','bleu','mer', 'my_rouge','perplexity', 'wer','word_info_lost','word_info_preserved']
metric_files = ['bleu','mer', 'my_rouge','wer','word_info_lost','word_info_preserved','bert_score','meteor']
# Load metrics
for metric_file in metric_files:
    metric = evaluate.load(f'metrics/{metric_file}.py',
                           experiment_id=generate_random_string(100))
    metrics.append(metric)

if args.random_choice:
    all_labels=[]
result_basename=(f'1formal_test_results{"_"+args.extra_name if args.extra_name is not None else ""}'
                 f'{"no_post_processing" if not args.post_processing else "post_processing"}'
                 f'{"_noise"if args.noise else ""}'
                 f'{"_randomChoice"}'
                 f'{"_tf" if args.teacher_forcing else ""}')
# 开始评估
batch_size=args.batch_size
output_file=os.path.join(args.lora_model,f'{result_basename}.txt')
best_bleu1=10
best_results=None
best_preds_labels={
    "preds":None,
    "labels":None,
}
for repeat_time in range(1):
    decoded_labels=test_sentences
    decoded_preds=random.choices(select_sentences,k=len(decoded_labels))
    if args.post_processing:
        decoded_preds=filter_ascii_text(decoded_preds)
        decoded_labels=filter_ascii_text(decoded_labels)
        decoded_preds=convert_lower_text(decoded_preds)
        decoded_labels=convert_lower_text(decoded_labels)

    if not args.random_choice:
        for metric in metrics:
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    # 计算评估结果

    results={}
    for metric in metrics:
        result = metric.compute()
        for key in result.keys():
            if type(result[key])==torch.Tensor:
                result[key]=result[key].item()
            results[key]=result[key]
    print(f"评估结果：{results}")

    bleu1=results['bleu-1']
    if bleu1<best_bleu1:
        best_preds_labels["preds"]=decoded_preds
        best_preds_labels["labels"]=decoded_labels
        best_results=results
json_file_path=os.path.join(args.lora_model,f'{result_basename}.json')
with open(json_file_path,'w') as f:
    json.dump(best_results,f)

with open(output_file, "w") as f:
    for pred, label in zip(best_preds_labels["preds"], best_preds_labels["labels"]):
        f.write(f"start********************************\n")
        f.write(f"Predicted: {pred}\n")
        f.write(f"True: {label}\n")
        f.write(f"end==================================\n\n")
