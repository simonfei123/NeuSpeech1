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
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding,generate_random_string, remove_punctuation, to_simple,contains_valid_letters
from utils.process_str import filter_ascii_text,model_generate
from utils.reader import CustomDataset,read_jsonlines,write_jsonlines
from utils.utils import print_arguments, add_arguments
from metrics.each_sentence_metrics import EachSentenceMetrics

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
add_arg("noise",  type=bool,  default=False, help="输入模型的是噪声")
add_arg("random_choice",  type=bool,  default=False, help="随机选择标签中的文本,选用这个，等于模型无效，noise无效")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
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

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                    device_map="auto",
                                                    local_files_only=args.local_files_only,)

device=model.device
meg_ch = args.eeg_ch
d_model = model.model.encoder.conv2.in_channels
conv1 = nn.Sequential(
    nn.Conv1d(meg_ch, d_model, kernel_size=3, padding=1),
    nn.GELU(),
    nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
)
conv1.stride = (2,)

# conv1 = nn.Conv1d(meg_ch, d_model, kernel_size=3, padding=1)
conv1 = conv1.to(device)
model.model.encoder.set_input_embeddings(conv1)
if args.lora_model is not None:
    model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
    model = model.merge_and_unload()
    print(model)
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
                             num_workers=args.num_workers, collate_fn=data_collator,
                             shuffle=False)

data_jsonl=read_jsonlines(args.test_data)
# 获取评估方法
# metrics = []
# # metric_files = ['bert_score','bleu','mer', 'my_rouge','perplexity', 'wer','word_info_lost','word_info_preserved']
# metric_files = ['bleu','mer', 'my_rouge','wer','word_info_lost','word_info_preserved']
# # Load metrics
# for metric_file in metric_files:
#     metric = evaluate.load(f'metrics/{metric_file}.py',
#                            experiment_id=generate_random_string(100))
#     metrics.append(metric)

sentence_metrics_calculator=EachSentenceMetrics(metrics_files=['bleu', 'mer', 'my_rouge', 'wer', 'word_info_lost', 'word_info_preserved'])
result_basename=f'formal_test_results_each_sentence'
sentence_metrics_list=[]
decoded_preds_list=[]
decoded_labels_list=[]
for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            input_features = batch["input_features"].cuda()
            if args.noise:
                input_features=torch.randn_like(input_features)

            generated_tokens = (
                model.generate(input_features,do_sample=False,num_beams=5,repetition_penalty=5.0,
                               decoder_input_ids=batch["labels"][:, :4].cuda(),)
            ).cpu().numpy()

            labels = batch["labels"].cpu().numpy()
            # print(f'labels:{labels}')
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # 将预测和实际的 token 转换为文本
            if not args.random_choice:
                decoded_preds = processor.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
            # decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds=filter_ascii_text(decoded_preds)
            decoded_labels=filter_ascii_text(decoded_labels)
            # print('decoded_labels')
            # print(decoded_labels)
            # print('\n')
            # print('end')
            sentence_metrics_single=sentence_metrics_calculator.compute(predictions=decoded_preds, references=decoded_labels)
            sentence_metrics_list.extend(sentence_metrics_single)
            decoded_preds_list.extend(decoded_preds)
            decoded_labels_list.extend(decoded_labels)

# 将结果都写入data_jsonl，并保存到测试模型文件夹下
for i,data in enumerate(data_jsonl):
    data['pred']=decoded_preds_list[i]
    data['label']=decoded_labels_list[i]
    data['metrics']=sentence_metrics_list[i]
filename=os.path.basename(args.test_data).split('.')[0]
jsonl_path=os.path.join(args.lora_model,f'{filename}_each_results_and_metrics.jsonl')
write_jsonlines(jsonl_path,data_jsonl)

