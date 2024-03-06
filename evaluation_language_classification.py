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
from utils.model_utils import projection_module
from peft import PeftModel

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding,generate_random_string, remove_punctuation, to_simple,contains_valid_letters
from utils.process_str import filter_ascii_text,model_generate,convert_lower_text,list_operation
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

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                    device_map="auto",
                                                    local_files_only=args.local_files_only,)

device=model.device
kwargs={
    'meg_ch':args.eeg_ch,
    'd_model':model.model.encoder.conv2.in_channels,
}

conv1=projection_module(config_name=args.config_name,**kwargs)

# conv1 = nn.Conv1d(meg_ch, d_model, kernel_size=3, padding=1)
conv1 = conv1.to(device)
model.model.encoder.set_input_embeddings(conv1)
if args.lora_model is not None:
    model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=args.local_files_only)
    model = model.merge_and_unload()
    print(model)
if args.random_initialize_whisper:
    model.model.decoder.post_init()
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
                             filter_dataset=args.filter_dataset,
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
metrics = []
# metric_files = ['bert_score','bleu','mer', 'my_rouge','perplexity', 'wer','word_info_lost','word_info_preserved']
metric_files = ['bleu','mer', 'my_rouge','wer','word_info_lost','word_info_preserved']
# Load metrics
for metric_file in metric_files:
    metric = evaluate.load(f'metrics/{metric_file}.py',
                           experiment_id=generate_random_string(100))
    metrics.append(metric)


# repeat_eval_times=1
# num_beams_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]
# bleu_results={}
# for num_beams in tqdm.tqdm(num_beams_list):
#     print('*'*50)
#     print(f'num_beams:{num_beams} \n\n')
if args.random_choice:
    all_labels=[]
# result_basename=(f'formal_test_results{"_"+args.extra_name if args.extra_name is not None else ""}'
#                  f'{"no_post_processing" if not args.post_processing else "post_processing"}'
#                  f'{"_noise"if args.noise else ""}{"_randomChoice"if args.random_choice else ""}'
#                  f'{"_tf" if args.teacher_forcing else ""}')
# 开始评估
# output_file=os.path.join(args.lora_model,f'{result_basename}.txt')
# with open(output_file, "w") as f:
for step, batch in tqdm.tqdm(enumerate(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if not args.random_choice:
                input_features = batch["input_features"].cuda()
                if args.noise:
                    input_features=torch.randn_like(input_features)
                if not args.teacher_forcing:
                    if args.language.lower() != 'english':
                        decoder_input_ids = batch["labels"][:, :4].cuda()
                        generation_kwargs={"decoder_input_ids":decoder_input_ids}
                    else:
                        generation_kwargs={}
                    generated_tokens = (

                        model.generate(input_features,is_multilingual=True,language=None,return_token_timestamps=True,
                                       # max_new_tokens=100,
                                       forced_decoder_ids=forced_decoder_ids
                                       )
                    )
                else:
                    # print(batch["labels"])
                    # print(batch["labels"].shape)
                    # exit()
                    # 50257
                    indices=batch["labels"]==-100
                    batch["labels"][indices]=50257
                    model_output=model(input_features,decoder_input_ids=batch["labels"].cuda(),)
                    logits=model_output.logits
                    # logits=logits.to('cpu').numpy()
                    # print(f'logits shape:{logits.shape}')
                    values,predictions=logits.softmax(dim=-1).topk(1)
                    # print(f'predictions shape:{predictions.shape}')
                    predictions=torch.squeeze(predictions,dim=-1)
                    # print(f'predictions:{predictions}')
                    generated_tokens=predictions.cpu().numpy()
                    generated_tokens[indices]=-100
                    # print(f'generated_tokens:{generated_tokens.shape}')

            labels = batch["labels"].cpu().numpy()
            # print(f'labels:{labels}')
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # 将预测和实际的 token 转换为文本
            # generated_tokens=generated_tokens[:,1]
            # labels=labels[:,1]
            decoded_preds = processor.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
            print('decoded_preds')
            print(decoded_preds)
            print('decoded_labels')
            print(decoded_labels)
            print('\n')
            print('end')
