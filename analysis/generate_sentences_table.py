import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os
import copy


def list_operation(l,f):
    for li,_ in enumerate(l):
        l[li]=f(_)
    return l

def rectify_metrics(jd:dict):
    for k,v in jd.items():
        # print(k,v)
        if k.startswith('bleu') or k.startswith('wer'):
            jd[k]=v*100
    return jd

def get_json_list(path_list):
    json_list = []
    for json_path in path_list:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
            json_list.append(json_dict)
    return json_list


def read_txt(path):
    lines = []
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines





def get_samples_from_num(file_list,num):
    p_list=[]
    for i,file in enumerate(file_list):
        pn=num*5+1
        ln=num*5+2
        # print(file[pn])
        # print(file[ln])
        file=copy.deepcopy(file)
        p=file[pn].split(':')[1].strip()
        l=file[ln].split(':')[1].strip()
        if i==0:
            p_list.append(l)
        p_list.append(p)
        print(i,p_list)
    return p_list


def get_part_table_for_samples(samples):
    content=("\multicolumn{1}{c}{\multirow{7}{*}{(1)}} & "+
             f"GT                &    {samples[0]}      "+r" \\ "+
             "\multicolumn{1}{c}{}  "+f"& NeuSpeech&     {samples[1]}"+r" \\ "+
             "\multicolumn{1}{c}{}  "+f"& NeuSpeech w/ tf&     {samples[2]}"+r" \\ "+
             "\multicolumn{1}{c}{}  "+f"& NeuSpeech w/ pt&     {samples[3]}"+r" \\ "+
             "\multicolumn{1}{c}{}  "+f"& NeuSpeech w/ jt&     {samples[4]}"+r" \\ "+
             "\multicolumn{1}{c}{}  "+r"& eeg-to-text~\cite{wang2022open_aaai_eeg2text}"+f"&     {samples[5]}"+r" \\ "+
             "\multicolumn{1}{c}{}  "+r"& eeg-to-text~\cite{wang2022open_aaai_eeg2text} w/ tf"+f"&     {samples[6]}"+r" \\ "
             )
    return content


def get_full_tabular_for_list(file_list,indexes):
    table=''
    for idx in indexes:
        samples=get_samples_from_num(file_list,idx)
        line=get_part_table_for_samples(samples)
        table+=line+'\n'
    return table


txt_path_list=[
    '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000/formal_test_resultsno_post_processing.txt',
    '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000/formal_test_resultsno_post_processing_tf.txt',
    # '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000',
    # '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/29-gwilliams2023-split2-no_aug/whisper-base/checkpoint-48000',
    # '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/30-schoffelen-pretrain27/whisper-base/checkpoint-12000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/31-gwilliams2023-split1-no_aug-pretrain28/whisper-base/checkpoint-44000/formal_test_resultsno_post_processing.txt',
    '/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000/formal_test_results_gwilliamsno_post_processing.txt',
    '/hpc2hdd/home/yyang937/brain2text/eeg-to-text/results/formal_test_resultsno_post_processing.txt',
    '/hpc2hdd/home/yyang937/brain2text/eeg-to-text/results/formal_test_resultsno_post_processing_tf.txt',
]

txt_name='formal_test_results_gwilliamsno_post_processing.txt'
txt_tf_name='formal_test_results_gwilliamsno_post_processing_tf.txt'
# txt_path_list=[os.path.join(path,txt_name) for path in ckpt_path_list]
txt_file_list=[read_txt(path) for path in txt_path_list]
# print(txt_file_list)
table=get_full_tabular_for_list(txt_file_list,[10,10,10,10,10])
print(table)
