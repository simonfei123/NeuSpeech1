import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib


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


ckpt_path_list=[
    'brain2text/whisper/abalation_models/3/tiny/checkpoint-16000',
    'brain2text/whisper/abalation_models/1/whisper-base/checkpoint-14000',
    'brain2text/whisper/abalation_models/4/small/checkpoint-25000',
    'brain2text/whisper/abalation_models/5/medium/checkpoint-27000',
    'brain2text/whisper/abalation_models/6/whisper-large/checkpoint-33000'
]
size_name_list=['Tiny','Base','Small','Medium','Large']
epoch_list=[119.402,104.477,46.641,25.186,15.398]
json_file_name='formal_test_resultsno_post_processing.json'
trainer_state_file_name='trainer_state.json'
home_dir = os.path.expanduser("~")
json_path_list=[os.path.join(home_dir,ckpt_path_list[i],json_file_name) for i,_ in enumerate(ckpt_path_list)]
trainer_state_path_list=[os.path.join(home_dir,ckpt_path_list[i],trainer_state_file_name) for i,_ in enumerate(ckpt_path_list)]

wanted_metrics=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4','rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall'][:1]
wanted_metrics_alter_name=['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4','ROUGE1-F', 'ROUGE1-P', 'ROUGE1-R'][:1]

metrics_list=get_json_list(json_path_list)
metrics_list=list_operation(metrics_list,rectify_metrics)
# ts_list=get_json_list(trainer_state_path_list)
# 画图
# 图一，横轴是模型尺寸，纵轴是各个表现分。不同系列metrics使用不同的颜色，同系列的节点使用不同的样式。图例要给出。
# plt.figure(figsize=(10,6))
# for mi,m in enumerate(wanted_metrics):
#     plt.plot([metrics_list[i][m] for i in range(5)])
# plt.legend(wanted_metrics_alter_name)
# plt.xticks(np.arange(len(ckpt_path_list)),size_name_list)
#
# plt.show()
# 画图
matplotlib.rcParams['font.size'] = 26
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

# 绘制指标曲线
for mi, m in enumerate(wanted_metrics):
    ax1.plot([metrics_list[i][m] for i in range(5)], label=wanted_metrics_alter_name[mi],marker='s',color='b')

for i, _ in enumerate(size_name_list):
    bias=0
    xbias=0
    if _=='Large':
        xbias=-0.2
        bias=4
    if _=='Tiny':
        xbias=0.6
        bias=0
    if _=='Base':
        xbias=-0.3
        bias=0
    ax1.text(i+xbias, metrics_list[i]['bleu-1']+bias, str(np.round(metrics_list[i]['bleu-1'],2)), ha='center', va='bottom',color='b')
# 绘制epoch曲线
ax2.plot(epoch_list,color='green', linestyle='--', label='Epoch',marker='o')
for i, epoch in enumerate(epoch_list):
    xbias=0.0
    bias=0
    if i==0:
        xbias=0.1
        bias=-15
    if i==1:
        xbias=0
        bias=2
    if i==2:
        xbias=0.1
        bias=0
    ax2.text(i+xbias, epoch+bias, str(int(np.ceil(epoch))), ha='center', va='bottom',color='green')
# 设置X轴刻度和标签
plt.xticks(np.arange(len(size_name_list)), size_name_list)

# 设置图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower right', bbox_to_anchor=(0.95, 0.2))

ax1.set_xlim(-0.1, 4.2)
ax1.set_ylim(14, 70)
ax2.set_ylim(14, 122)
# 设置Y轴标签
ax1.set_xlabel('Sizes')
ax1.set_ylabel('BLEU-1',color='b')
ax2.set_ylabel('Epoch',color='g')
plt.tight_layout()
plt.savefig('/hpc2hdd/home/yyang937/brain2text/whisper/figures/performance_with_sizes.pdf')
# 显示图形
plt.show()