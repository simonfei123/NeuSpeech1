import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os


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
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/28/whisper-base/checkpoint-2000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/29/whisper-base/checkpoint-5000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/30/whisper-base/checkpoint-9000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/31/whisper-base/checkpoint-12000',
]
tune_layer_list=[0.2,0.4,0.6,0.8]
marker_list=['o','s','d']
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
# epoch_list=[119.402,104.477,46.641,25.186,15.398]
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
# 图一，横轴是mask ratio，纵轴是表现分。只画出BLEU-1的分数
# plt.figure(figsize=(10,6))
# for mi,m in enumerate(wanted_metrics):
#     plt.plot([metrics_list[i][m] for i in range(5)])
# plt.legend(wanted_metrics_alter_name)
# plt.xticks(np.arange(len(ckpt_path_list)),size_name_list)
#
# plt.show()
# 画图
matplotlib.rcParams['font.size'] = 16
fig, ax1 = plt.subplots(figsize=(6, 4))

# 绘制指标曲线
# for tune_layer_idx,tune_layer in enumerate(tune_layer_list):
for mi, m in enumerate(wanted_metrics):
    ax1.plot(tune_layer_list,[metrics_list[i][m] for i in range(4)],
             label=wanted_metrics_alter_name[mi],marker='o'
             )

for i, tune_layer in enumerate(tune_layer_list):
    # print(j, metrics_list[i*3+j]['bleu-1'])
    # bias=0
    # if tune_layer==0.5:
    #     bias=-5
    ax1.text(tune_layer, metrics_list[i]['bleu-1']+0.5, str(np.round(metrics_list[i]['bleu-1'],2)),
             ha='center', va='bottom')

ax1.hlines(41.66,xmin=0.2,xmax=0.8,linestyles='--',colors='black')
ax1.annotate('Baseline:41.66', xy=(3, 41.66), xytext=(3, 42),
            arrowprops=None, ha='center')
# 设置X轴刻度和标签
plt.xticks(tune_layer_list)

# 设置图例
lines, labels = ax1.get_legend_handles_labels()
# print(lines)
# ax1.legend(lines, mask_name_list, loc='lower right', bbox_to_anchor=(0.6, 0.15))
ax1.set_ylim(8,49)
# 设置Y轴标签
ax1.set_xlabel('Data Ratio')
ax1.set_ylabel('BLEU-1')
plt.tight_layout()
plt.savefig('../figures/performance_with_data.pdf')
# 显示图形
plt.show()