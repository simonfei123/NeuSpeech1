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
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/23/whisper-base/checkpoint-7000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/24/whisper-base/checkpoint-12000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/25/whisper-base/checkpoint-16000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/26/whisper-base/checkpoint-16000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/27/whisper-base/checkpoint-16000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/28/whisper-base/checkpoint-2000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/29/whisper-base/checkpoint-5000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/30/whisper-base/checkpoint-9000',
    '/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/31/whisper-base/checkpoint-12000',
]
layers_list=[i for i in range(1,6)]
layers_name_list=[i for i in range(1,6)]
prob_list=[np.round(i*0.2,1) for i in range(1,5)]
prob_replace_list=[i for i in range(1,5)]
marker_list=['o','s','d']
marker2_list=['.','v','*']
mask_ratio_list=[0.25,0.5,0.75]
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
matplotlib.rcParams['font.size'] = 26
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ax2=ax1.twiny()

# 绘制指标曲线
print()
for mi, m in enumerate(wanted_metrics):
    ax1.plot(layers_name_list,[metrics_list[i][m] for i in range(5)],
             label=wanted_metrics_alter_name[mi],marker=marker_list[0],
             color=color_list[0]
             )

for i, layer in enumerate(layers_name_list):
    # print(j, metrics_list[i*3+j]['bleu-1'])
    bias=0
    xbias=0
    if layer==2 or layer==3:
        bias=1.5
    if layer==5:
        bias=-8
    if layer==4:
        bias=-9
        xbias=0.1
    ax1.text(layer+xbias, metrics_list[i]['bleu-1']+0.5+bias, str(np.round(metrics_list[i]['bleu-1'],2)),
             ha='center', va='bottom',color=color_list[0])

ax1.hlines(41.66,xmin=1,xmax=5,linestyles='--',colors='black')
# ax1.vlines(1,ymin=0,ymax=50,linestyles='--',colors='blue')
# ax2.vlines(1,ymin=0,ymax=50,linestyles='--',colors='black')
ax1.annotate('Baseline:41.66', xy=(3, 42), xytext=(3, 35),
            arrowprops=None, ha='center')

# ax2
for mi, m in enumerate(wanted_metrics):
    print(prob_replace_list,[metrics_list[i +5][m] for i in range(4)])
    ax2.plot(np.array(prob_replace_list), [metrics_list[i +5][m] for i in range(4)],
             label=wanted_metrics_alter_name[mi], marker=marker2_list[2],
             color=color_list[2]
             )


for i, da_type in enumerate(prob_replace_list):
    # print(j, metrics_list[i*3+j]['bleu-1'])
    bias=0
    x_bias=0
    if da_type==1 or da_type==2 or da_type==3:
        bias=2
        if da_type==1:
            x_bias=0.1
    if da_type==4:
        bias=-10
        x_bias=0
    if da_type==3:
        bias=-8
        x_bias=0
    # if da_type=='SNR 15dB' and mask_ratio==1:
    #     bias=0.5

    ax2.text(prob_replace_list[i]+x_bias, metrics_list[i+5]['bleu-1']+bias, str("{:.2f}".format(np.round(metrics_list[i+5]['bleu-1'],2))),
             ha='center', va='bottom',color=color_list[2])


# 设置X轴刻度和标签
plt.xticks(mask_ratio_list)
# ax1.set_xlim(0.2, 0.8)
bias=0.2
ax1.set_xlim(1-bias,5+bias)
ax2.set_xlim(1-0.5, 4+0.5)
ax1.set_xticks(layers_list)
ax1.set_xticklabels(layers_name_list,color=color_list[0])
ax2.set_xticks(prob_replace_list)
ax2.set_xticklabels(prob_list,color='g')
# 设置图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend=ax1.legend(lines + lines2, ['Layers','Data Ratio'], loc='lower right', bbox_to_anchor=(0.35, 0.28))
legend.get_frame().set_alpha(0)
# print(lines)
# ax1.legend(lines, mask_name_list, loc='lower right', bbox_to_anchor=(0.6, 0.15))
ax1.set_ylim(8,48)
# 设置Y轴标签
ax1.set_xlabel('Fine-tune Layers',color=color_list[0])
ax2.set_xlabel('Data Ratio',color='g')
ax1.set_ylabel('BLEU-1')
plt.tight_layout()
plt.savefig('../figures/performance_with_data_layers.pdf')
# 显示图形
plt.show()