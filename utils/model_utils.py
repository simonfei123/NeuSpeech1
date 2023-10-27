import torch
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn as nn
import torch.nn.functional as F
import os

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def load_from_checkpoint(resume_from_checkpoint, model=None):
    pass


def trainer_save_model(output_dir=None, state_dict=None):
    os.makedirs(output_dir, exist_ok=True)


def compute_accuracy(pred):
    ## 1.处理 pred.predictions
    # 每个样本的预测结果为vocab大小
    predict_res = torch.Tensor(pred.predictions[0])  # size：[验证集样本量, label的token长度, vocab大小]
    pred_ids = predict_res.argmax(dim=2)

    ## 2.处理 pred.label_ids
    labels_actual = torch.LongTensor(pred.label_ids)

    ## 3.计算accuracy
    total_num = labels_actual.shape[0]
    acc = torch.sum(torch.all(torch.eq(pred_ids, labels_actual), dim=1)) / total_num
    return {'accuracy': acc}

