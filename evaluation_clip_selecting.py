import jsonlines
import os
import sys
from sentence_transformers import SentenceTransformer, util
import evaluate
import json
from utils.data_utils import generate_random_string
import numpy as np
import torch
# 获取当前脚本的文件路径
current_path = os.path.abspath(__file__)
# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(current_path))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import argparse
import functools
from utils.utils import add_arguments


def write_jsonlines(file_path, json_dicts):
    with jsonlines.open(file_path, mode='w') as writer:
        for json_dict in json_dicts:
            writer.write(json_dict)


def read_jsonlines(file_path):
    json_dicts = []
    with jsonlines.open(file_path, mode='r') as reader:
        for json_obj in reader:
            json_dicts.append(json_obj)
    return json_dicts

def makedirs(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    return path

if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("query_jsonl",    type=str, default=None,       help="jsonl文件路径")
    add_arg("corpus_jsonl",    type=str, default=None,       help="jsonl文件路径")
    add_arg("output_dir",    type=str, default=None,       help="输出文件路径")
    args = parser.parse_args()
    metrics = []
    # metric_files = ['bert_score','bleu','mer', 'my_rouge','perplexity', 'wer','word_info_lost','word_info_preserved']
    metric_files = ['bleu','mer', 'my_rouge','wer','word_info_lost','word_info_preserved','bert_score','meteor']
    # Load metrics
    for metric_file in metric_files:
        metric = evaluate.load(f'metrics/{metric_file}.py',
                               experiment_id=generate_random_string(100))
        metrics.append(metric)
    result_basename = 'formal_test_results_clip'
    datas = read_jsonlines(args.query_jsonl)
    query_sentences = [line['sentence'] for line in datas]
    unique_query_sentences = set(query_sentences)
    datas = read_jsonlines(args.corpus_jsonl)
    corpus_sentences = list({line['sentence'] for line in datas})

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embedder.encode(corpus_sentences, convert_to_tensor=True)
    # 找到每个unique query对应的corpus，做个字典。然后再产生query对应的value。
    qv_dict={}
    for query in unique_query_sentences:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=1)
        best_sentence = corpus_sentences[top_results[1][0]]

        # print("\n\n======================\n\n")
        print("Query:", query)
        print("value:", best_sentence)
        # print("\nTop 5 most similar sentences in corpus:")
        qv_dict[query]=best_sentence
    value_sentences=[qv_dict[q] for q in query_sentences]

    output_file=os.path.join(args.output_dir,f'{result_basename}.txt')
    with open(output_file, "w") as f:
        for prediction,label in zip(value_sentences,query_sentences):
            f.write(f"start********************************\n")
            f.write(f"Predicted: {prediction}\n")
            f.write(f"True: {label}\n")
            f.write(f"end==================================\n\n")

    for metric in metrics:
        metric.add_batch(predictions=value_sentences, references=query_sentences)

    results={}
    for metric in metrics:
        result = metric.compute()
        for key in result.keys():
            if type(result[key])==torch.Tensor:
                result[key]=result[key].item()
            results[key]=result[key]
    print(results)
    output_file=os.path.join(args.output_dir,f'{result_basename}.json')
    with open(output_file,'w') as f:
        json.dump(results,f)