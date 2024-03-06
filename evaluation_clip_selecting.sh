python evaluation_clip_selecting.py\
 --query_jsonl='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
  --corpus_jsonl='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/train.jsonl'

python evaluation_clip_selecting.py \
 --query_jsonl='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --corpus_jsonl='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/train.jsonl'\
 --output_dir='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/48-gwilliams2023-split4/whisper-base/checkpoint-96000'

 python evaluation_clip_selecting.py \
 --query_jsonl='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --corpus_jsonl='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/train.jsonl'\
 --output_dir='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/51-gwilliams2023-split4/whisper-base/checkpoint-97000'
