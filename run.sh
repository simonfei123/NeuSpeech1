#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --use_8bit=False --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-base --use_8bit=False --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-base/checkpoint-final

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py  --per_device_train_batch_size=2 --per_device_eval_batch_size=2
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-base/checkpoint-final

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 finetune.py
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-base/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-small --use_8bit=True --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-small/checkpoint-final

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-medium --use_8bit=True --per_device_train_batch_size=4 --per_device_eval_batch_size=2 --gradient_accumulation_steps=2
CUDA_VISIBLE_DEVICES=5 python merge_lora.py --lora_model=output/large_finetune/checkpoint-best --output_dir=output/large_finetune/checkpoint-best-after-lora

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-large-v2 --use_8bit=True --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --gradient_accumulation_steps=4
CUDA_VISIBLE_DEVICES=5 python merge_lora.py --lora_model=output/large_finetune/checkpoint-final

CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-tiny-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-base-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-small-finetune
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-medium-finetune
CUDA_VISIBLE_DEVICES=5 python evaluation.py --model_path=models/whisper-large-v2-finetune
CUDA_VISIBLE_DEVICES=5 python evaluation.py --model_path="/home/yyang/research/eeg2text/output/base/checkpoint-final/" --batch_size=1 --num_workers=1
/home/yyang/research/eeg2text


CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py  --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --output_dir='output3' --eval_steps=100 --save_steps=100 --learning_rate=2e-4
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 full_finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output7' --eval_steps=20 --save_steps=20 --learning_rate=1e-3 --fp16=True\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output11' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --use_8bit=False --num_workers=6 --modal='speech' --sampling_rate=16000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'

 # 训练eeg
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output13' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=3\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output14' --eval_steps=1000 --save_steps=1000 --learning_rate=2e-4 --fp16=True\
 --num_train_epochs=3\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output15' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=3\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output16' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=3\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output17' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=3\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

# output18
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output18' --eval_steps=1000 --save_steps=1000 --learning_rate=5e-4 --fp16=True\
 --num_train_epochs=30\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

CUDA_VISIBLE_DEVICES=4 python merge_lora.py --modal='eeg' --output_dir='models2/'\
 --lora_model="/home/yyang/research/eeg2text/output18/whisper-small-finetune/checkpoint-35000/"

CUDA_VISIBLE_DEVICES=4 python evaluation.py\
 --test_data=/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/test_data.jsonl\
 --model_path=models2/whisper-small-finetune-finetune/\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64

# output19
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output19' --eval_steps=1000 --save_steps=1000 --learning_rate=5e-4 --fp16=True\
 --num_train_epochs=30\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='models2/whisper-small-finetune-finetune/'

CUDA_VISIBLE_DEVICES=4 python merge_lora.py --modal='eeg' --output_dir='models3/'\
 --lora_model="/home/yyang/research/eeg2text/output18/whisper-small-finetune/checkpoint-35000/"

CUDA_VISIBLE_DEVICES=4 python evaluation.py\
 --test_data=/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/test_data.jsonl\
 --model_path=models2/whisper-small-finetune-finetune/\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64

# output20 full ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 full_finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output20' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=30\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='models2/whisper-small-finetune-finetune/'

# output21 full ft from
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=25641 full_finetune.py --per_device_train_batch_size=2\
 --per_device_eval_batch_size=2 --output_dir='output21' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=300\
 --use_8bit=False --num_workers=2 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

# output22 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output22' --eval_steps=1000 --save_steps=1000 --learning_rate=5e-4 --fp16=True\
 --num_train_epochs=90\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='models2/whisper-small-finetune-finetune/'

# output23 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output23' --eval_steps=1000 --save_steps=1000 --learning_rate=5e-4 --fp16=True\
 --num_train_epochs=90 --warmup_steps=10000\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='models2/whisper-small-finetune-finetune/'

# output24 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output24' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=90 --warmup_steps=10000\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

# output24 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output24' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=90 --warmup_steps=10000\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'


# output25 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output25' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=10\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'
# --lora_model="/home/yyang/research/eeg2text/output24/whisper-small-finetune/checkpoint-final/"

# output26 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output26' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=10\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'
 --lora_model="/home/yyang/research/eeg2text/output25/whisper-small-finetune/checkpoint-2000/"

CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model='/home/yyang/research/eeg2text/output26/whisper-small-finetune/checkpoint-5500/'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64 --batch_size=1 --num_workers=2

# output27 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output27' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=100\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'


 CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model='/home/yyang/research/eeg2text/output27/whisper-small-finetune/checkpoint-13500/'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64 --batch_size=1 --num_workers=2


# output28 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output28' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=100\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'


# output29 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output29' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=100\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model="/home/yyang/research/eeg2text/output28/whisper-small-finetune/checkpoint-30000/"

CUDA_VISIBLE_DEVICES=4 python merge_lora.py --modal='eeg' --output_dir='models2/'\
 --lora_model="/home/yyang/research/eeg2text/output18/whisper-small-finetune/checkpoint-35000/"

CUDA_VISIBLE_DEVICES=4 python evaluation.py\
 --test_data=/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/test_data.jsonl\
 --model_path=models2/whisper-small-finetune-finetune/\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64



# output30 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output30' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=100\
 --use_8bit=False --num_workers=6 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg5s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg5s_singe_sentence/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model='/home/yyang/research/eeg2text/output30/whisper-small-finetune/checkpoint-215000/'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg5s_singe_sentence/test_data.jsonl'\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64 --batch_size=16 --num_workers=4


# output31 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='output31' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg5s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg5s_singe_sentence/val_data.jsonl'\
 --lora_model='/home/yyang/research/eeg2text/output30/whisper-small-finetune/checkpoint-215000/'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

# output32 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='output32' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=1000\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'
 # 这个在训练集上非常好，可以达到0.18以下，在验证集上是1.4 。 是因为出问题了。resample 的时候并没有真的操作，实际的采样率仍然是1000Hz。

# output33 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='output33' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=200\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

# output34 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='output34' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=200\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/val_data.jsonl'\
 --lora_model="/home/yyang/research/eeg2text/output33/whisper-small-finetune/checkpoint-275000/"\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'


CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model="/home/yyang/research/eeg2text/output34/whisper-small-finetune/checkpoint-375000/"\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=64 --batch_size=16 --num_workers=4


# 这个35是只用了64个数据来测试是否跑通，因为之前34在训练数据上也没有成功
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='output35' --eval_steps=100 --save_steps=100 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=50\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=200\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'

# 35说明这样训练和测试的流程是对的。这样也就说明34可能是训练太多废了。
CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model="/home/yyang/research/eeg2text/output35/whisper-small-finetune/checkpoint-2000/"\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=64 --batch_size=16 --num_workers=4


# output36 lora ft
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=12 --output_dir='output36' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=15\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=200\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'



# output37 lora ft train on one subject
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=12 --output_dir='output37' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=7.5\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=64 --sampling_rate=400\
 --train_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/sub_1_train_data.jsonl'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s_singe_sentence/sub_1_val_data.jsonl'\
 --base_model='/home/yyang/dataset/multi_media/transformers_whisper_models/tiny-finetune'

# 下载预训练模型
bypy downdir transformers_whisper_models/whisper-small-finetune /home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune

CUDA_VISIBLE_DEVICES=4 python merge_lora.py --modal='eeg' --output_dir='models2/'\
 --lora_model="/home/yyang/research/eeg2text/output18/whisper-small-finetune/checkpoint-35000/"
#cp -rf /home/yyang/dataset/multi_media/transformers_whisper_models/large_finetune/*.json output/large_finetune/checkpoint-best
cp -rf /home/yyang/dataset/multi_media/transformers_whisper_models/large_finetune/*.{json,txt} output/large_finetune/checkpoint-best
cp -rf /home/yyang/dataset/multi_media/transformers_whisper_models/large_finetune/*.{json,txt} output4/large_finetune/checkpoint-100
CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --test_data=/home/yyang/dataset/multi_media/formal_dataset/cut/train_data.jsonl\
 --model_path=output/large_finetune/checkpoint-12000


CUDA_VISIBLE_DEVICES=5 python evaluation.py --model_path=output/large_finetune/checkpoint-final
CUDA_VISIBLE_DEVICES=4 python merge_lora.py --output_dir='models1/' --lora_model="/home/yyang/research/eeg2text/output9/large_finetune/checkpoint-best/" --base_model=/home/yyang/research/eeg2text/output2/large_finetune/checkpoint-init
CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model='/home/yyang/research/eeg2text/output25/whisper-small-finetune/checkpoint-1500/'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64 --batch_size=1 --num_workers=2

CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model='/home/yyang/research/eeg2text/output25/whisper-small-finetune/checkpoint-2000/'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/train_data.jsonl'\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64 --batch_size=1 --num_workers=2

CUDA_VISIBLE_DEVICES=5 python evaluation.py\
 --model_path='/home/yyang/dataset/multi_media/transformers_whisper_models/whisper-small-finetune'\
 --lora_model='/home/yyang/research/eeg2text/output24/whisper-small-finetune/checkpoint-final/'\
 --test_data='/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/test_data.jsonl'\
 --modal='eeg' --sampling_rate=1000 --eeg_ch=64 --batch_size=8 --num_workers=2


CUDA_VISIBLE_DEVICES=4 python evaluation.py\
 --test_data=/home/yyang/dataset/multi_media/formal_dataset/cut_seg10s/test_data.jsonl\
 --model_path=models2/large_finetune-finetune/\
 --modal='speech' --sampling_rate=16000 --eeg_ch=1

