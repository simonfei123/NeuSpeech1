CUDA_LAUNCH_BLOCKING=1 NCCL_SOCKET_IFNAME=eth2 torchrun --nproc_per_node=2  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/5-schoffelen2019n-audio-double'\
 --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda' \
 --resume_from_checkpoint='output_models/5-schoffelen2019n-audio-double/whisper-base/checkpoint-4000'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/5-schoffelen2019n-audio'\
 --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'


# 为了防止双卡故障浪费时间，单卡也要跑起来。
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/7-schoffelen2019n-audio-double'\
 --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cpu' \
 --resume_from_checkpoint='output_models/7-schoffelen2019n-audio-double/whisper-base/checkpoint-6500'


python finetune.py --per_device_train_batch_size=16\
 --per_device_eval_batch_size=16 --output_dir='output_models/7-schoffelen2019n-audio-cpu'\
 --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cpu' \
 --resume_from_checkpoint='output_models/7-schoffelen2019n-audio-double/whisper-base/checkpoint-6500'


CUDA_LAUNCH_BLOCKING=1 NCCL_SOCKET_IFNAME=eth2 torchrun --nproc_per_node=2  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/8-schoffelen2019n-audio'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=301 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda' \
 --resume_from_checkpoint='output_models/8-schoffelen2019n-audio/whisper-base/checkpoint-40000'


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/15-schoffelen2019n-audio-sentences'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=301 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base' --filter_dataset=True\
 --local_files_only=False --language='Dutch' --device='cuda'\
 --resume_from_checkpoint='output_models/15-schoffelen2019n-audio-sentences/whisper-base/checkpoint-5000'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/22-schoffelen-pretrain-9'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=301 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base' --filter_dataset=True\
 --local_files_only=False --language='Dutch' --device='cuda'\
 --lora_model='output_models/9-gwilliams2023-split1/whisper-base/checkpoint-180000'\
 --lora_eeg_ch=224


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/26-schoffelen'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base' --filter_dataset=True\
 --local_files_only=False --language='Dutch' --device='cuda'

#train28
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/28-schoffelen'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base' --filter_dataset=True --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'


#train30
CUDA_VISIBLE_DEVICES=1 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/30-schoffelen-pretrain27'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base' --filter_dataset=True --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'\
 --lora_model='output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000'\
 --lora_eeg_ch=208\
 --resume_from_checkpoint='output_models/30-schoffelen-pretrain27/whisper-base/checkpoint-12000'


# eval15
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/15-schoffelen2019n-audio-sentences/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=301 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True

# eval22
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/22-schoffelen-pretrain-9/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=301 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --filter_dataset=True

# eval26
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/26-schoffelen/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --filter_dataset=True

# eval28
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/28-schoffelen/whisper-base/checkpoint-14000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --filter_dataset=True

# eval30
python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000/full_model'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/30-schoffelen-pretrain27/whisper-base/checkpoint-12000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/audio_split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --filter_dataset=True

# 语音
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/5-schoffelen2019n-audio1' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=4 \
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess3/audio_split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language=None


 exec bash
 conda activate brainimagick
 cd ~/brain2text/whisper/

