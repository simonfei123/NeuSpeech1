CUDA_VISIBLE_DEVICES=7 python finetune.py --per_device_train_batch_size=64 --per_device_eval_batch_size=64 --output_dir='output1' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False --num_train_epochs=200 --warmup_steps=500 --max_audio_len=30 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200 --train_data='datasets/gwilliams2023/preprocess/split1/train.jsonl' --test_data='datasets/gwilliams2023/preprocess/split1/val.jsonl' --base_model='openai/whisper-base' --local_files_only=False --device cuda:7

CUDA_VISIBLE_DEVICES=7 python evaluation.py --model_path='openai/whisper-base' --lora_model="output1/whisper-base/checkpoint-final" --test_data='datasets/gwilliams2023/preprocess/split1/test.jsonl' --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=1 --num_workers=4 --language='English' --timestamps=False --local_files_only=False
CUDA_VISIBLE_DEVICES=7 python evaluation.py --model_path='openai/whisper-base' --lora_model="output1/whisper-base/checkpoint-29000" --test_data='datasets/gwilliams2023/preprocess/split1/test.jsonl' --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=1 --num_workers=4 --language='English' --timestamps=False --local_files_only=False

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=12 --output_dir='output1' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 finetune.py --per_device_train_batch_size=8 --per_device_eval_batch_size=12 --output_dir='output1' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=False --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200 --train_data='datasets/gwilliams2023/preprocess/split1/train.jsonl' --test_data='datasets/gwilliams2023/preprocess/split1/val.jsonl' --base_model='openai/whisper-base' --local_files_only=False

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output1' --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=200 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False


python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output1/whisper-base/checkpoint-46500"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=1 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=False


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output2' --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output2/whisper-base/checkpoint-116000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=128 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=False


python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output2/whisper-base/checkpoint-116000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=128 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=False --noise=False


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/3-try-save' --eval_steps=50 --save_steps=50\
  --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=50 --warmup_steps=5 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False


NCCL_SOCKET_IFNAME=eth2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/3-try-save' --eval_steps=500 --save_steps=500\
  --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=5 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/6-base+bigKernel' --eval_steps=500 --save_steps=500\
  --learning_rate=1e-3 --fp16=False\
 --num_train_epochs=500 --warmup_steps=5 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'\
 --config_name='base+bigKernel'

# train9
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/9-gwilliams2023-split1'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'

# train25
python finetune.py --per_device_train_batch_size=32\
 --per_device_eval_batch_size=32 --output_dir='output_models/25-gwilliams2023-split1'\
  --eval_steps=2000 --save_steps=2000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'\
 --resume_from_checkpoint='output_models/25-gwilliams2023-split1/whisper-base/checkpoint-44000'

# train27
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/27-gwilliams2023-split1-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

# train31
CUDA_VISIBLE_DEVICES=0 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/31-gwilliams2023-split1-no_aug-pretrain28'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'\
 --lora_model='output_models/28-schoffelen/whisper-base/checkpoint-14000'\
 --lora_eeg_ch=273

# train29
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/29-gwilliams2023-split2-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

python debug_lora.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/29-gwilliams2023-split2-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda' --fine_tune_layers=0

# train34
python finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='output_models/34-gwilliams2023-split1-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-large' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

# train35
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com python finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='output_models/35-gwilliams2023-split1-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-medium' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

HF_ENDPOINT=https://hf-mirror.com python finetune.py --per_device_train_batch_size=16\
 --per_device_eval_batch_size=16 --output_dir='output_models/36-gwilliams2023-split1-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=8 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-small' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/10-gwilliams2023-split2'\
  --eval_steps=5000 --save_steps=5000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'


python finetune_speech.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/14-speech-gwilliams2023' \
 --eval_steps=500 --save_steps=500 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=30 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='speech' \
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English'


# train16
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/16-finetune-gwilliams2023-15-dutch' \
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200 \
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English'\
 --lora_model='output_models/15-schoffelen2019n-audio-sentences/whisper-base/checkpoint-final'\
 --lora_eeg_ch=301
#train17
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/17-gwilliams2023-split1'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=1000 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'\
 --resume_from_checkpoint='output_models/9-gwilliams2023-split1/whisper-base/checkpoint-180000'

#train18
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/18-finetune-gwilliams2023-15-dutch' \
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-5 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200 \
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English'\
 --lora_model='output_models/15-schoffelen2019n-audio-sentences/whisper-base/checkpoint-final'\
 --lora_eeg_ch=301

#train19
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/19-gwilliams2023-split1'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-5 --fp16=True\
 --num_train_epochs=1000 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'\
 --resume_from_checkpoint='output_models/9-gwilliams2023-split1/whisper-base/checkpoint-180000'

#train23
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/23-scratch-gwilliams2023-split1'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=224 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='English' --device='cuda'\
 --random_initialize_whisper=True


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/37-gwilliams2023-split3-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'


# cable_spool_fort lw1
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/38-gwilliams2023-split3-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/cable_spool_fort/lw1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/cable_spool_fort/lw1/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

# split4
python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/39-gwilliams2023-split3-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/40-gwilliams2023-split3-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_shift100.json'\
 --local_files_only=False --language='English' --device='cuda'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/41-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_shift_noise.json'\
 --local_files_only=False --language='English' --device='cuda'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/42-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-tiny' --augment_config_path='configs/aug_shift100.json'\
 --local_files_only=False --language='English' --device='cuda'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/43-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_taylor.json'\
 --local_files_only=False --language='English' --device='cuda' --timestamps=False

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/44-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_taylor.json'\
 --local_files_only=False --language='English' --device='cuda'\
  --timestamps=False --combine_sentences=True


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/45-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-tiny' --augment_config_path='configs/aug_taylor.json'\
 --local_files_only=False --language='English' --device='cuda'\
  --timestamps=False --combine_sentences=True

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/46-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'\
  --timestamps=False --split_sentences=True


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/47-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000/full_model'\
  --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'\
  --timestamps=False --split_sentences=True

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/48-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/val.jsonl'\
 --base_model='openai/whisper-base'\
  --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'\
  --timestamps=False --split_sentences=True

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/49-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/val.jsonl'\
 --base_model='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000/full_model'\
  --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda'\
  --timestamps=False


python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/50-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda' --timestamps=False

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/51-gwilliams2023-split4'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='English' --device='cuda' --timestamps=False --split_sentences=True

# eval23
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/23-scratch-gwilliams2023-split1/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --random_initialize_whisper=True

# eval16
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/16-finetune-gwilliams2023-15-dutch/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True


# eval9
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/9-gwilliams2023-split1/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True

CUDA_LAUNCH_BLOCKING=1 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/9-gwilliams2023-split1/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --teacher_forcing=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/9-gwilliams2023-split1/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --noise=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/9-gwilliams2023-split1/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --random_choice=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/9-gwilliams2023-split1/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --random_initialize_whisper=True

# eval10
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/10-gwilliams2023-split2/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True

# eval29
CUDA_VISIBLE_DEVICES=1 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/29-gwilliams2023-split2-no_aug/whisper-base/checkpoint-48000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True

# eval31
CUDA_VISIBLE_DEVICES=1 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000/full_model'\
 --lora_model="output_models/31-gwilliams2023-split1-no_aug-pretrain28/whisper-base/checkpoint-44000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True



python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output2/whisper-base/checkpoint-116000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess2/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=128 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=False --noise=False


#eval10 subjects
python evaluation_subjects.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/10-gwilliams2023-split2/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True

#eval10 each sentence
python evaluation_each_sentence.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/10-gwilliams2023-split2/whisper-base/checkpoint-final"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True

# eval16
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/16-finetune-gwilliams2023-15-dutch/whisper-base/checkpoint-181000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=224 --batch_size=64 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True

# eval27
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --teacher_forcing=True


python evaluation_speech.py\
 --base_model='openai/whisper-base'\
 --lora_model="output_models/14-speech-gwilliams2023/whisper-base/checkpoint-1000"\
 --load_lora_model=True\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='speech' --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=False

python evaluation_speech.py\
 --base_model='openai/whisper-base'\
 --lora_model="output_models/14-speech-gwilliams2023/whisper-base/checkpoint-1000"\
 --load_lora_model=True\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='speech' --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=False --noise=True

python evaluation_speech.py\
 --base_model='openai/whisper-base'\
 --lora_model="output_models/14-speech-gwilliams2023/whisper-base/checkpoint-1000"\
 --load_lora_model=False\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='speech' --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=False --noise=False


 exec bash
 conda activate brainimagick
 cd ~/brain2text/whisper/

