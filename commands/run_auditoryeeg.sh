
 exec bash
 conda activate brainimagick
 cd ~/brain2text/whisper/

CUDA_LAUNCH_BLOCKING=1 NCCL_SOCKET_IFNAME=eth2 torchrun --nproc_per_node=2  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/20-auditory-eeg'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=1000 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=32 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'


 # 从MEG的pretrained model继续训练

python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/21-pretrain15-auditory-eeg'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'\
 --lora_model='output_models/15-schoffelen2019n-audio-sentences/whisper-base/checkpoint-final'\
 --lora_eeg_ch=301


python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

# train33
python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/33-auditory_eeg_decoding'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-1'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-2'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

 python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-3'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

 python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-4'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

 python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-5'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

 python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-6'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'

 python  finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/24-7'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=64 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/auditory_eeg_decoding/lbollens/preprocess1/split1/val.jsonl'\
 --base_model='openai/whisper-base'\
 --local_files_only=False --language='Dutch' --device='cuda'