python process_dataset/music_cut_words_using_jsonl.py
python process_dataset/split_jsonl.py

python finetune.py --per_device_train_batch_size=16\
 --per_device_eval_batch_size=16 --output_dir='output_music_models/1-split1-no_aug'\
  --eval_steps=500 --save_steps=500\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=208 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/music/processed1/speed_1/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/music/processed1/speed_1/split1/val.jsonl'\
 --base_model='openai/whisper-small' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Chinese' --device='cuda'

 python evaluation.py\
 --model_path='openai/whisper-small'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_music_models/1-split1-no_aug/whisper-small/checkpoint-5000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/music/processed1/speed_1/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=16 --num_workers=8 --language='Chinese'\
 --timestamps=False --local_files_only=True


python finetune.py --per_device_train_batch_size=16\
 --per_device_eval_batch_size=16 --output_dir='output_music_models/2-words-no_aug'\
  --eval_steps=500 --save_steps=500\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=256 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/music/processed2/cut_words/split1/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/music/processed2/cut_words/split1/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Chinese' --device='cuda'

 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_music_models/2-words-no_aug/whisper-base/checkpoint-156000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/music/processed2/cut_words/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=256 --batch_size=16 --num_workers=8 --language='Chinese'\
 --timestamps=False --local_files_only=True

 python evaluation_speech.py\
 --base_model='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_music_models/2-words-no_aug/whisper-base/checkpoint-156000"\
 --load_lora_model=False\
 --test_data='/hpc2hdd/home/yyang937/datasets/music/processed2/cut_words/split1/test.jsonl'\
 --modal='speech' --batch_size=4 --num_workers=4 --language='Chinese'\
 --timestamps=False --local_files_only=False --noise=False

