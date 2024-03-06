python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='output_models/32-combine-no_aug'\
  --eval_steps=1000 --save_steps=1000\
  --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=500 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/gwilliams_schoffelen/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams_schoffelen/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language=None --device='cuda'\
 --resume_from_checkpoint='output_models/32-combine-no_aug/whisper-base/checkpoint-87000'


python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --extra_name="schoffelen"


python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --extra_name="gwilliams"


 python evaluation_language_classification.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=64 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --extra_name="schoffelen"
