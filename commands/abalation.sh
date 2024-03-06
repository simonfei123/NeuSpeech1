CUDA_VISIBLE_DEVICES=0 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/1'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/2'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --config_name='replace'

 CUDA_VISIBLE_DEVICES=1 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/3'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='/hpc2hdd/home/yyang937/transformers_models/tiny' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

python finetune.py --per_device_train_batch_size=16\
 --per_device_eval_batch_size=16 --output_dir='abalation_models/4'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='/hpc2hdd/home/yyang937/transformers_models/small' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

python finetune.py --per_device_train_batch_size=8\
 --per_device_eval_batch_size=8 --output_dir='abalation_models/5'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='/hpc2hdd/home/yyang937/transformers_models/medium' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'\
 --resume_from_checkpoint='/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/5/medium/checkpoint-27000'

python finetune.py --per_device_train_batch_size=4\
 --per_device_eval_batch_size=4 --output_dir='abalation_models/6'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=4 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-large' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

 python finetune.py --per_device_train_batch_size=128\
 --per_device_eval_batch_size=128 --output_dir='abalation_models/7'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='/hpc2hdd/home/yyang937/transformers_models/tiny' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

 ####################################mask###############################################################################
 ####################################mask###############################################################################
 ####################################mask###############################################################################
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/8'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_b25.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/9'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_b50.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/10'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_b75.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

  python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/11'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_t25.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/12'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_t50.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/13'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_t75.json'\
 --local_files_only=False --language='Dutch' --device='cuda'

  python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/14'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_c25.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/15'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_c50.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/16'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_mask_c75.json'\
 --local_files_only=False --language='Dutch' --device='cuda'


#############################noise######################################################################################
#############################noise######################################################################################
#############################noise######################################################################################
CUDA_VISIBLE_DEVICES=0 && python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/17'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_noise_snr0_p50.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/18'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_noise_snr15_p50.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/19'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_noise_snr0_p100.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/20'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_noise_snr15_p100.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/21'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_shift50.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/22'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/aug_shift100.json'\
 --local_files_only=False --language='Dutch' --device='cuda' &&\


####################################ft##################################################################################
####################################ft##################################################################################
####################################ft##################################################################################
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/23'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --fine_tune_layers=1  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/24'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --fine_tune_layers=2  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/25'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --fine_tune_layers=3  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/26'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --fine_tune_layers=4  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/27'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --fine_tune_layers=5  &&\

####################################data ratio##########################################################################
####################################data ratio##########################################################################
####################################data ratio##########################################################################

 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/28'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --data_ratio=0.2  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/29'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --data_ratio=0.4  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/30'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --data_ratio=0.6  &&\
 python finetune.py --per_device_train_batch_size=64\
 --per_device_eval_batch_size=64 --output_dir='abalation_models/31'\
 --eval_steps=1000 --save_steps=1000 --learning_rate=1e-3 --fp16=True\
 --num_train_epochs=120 --warmup_steps=500 --max_audio_len=30\
 --use_8bit=False --num_workers=16 --modal='eeg' --eeg_ch=273 --sampling_rate=200 --orig_sample_rate=200\
 --train_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl'\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl'\
 --base_model='openai/whisper-base' --augment_config_path='configs/augmentation1.json'\
 --local_files_only=False --language='Dutch' --device='cuda' --data_ratio=0.8  &&\
