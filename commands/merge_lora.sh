python merge_lora.py \
--lora_model='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000'\
 --model_path='openai/whisper-base' --eeg_ch=208

 python merge_lora.py \
--lora_model='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000'\
 --model_path='openai/whisper-base' --eeg_ch=273