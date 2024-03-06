#eval27
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True

 #eval28
CUDA_VISIBLE_DEVICES=1  python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/28-schoffelen/whisper-base/checkpoint-14000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True


#eval29
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/29-gwilliams2023-split2-no_aug/whisper-base/checkpoint-48000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True

 # eval30
CUDA_VISIBLE_DEVICES=1  python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000/full_model'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/30-schoffelen-pretrain27/whisper-base/checkpoint-12000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True

 # eval31
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000/full_model'\
 --lora_model="output_models/31-gwilliams2023-split1-no_aug-pretrain28/whisper-base/checkpoint-44000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True


# eval32
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --extra_name="schoffelen" &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --extra_name="gwilliams"

 # eval35
 python evaluation.py\
 --model_path='openai/whisper-medium'\
 --lora_model="output_models/35-gwilliams2023-split1-no_aug/whisper-medium/checkpoint-127000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True && \
 python evaluation.py\
 --model_path='openai/whisper-small'\
 --lora_model="output_models/36-gwilliams2023-split1-no_aug/whisper-small/checkpoint-59000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True

 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/37-gwilliams2023-split3-no_aug/whisper-base/checkpoint-47000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/37-gwilliams2023-split3-no_aug/whisper-base/checkpoint-47000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --teacher_forcing=True --post_processing=True

 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/37-gwilliams2023-split3-no_aug/whisper-base/checkpoint-47000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --noise=True

# evaluation_random_choices
 python evaluation_random_choices.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/37-gwilliams2023-split3-no_aug/whisper-base/checkpoint-47000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --select_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/train.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True



 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/37-gwilliams2023-split3-no_aug/whisper-base/checkpoint-47000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True



 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/48-gwilliams2023-split4/whisper-base/checkpoint-96000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=False

 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/50-gwilliams2023-split4/whisper-base/checkpoint-39000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --add_sequence_bias=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/51-gwilliams2023-split4/whisper-base/checkpoint-90000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

TOKENIZERS_PARALLELISM=True python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/51-gwilliams2023-split4/whisper-base/checkpoint-97000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=16 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --add_sequence_bias=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/51-gwilliams2023-split4/whisper-base/checkpoint-106000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=16 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/51-gwilliams2023-split4/whisper-base/checkpoint-90000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=16 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

 python evaluation_random_choices.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --select_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

 python evaluation_random_choices.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/51-gwilliams2023-split4/whisper-base/checkpoint-90000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/test.jsonl'\
 --select_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split4/mix_train_val/train.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

#brain2text/whisper/output_models/48-gwilliams2023-split4/whisper-base/checkpoint-96000/formal_test_results_clip.json
 python evaluation_random_choices.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/48-gwilliams2023-split4/whisper-base/checkpoint-96000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/test.jsonl'\
 --select_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split3/lw1/train.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

 python evaluation_random_choices.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/48-gwilliams2023-split4/whisper-base/checkpoint-48000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --select_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/train.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True

 python evaluation_random_choices.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/48-gwilliams2023-split4/whisper-base/checkpoint-48000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --select_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/train.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=4 --num_workers=4 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True
########################################################################################################################
 # add post processing
########################################################################################################################
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/28-schoffelen/whisper-base/checkpoint-14000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --post_processing=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/29-gwilliams2023-split2-no_aug/whisper-base/checkpoint-48000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000/full_model'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/30-schoffelen-pretrain27/whisper-base/checkpoint-12000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --post_processing=True  &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000/full_model'\
 --lora_model="output_models/31-gwilliams2023-split1-no_aug-pretrain28/whisper-base/checkpoint-44000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True  &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --post_processing=True --extra_name="schoffelen"  &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/32-combine-no_aug/whisper-base/checkpoint-87000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --extra_name="gwilliams"

########################################################################################################################
 # add teacher forcing and post processing
########################################################################################################################
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --teacher_forcing=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/28-schoffelen/whisper-base/checkpoint-14000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --post_processing=True --teacher_forcing=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/29-gwilliams2023-split2-no_aug/whisper-base/checkpoint-48000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split2/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --teacher_forcing=True &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000/full_model'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/output_models/30-schoffelen-pretrain27/whisper-base/checkpoint-12000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=128 --num_workers=16 --language='Dutch'\
 --timestamps=False --local_files_only=True --post_processing=True --teacher_forcing=True  &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/brain2text/whisper/output_models/28-schoffelen/whisper-base/checkpoint-14000/full_model'\
 --lora_model="output_models/31-gwilliams2023-split1-no_aug-pretrain28/whisper-base/checkpoint-44000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=128 --num_workers=16 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=True --teacher_forcing=True

python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="output_models/27-gwilliams2023-split1-no_aug/whisper-base/checkpoint-54000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/gwilliams2023/preprocess5/split1/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=208 --batch_size=64 --num_workers=8 --language='English'\
 --timestamps=False --local_files_only=True --post_processing=False --teacher_forcing=True

########################################################################################################################
 # evaluate all ablation studies
########################################################################################################################
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/1/whisper-base/checkpoint-14000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True  &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/2/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True  --config_name='replace'  &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/transformers_models/tiny'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/3/tiny/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True  &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/transformers_models/small'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/4/small/checkpoint-25000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True  &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/transformers_models/medium'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/5/medium/checkpoint-27000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-large'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/6/whisper-large/checkpoint-33000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='/hpc2hdd/home/yyang937/transformers_models/tiny'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/7/tiny/checkpoint-8000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True

###################################masking##############################################################################
###################################masking##############################################################################
###################################masking##############################################################################
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/8/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/9/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/10/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/11/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/12/whisper-base/checkpoint-13000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/13/whisper-base/checkpoint-2000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/14/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/15/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/16/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\

######################################ft_layers#########################################################################
######################################ft_layers#########################################################################
######################################ft_layers#########################################################################
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/17/whisper-base/checkpoint-15000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/18/whisper-base/checkpoint-13000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/19/whisper-base/checkpoint-14000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/20/whisper-base/checkpoint-10000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/21/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/22/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/23/whisper-base/checkpoint-7000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/24/whisper-base/checkpoint-12000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/25/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/26/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/27/whisper-base/checkpoint-16000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/28/whisper-base/checkpoint-2000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/29/whisper-base/checkpoint-5000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/30/whisper-base/checkpoint-9000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
 python evaluation.py\
 --model_path='openai/whisper-base'\
 --lora_model="/hpc2hdd/home/yyang937/brain2text/whisper/abalation_models/31/whisper-base/checkpoint-12000"\
 --test_data='/hpc2hdd/home/yyang937/datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl'\
 --modal='eeg' --sampling_rate=200 --eeg_ch=273 --batch_size=4 --num_workers=4 --language='Dutch'\
 --timestamps=False --local_files_only=True &&\
