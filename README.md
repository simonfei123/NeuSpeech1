# NeuSpeech: Decode Neural signal as Speech
paper at https://arxiv.org/abs/2403.01748
![img.png](img.png)

## Getting started
+ For mac and linux:
```
virtualenv pyenv --python=3.10.12
source pyenv/bin/activate
pip install -r requirements.txt
```
+ For Windows:
```
virtualenv pyenv --python=3.10.12
pyenv\Scripts\activate
pip install -r requirements.txt
```

how to run the code?
1. preprocess data with process_dataset/gwilliams2023_process.py and process_dataset/schoffelen_process.py
2. run the training with python finetune.py, cmd examples in commands/run_gwilliams.sh commands/run_schoffelen.sh
3. run evaluation with python evaluation.py

```bash
wget https://files.osf.io/v1/resources/ag3kj/providers/osfstorage/?zip=
mkdir -p datasets/gwilliams2023
unzip index.html?zip= -d datasets/gwilliams2023
rm index.html?zip=
python process_dataset/gwilliams2023_process.py
```

# contact
please do not hesitate to send me email and start collaboration with us
yyang937@connect.hkust-gz.edu.cn

Thanks yeyupiaoling for finetuning whisper pipeline https://github.com/yeyupiaoling/Whisper-Finetune
