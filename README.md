
# Voice Cloning

Cloning voice using TTS on a custom Indian Dataset


## Installation

Require Python 3.11

Create a virtual env

```bash
  python -m venv .venv
  .venv/Scripts/activate
```

Install TTS using this 

```bash
  git clone https://github.com/coqui-ai/TTS
  cd TTS
  pip install -e .[all,dev,notebooks]
```
    
## To Train the Model

```bash
set CUDA_VISIBLE_DEVICES=0
python train.py --restore_path <path of the model_file.pth> 
```

## To run the trained Model

```bash
tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path folder/to/save/output.wav 
```
