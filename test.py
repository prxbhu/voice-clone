import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
textmsg = 'Hi, i wanted to remind you about your appointment booking today with Doctor KVLN Sharma'
wav = tts.tts(text=textmsg, speaker_wav="wavs\sample1.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text=textmsg, speaker_wav="wavs\sample1.wav", language="en", file_path="output.wav")