import os
import time
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import formatters


def remove_experiment_folder(path):
    max_retries = 5
    for i in range(max_retries):
        try:
            fs.rm(path, recursive=True)
            break
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                print(f"Warning: Could not remove {path} after {max_retries} attempts.")

# Modify the ljspeech formatter
def debug_ljspeech(root_path, meta_file, **kwargs):
    """Patch for the ljspeech formatter with added debug information"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r', encoding='utf-8') as ttf:
        for idx, line in enumerate(ttf):
            cols = line.split('|')
            if len(cols) < 3:
                print(f"Error in line {idx + 1}: {line.strip()}")
                print(f"Number of columns: {len(cols)}")
                continue
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[2]
            items.append({'text': text, 'audio_file': wav_file})
    return items

# Replace the original formatter with our debug version
formatters.ljspeech = debug_ljspeech

# Function to create a unique output path
def create_unique_output_path(base_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(base_path, f"vits_ljspeech_{timestamp}")

if __name__ == '__main__':
    output_path = create_unique_output_path(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_path, exist_ok=True)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="C:/Users/prabh/OneDrive/Documents/Circle Health/VoiceFinal/tacotron2_dataset/metadata.csv", 
        path=os.path.join(output_path, "C:/Users/prabh/OneDrive/Documents/Circle Health/VoiceFinal/tacotron2_dataset")
    )
    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ljspeech",
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        eval_split_size=0.2 # Reduce this value for small datasets
    )

    # INITIALIZE THE AUDIO PROCESSOR
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size
    )

    # Print the number of samples
    print(f"Number of training samples: {len(train_samples)}")
    print(f"Number of evaluation samples: {len(eval_samples)}")

    print(f"Dataset path: {dataset_config.path}")
    print(f"Metadata file: {dataset_config.meta_file_train}")
    print(f"First few train samples:")
    for sample in train_samples[:5]:
        print(f"  Text: {sample['text'][:30]}...")
        print(f"  Audio file: {sample['audio_file']}")
        print(f"  File exists: {os.path.exists(sample['audio_file'])}")
        print()

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()