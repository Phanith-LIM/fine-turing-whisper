'''
!pip install datasets
'''

from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torch
from dataclasses import dataclass


# -------------------- Authentication --------------------
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hg_key = user_secrets.get_secret("hg-main")
login(token=hg_key, add_to_git_credential=False)

# -------------------- Config --------------------
@dataclass(frozen=True)
class Config:
    MODEL_NAME:            str = "PhanithLIM/whisper-medium-aug-05-june"
    PROCESSOR_NAME:        str = "PhanithLIM/whisper-medium-aug-05-june"
    TASK:                  str = "transcribe"
    LANGUAGE:              str = "Khmer"
    DATASET_NAME:          str = "PhanithLIM/asr-wmc-evaluate"
    COLUMN_NAMES:          str = MODEL_NAME.split("/")[-1]
    SPLIT:                 str = "test"
    DEVICE:                str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH:     int | None = 2024
    NUM_BEAMS:      int | None = 5
    SAMPLING_RATE:         int = 16_000

# -------------------- Load Model and Processor --------------------
dataset = load_dataset(Config.DATASET_NAME, split=Config.SPLIT)
dataset = dataset.cast_column("audio", Audio(sampling_rate=Config.SAMPLING_RATE))
print(dataset)


# -------------------- Load Model and Processor --------------------

model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
model.to(Config.DEVICE)
processor = WhisperProcessor.from_pretrained(Config.PROCESSOR_NAME, language=Config.LANGUAGE, task=Config.TASK)

# -------------------- Pipeline --------------------
def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to(Config.DEVICE), language=Config.LANGUAGE, max_length=Config.MAX_LENGTH, num_beams=Config.NUM_BEAMS)[0]
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    batch[f"{Config.COLUMN_NAMES}"] = transcription
    return batch


result = dataset.map(map_to_pred)
result.push_to_hub(Config.DATASET_NAME, split='test')
print("Results pushed to the hub successfully.")