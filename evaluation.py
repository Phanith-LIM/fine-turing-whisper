
'''
!pip install python-Levenshtein tha datasets khmerspell
'''

import numpy as np
import re

from datasets import load_dataset, Audio
from Levenshtein import distance
from tqdm import tqdm
from IPython.display import Audio as IPAudio
from prettytable import PrettyTable
from tha.decimals import processor
from khmerspell import khnormal
from dataclasses import dataclass
from typing import List,


# -------------------- Authentication --------------------
from huggingface_hub import login
from google.colab import userdata

login(token=userdata.get('hg-main'))


# -------------------- Config --------------------
@dataclass(frozen=True)
class Config:
    DATASET_NAME:            str = 'PhanithLIM/gfleurs-evaluation'
    SPLIT:                   str = 'test'
    AUDIO_COLUMN:            str = 'audio'
    TEXT_COLUMN:             str = 'text'
    EXCLUDE_SYMBOLS:   List[str] = [
        '(', ')', '[', ']', '{', '}', '<', '>',
        '“', '”', '‘', '’', '«', '»', ',', '?',
        '「', '」', '『', '』', '▁', '-', ' ', "%", '.',
        '៖', '។', '៛', '៕', '!', '​', '–', 'ៗ', '�', ''
    ]

# -------------------- Load Dataset --------------------
dataset = load_dataset(Config.DATASET_NAME, split=Config.SPLIT)
print(dataset)

# -------------------- Preprocess Dataset --------------------
def remove_symbols(example):
    symbols_to_remove = Config.EXCLUDE_SYMBOLS
    for col, value in example.items():
        if isinstance(value, str):  # Ensure the value is a string before processing
            for symbol in symbols_to_remove:
                value = value.replace(symbol, '')
            example[col] = value
    return example

def convert_num2text(example):
    for col, value in example.items():
        if isinstance(value, str):  # Ensure the value is a string before applying regex
            value = re.sub(r'[0-9,]+(?:\.[0-9,]+)?', lambda match: processor(match.group(0)), value)
            value = re.sub(r'[០-៩,]+(?:\.[០-៩,]+)?', lambda match: processor(match.group(0)), value)
            example[col] = value.replace('▁', ' ')
    return example

def filter_english_rows(example):
    for _, value in example.items():
        if isinstance(value, str) and bool(re.search(r'[a-zA-Z]', value)):
            return False
    return True

def is_nonempty_row(example):
    for _, value in example.items():
        if isinstance(value, str) and value.strip():
            return True
    return False

def filter_long_audio(example):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    duration = len(audio_array) / sampling_rate
    return duration <= 30  # keep if less than 30 sec

# ----------------- Filter Long Audio --------------------

clean_dataset = dataset.filter(filter_long_audio)
clean_dataset = clean_dataset.map(convert_num2text)
clean_dataset = clean_dataset.map(remove_symbols)
clean_dataset = clean_dataset.filter(filter_english_rows)
clean_dataset = clean_dataset.filter(is_nonempty_row)


# ----------------- Calculate CER --------------------
pt = PrettyTable()
pt.field_names = ['model', 'cer (%)']
pt.align['model'] = 'l'
pt.align['cer (%)'] = 'l'


columns_kept = [col for col in clean_dataset.column_names if col != Config.AUDIO_COLUMN and col != Config.TEXT_COLUMN]
columns_kept.insert(0, Config.TEXT_COLUMN)

# Loop over each model column (skip the reference column)
for col in columns_kept[1:]:
    count_chars = 0
    count_errors = 0

    for example in clean_dataset:
        ref_text = khnormal(str(example[Config.TEXT_COLUMN]))
        hyp_text = khnormal(str(example[col]))
        if not ref_text.strip() or not hyp_text.strip():
            continue
        err = distance(ref_text, hyp_text)
        if err < 100:
            count_chars += len(ref_text)
            count_errors += err

    cer = (count_errors / count_chars) * 100 if count_chars > 0 else 0
    pt.add_row([col, f"{cer:.3f}"])
# Print the results table
print('Model Performance on Khmer Test Set')
print(pt)