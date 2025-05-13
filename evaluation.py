
'''
!pip install python-Levenshtein tha datasets khmerspell
'''

import numpy as np
from datasets import load_dataset, Audio
from Levenshtein import distance
from tqdm import tqdm
from IPython.display import Audio as IPAudio
from prettytable import PrettyTable
import re
from tha.decimals import processor
from khmerspell import khnormal


# -------------------- Authentication --------------------
from huggingface_hub import login
from google.colab import userdata

login(token=userdata.get('hg-main'))

# -------------------- Load Dataset --------------------
dataset = load_dataset('PhanithLIM/gfleurs-evaluation', split='test')
print(dataset)

# -------------------- Preprocess Dataset --------------------
def remove_symbols(example):
    symbols_to_remove = [
        '(', ')', '[', ']', '{', '}', '<', '>',
        '“', '”', '‘', '’', '«', '»', ',', '?',
        '「', '」', '『', '』', '▁', '-', ' ', "%", '.',
        '៖', '។', '៛', '៕', '!', '​', '–', 'ៗ', '�', ''
    ]
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
    for col, value in example.items():
        if isinstance(value, str) and bool(re.search(r'[a-zA-Z]', value)):
            return False
    return True

def is_nonempty_row(example):
    for col, value in example.items():
        if isinstance(value, str) and value.strip():
            return True
    return False

def filter_long_audio(example):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    duration = len(audio_array) / sampling_rate
    return duration <= 30  # keep if less than 30 sec

# ----------------- Filter Long Audio --------------------

dataset2 = dataset.filter(filter_long_audio)
dataset2 = dataset2.map(convert_num2text)
dataset2 = dataset2.map(remove_symbols)
dataset2 = dataset2.filter(filter_english_rows)
dataset2 = dataset2.filter(is_nonempty_row)


# ----------------- Calculate CER --------------------
pt = PrettyTable()
pt.field_names = ['model', 'cer (%)']
pt.align['model'] = 'l'
pt.align['cer (%)'] = 'l'

ref_col = 'text'
columns_kept = [col for col in dataset2.column_names if col != 'audio' and col != ref_col]
columns_kept.insert(0, ref_col)

# Loop over each model column (skip the reference column)
for col in columns_kept[1:]:
    count_chars = 0
    count_errors = 0

    for example in dataset2:
        ref_text = khnormal(str(example[ref_col]))
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