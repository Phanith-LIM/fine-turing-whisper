'''
Forced Alignment for Multilingual Data
This notebook demonstrates how to use the `forced_alignment` function from the `align` module to perform forced alignment on multilingual data. The function takes a list of audio files and their corresponding transcripts, and aligns the audio with the text. If you manually segment transcript file.
- !pip install uroman tha
'''

import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt
import re
import os
import uroman as ur
import soundfile as sf
import numpy as np
import librosa

from typing import List, Tuple
from torchaudio.pipelines import MMS_FA as bundle
from tha.decimals import processor
from scipy.io import wavfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = bundle.get_model()
model.to(device)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

uroman = ur.Uroman()

def convert_number_to_khmer(text: str):
    return processor(text).replace('▁', '')

def normalize_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", "", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def normalize_transcript(text_path: str) -> Tuple[List[str], List[str]]:
    lines = [line.strip().replace('\u200b','') for line in open(text_path)]
    text = ''
    for line in lines:
        tsub = line
        numbers_only = re.findall(r'\d+', tsub)
        if len(numbers_only) > 0:
            for num in numbers_only:
                khmer_num = convert_number_to_khmer(num)
                tsub = re.sub(r'\b' + re.escape(num) + r'\b', khmer_num, tsub)
        t = uroman.romanize_string(tsub, lcode='khm')
        t = normalize_uroman(t).replace(' ', '')
        text += t + '\t'
    kh_texts_in_latins = text.strip().split('\t')
    kh_texts = lines
    return kh_texts_in_latins, kh_texts

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans


def save_utterance(path: str, waveform: torch.Tensor, spans: List[torch.Tensor], num_frames: int, transcript: List[str], sample_rate: int = bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    segment = waveform[:, x0:x1]
    sf.write(path, np.ravel(segment.numpy()), sample_rate)

def algin_audio_text(audio_path: str, transcript_path: str, out_dir: str):
    waveform_raw, sr = librosa.load(audio_path, sr=bundle.sample_rate)
    latin_texts, khm_texts  = normalize_transcript(transcript_path)
    tokens = tokenizer(latin_texts)
    waveform = torch.tensor(waveform_raw).unsqueeze(0)
    emission, token_spans = compute_alignments(waveform, latin_texts)
    num_frames = emission.size(1)
    base_path = os.path.basename(audio_path)[:-4]
    for idx in range(len(token_spans)):
        path = os.path.join(out_dir,'{}_{}.wav'.format(base_path,str(idx)))
        save_utterance(path, waveform, token_spans[idx], num_frames, latin_texts[idx])
        path_text = os.path.join(out_dir,'{}_{}.txt'.format(base_path,str(idx)))
        with open(path_text, 'w') as the_file:
            the_file.write(khm_texts[idx])

if __name__ == "__main__":
    audio_path = 'data/audio.wav'
    transcript_path = 'data/transcript.txt'
    out_dir = 'data/output'
    os.makedirs(out_dir, exist_ok=True)
    algin_audio_text(audio_path, transcript_path, out_dir)