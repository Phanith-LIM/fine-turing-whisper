"""
Forced Alignment for Multilingual Data

This script performs forced alignment between audio and corresponding multilingual transcripts,
using the torchaudio MMS forced alignment pipeline. It supports Khmer numeral conversion and
romanization via `tha` and `uroman`.
"""

import os
import re
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from typing import List, Tuple
from torchaudio.pipelines import MMS_FA as bundle
from tha.decimals import processor
import uroman as ur

# Initialize device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = bundle.get_model().to(device)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
uroman = ur.Uroman()


def convert_number_to_khmer(text: str) -> str:
    """Convert Arabic numerals in a string to Khmer words."""
    return processor(text).replace('▁', '')


def normalize_uroman(text: str) -> str:
    """Normalize romanized text by lowercasing and removing non-alphabetic characters."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]", "", text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_transcript(text_path: str) -> Tuple[List[str], List[str]]:
    """
    Normalize transcript by:
    - Converting Arabic numerals to Khmer words
    - Romanizing Khmer text using uroman
    - Removing non-latin characters
    Returns both romanized texts and original Khmer lines.
    """
    with open(text_path, encoding='utf-8') as f:
        lines = [line.strip().replace('\u200b', '') for line in f]

    romanized_lines = []
    for line in lines:
        for num in re.findall(r'\d+', line):
            khmer_num = convert_number_to_khmer(num)
            line = re.sub(r'\b{}\b'.format(re.escape(num)), khmer_num, line)

        romanized = uroman.romanize_string(line, lcode='khm')
        romanized = normalize_uroman(romanized).replace(' ', '')
        romanized_lines.append(romanized)

    return romanized_lines, lines


def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    """
    Compute token-level alignments between waveform and transcript using the MMS aligner.
    Returns emissions and token spans.
    """
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans


def save_utterance(
    output_path: str,
    waveform: torch.Tensor,
    spans: List[torch.Tensor],
    num_frames: int,
    sample_rate: int = bundle.sample_rate
):
    """
    Save an audio segment corresponding to the span of aligned tokens.
    """
    ratio = waveform.size(1) / num_frames
    start = int(ratio * spans[0].start)
    end = int(ratio * spans[-1].end)
    segment = waveform[:, start:end]
    sf.write(output_path, np.ravel(segment.numpy()), sample_rate)


def align_audio_to_text(audio_path: str, transcript_path: str, output_dir: str):
    """
    Align audio with text and save aligned segments and corresponding transcripts.
    """
    print(f"Processing: {audio_path} with transcript {transcript_path}")

    waveform_np, _ = librosa.load(audio_path, sr=bundle.sample_rate)
    waveform = torch.tensor(waveform_np).unsqueeze(0)

    latin_texts, khmer_texts = normalize_transcript(transcript_path)
    emission, token_spans = compute_alignments(waveform, latin_texts)
    num_frames = emission.size(1)
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    for idx, (span, original_text) in enumerate(zip(token_spans, khmer_texts)):
        audio_filename = f"{base_filename}_{idx}.wav"
        text_filename = f"{base_filename}_{idx}.txt"

        audio_out_path = os.path.join(output_dir, audio_filename)
        text_out_path = os.path.join(output_dir, text_filename)

        save_utterance(audio_out_path, waveform, span, num_frames)

        with open(text_out_path, 'w', encoding='utf-8') as f:
            f.write(original_text)

    print(f"✅ Finished alignment for {audio_path}")


if __name__ == "__main__":
    AUDIO_PATH = 'data/audio.wav'
    TRANSCRIPT_PATH = 'data/transcript.txt'
    OUTPUT_DIR = 'data/output'

    align_audio_to_text(AUDIO_PATH, TRANSCRIPT_PATH, OUTPUT_DIR)
