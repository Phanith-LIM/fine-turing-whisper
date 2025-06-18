from transformers import pipeline
import gradio as gr

pipe = pipeline(
    model='PhanithLIM/whisper-small-aug-28-april-lightning-v1',
    tokenizer='PhanithLIM/whisper-small-aug-28-april-lightning-v1',
    task='automatic-speech-recognition',
    device='cuda',
    generate_kwargs={
        "language": "Khmer",
        "task": "transcribe",
        "num_beams": 5,
    },
    return_timestamps=True,
)

def transcribe(audio):
    text = pipe(audio)['text']
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
    outputs='text',
    title="🎙️ Khmer Speech-to-Text Demo",
    description="Transcribe spoken Khmer using the fine-tuned Whisper model (PhanithLIM/whisper-small-aug-28-april-lightning-v1). Supports microphone and audio file input."
)

iface.launch(share=True)
