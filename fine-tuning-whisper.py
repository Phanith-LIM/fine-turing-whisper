import warnings
warnings.filterwarnings("ignore")

import torch
import evaluate
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from huggingface_hub import login
import logging

# -------------------- Configuration --------------------
# Authentication
login(token="HF_KEY")
wandb.login(key="WandB_KEY")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("training.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Training Settings
base_model = "openai/whisper-base"
model_name = "whisper-base-aug-10-may-lightning-v1"
out_dir = f"./output/{model_name}"
language = "Khmer"
epochs = 10
batch_size = 16
learning_rate = 3e-5
gradient_accumulation_steps = 2
report_to = "wandb"
task = "transcribe"
apply_spec_augment = True
lang_model = "km"
seed = 42

# Weights & Biases settings
run = wandb.init(
    entity="KAK-AI",
    project="Experiment-Khmer-Whisper",
    name=model_name,
    config={
        "learning_rate": learning_rate,
        "model": base_model,
        "epochs": epochs,
        "lang": language,
        "batch_size": batch_size,
        "dataset": [],
        "total_num": 0,
        "total_hour": "0",
        "train_hour": "0",
        "test_hour": "0",
        "train_set": 0,
        "test_set": 0,
    }
)

# -------------------- Utilities --------------------
def calculate_total_duration(dataset):
    """Calculates the total duration of audio in a dataset."""
    total_duration_seconds = 0.
    for example in dataset:
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        duration_seconds = len(audio_array) / sampling_rate
        total_duration_seconds += duration_seconds

    hours = int(total_duration_seconds // 3600)
    minutes = int((total_duration_seconds % 3600) // 60)
    seconds = int(total_duration_seconds % 60)
    return hours, minutes, seconds

def filter_long_audio(example):
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    duration = len(audio_array) / sampling_rate
    return duration < 30

# -------------------- Load Dataset --------------------
## Download Dataset
logger.info('⬇️ Downloading openslr/openslr')
opensrl = load_dataset("openslr/openslr","SLR42", split="train", trust_remote_code=True)
opensrl = opensrl.rename_column("sentence", "text")
opensrl = opensrl.cast_column("audio", Audio(sampling_rate=16_000))
opensrl = opensrl.remove_columns([col for col in opensrl.column_names if col not in ["audio", "text"]])
hour, minute, s = calculate_total_duration(opensrl)
logger.info(f'⏳ openslr/openslr: {opensrl}')
logger.info(f'⏳ openslr/openslr: {hour}h, {minute}m, {s}s')

logger.info('⬇️ Downloading google/fleurs')
fleurs_t = load_dataset("google/fleurs","km_kh", split="train", trust_remote_code=True)
fleurs_v = load_dataset("google/fleurs","km_kh", split="validation", trust_remote_code=True)
fleurs = concatenate_datasets([fleurs_t, fleurs_v])
fleurs = fleurs.rename_column("transcription", "text")
fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16_000))
fleurs = fleurs.remove_columns([col for col in fleurs.column_names if col not in ["audio", "text"]])
hour, minute, s = calculate_total_duration(fleurs)
logger.info(f'⏳ google/fleurs: {fleurs}')
logger.info(f'⏳ google/fleurs: {hour}h, {minute}m, {s}s')


## Merge Datasets
logger.info('⬇️ Merging datasets')
dataset = concatenate_datasets([opensrl, fleurs])
logger.info(f'⏳ Merged dataset: {dataset}')
logger.info(f'⏳ Merged dataset: {calculate_total_duration(dataset)}')

## Filter Long Audio
logger.info('⬇️ Filtering long audio')
dataset = dataset.filter(filter_long_audio)
logger.info(f'⏳ Filtered dataset: {dataset}')
logger.info(f'⏳ Filtered dataset: {calculate_total_duration(dataset)}')


logger.info('⬇️ Update WandB config')
dataset_list = [
   "openslr/openslr",
   "google/fleurs"
]
run.config.update({
    'total_num': dataset.num_rows,
    'total_hour': f'{hour}h, {minute}m, {s}s',
    'dataset': dataset_list
}, allow_val_change=True)
print("Update total_num:", run.config['total_num'])
print("Update total_hour:", run.config['total_hour'])
print("Update dataset:", run.config['dataset'])


# -------------------- Split --------------------
logger.info('⬇️ Splitting dataset')
dataset = dataset.train_test_split(test_size=0.2, seed=seed)
train_set = dataset['train']
validate_set = dataset['test']
logger.info(f'⏳ Train set: {train_set}')
logger.info(f'⏳ Validate set: {validate_set}')

logger.info('⬇️ Update WandB config')
run.config.update({
    'train_set': train_set.num_rows,
    'test_set': validate_set.num_rows,
    'train_hour': f'{calculate_total_duration(train_set)}',
    'test_hour': f'{calculate_total_duration(validate_set)}'
}, allow_val_change=True)

# -------------------- Load Tokenizer and Feature Extraction --------------------
logger.info('⬇️ Loading tokenizer and feature extractor')
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)

def prepare_dataset(batch):
    batch['text'] = [text.replace('​', '') for text in batch['text']]
    audio_arrays = [audio['array'] for audio in batch['audio']]
    sampling_rate = batch['audio'][0]['sampling_rate'] 
    batch['input_features'] = feature_extractor(audio_arrays, sampling_rate=sampling_rate).input_features
    batch['labels'] = tokenizer(batch['text'], max_length=448, truncation=True).input_ids
    return batch

# -------------------- Preprocess Dataset --------------------
logger.info('⬇️ Preprocessing dataset')
train_set = train_set.map(prepare_dataset, remove_columns=train_set.column_names, num_proc=4, batched=True)
validate_set = validate_set.map(prepare_dataset, remove_columns=validate_set.column_names, num_proc=4, batched=True)

logger.info(f'⏳ Train set: {train_set}')
logger.info(f'⏳ Validate set: {validate_set}')


# -------------------- DataLoader --------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels

        return batch
    
# -------------------- Load Model --------------------
logger.info('⬇️ Loading model')
processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.generation_config.task = task
model.generation_config.forced_decoder_ids = None
model.config.apply_spec_augment = apply_spec_augment
#model.config.activation_dropout = 0.1
#model.config.dropout = 0.1


# -------------------- Compute Metrics --------------------
logger.info('⬇️ Loading metrics')
metric = evaluate.load('wer')
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {'wer': wer}


# -------------------- Training Arguments --------------------
logger.info('⬇️ Setting training arguments')
training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,  
    learning_rate=learning_rate,
    warmup_steps=1000,
    bf16=True,
    fp16=False,
    num_train_epochs=epochs,
    eval_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    predict_with_generate=True,
    generation_max_length=256,
    report_to=report_to,
    load_best_model_at_end=True,
    metric_for_best_model='wer',
    greater_is_better=False,
    dataloader_num_workers=6,
    save_total_limit=None,
    lr_scheduler_type='constant',
    seed=seed,
    data_seed=seed
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_set,
    eval_dataset=validate_set,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    ),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


# -------------------- Training --------------------
logger.info('⬇️ Training model')
trainer.train()

logger.info('⬇️ Saving model')
trainer.save_model(out_dir)

logger.info('⬇️ Saving processor')
processor.save_pretrained(out_dir)


logger.info('⬇️ Finalizing WandB')
run.finish()

logger.info('✅ Push to HuggingFace')
trainer.push_to_hub(commit_message="Upload checkpoint")