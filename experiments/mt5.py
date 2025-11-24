import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import torch

# -------------------------------
# Load CSVs
# -------------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Rename columns to match expected keys
train_df = train_df.rename(columns={"Input Sentences": "input_text", "Output Sentences": "target_text"})
test_df = test_df.rename(columns={"Input Sentences": "input_text", "Output Sentences": "target_text"})

# Convert to HuggingFace Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# -------------------------------
# Load Tokenizer & Model
# -------------------------------
model_name = "google/mt5-small"   # Try mt5-base for better accuracy
tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_function(examples):
    # Encode inputs
    model_inputs = tokenizer(
        examples["input_text"], max_length=128, truncation=True
    )
    # Encode targets with text_target
    labels = tokenizer(
        text_target=examples["target_text"], max_length=128, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# -------------------------------
# Data Collator & Metrics
# -------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert logits to token IDs if needed
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 with pad_token_id before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu expects list of list for references
    decoded_labels = [[l] for l in decoded_labels]

    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# -------------------------------
# Training Arguments
# -------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-corrector",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -------------------------------
# Train
# -------------------------------
trainer.train()

# -------------------------------
# Test Inference
# -------------------------------
test_sentence = "कहते है 'शिक्षा शेरनी को वो दुध है जिसने जितना पिया उतना ही दहाडा है'।"
inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True).to(model.device)
outputs = model.generate(**inputs, max_length=128)
print("Input: ", test_sentence)
print("Prediction: ", tokenizer.decode(outputs[0], skip_special_tokens=True))