# mt5_hindi_grammar_with_metrics.py

import torch
from datasets import load_dataset
import evaluate
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

# -----------------------------
# Check GPU
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
model_name = "google/mt5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# -----------------------------
# Task Prefix
# -----------------------------
TASK_PREFIX = "grammar correction: "

# -----------------------------
# Load Dataset (CSV: input,target)
# -----------------------------
dataset = load_dataset("csv", data_files={"train": "Hindi/train.csv", "test": "Hindi/dev.csv"})

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(batch):
    inputs = [TASK_PREFIX + text for text in batch["Input sentence"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    labels = tokenizer(batch["Output sentence"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    # replace padding token id's in labels by -100 to ignore in loss
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in model_inputs["labels"]
    ]
    
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# -----------------------------
# Metrics Functions
# -----------------------------
def exact_match(preds, labels):
    correct = 0
    for p, l in zip(preds, labels):
        if p.strip() == l.strip():
            correct += 1
    return correct / len(preds)

def char_accuracy(preds, labels):
    total_chars = 0
    correct_chars = 0
    for p, l in zip(preds, labels):
        for pc, lc in zip(p, l):
            total_chars += 1
            if pc == lc:
                correct_chars += 1
        # remaining unmatched characters
        total_chars += abs(len(p) - len(l))
    return correct_chars / total_chars

bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Exact match
    exact = exact_match(decoded_preds, decoded_labels)
    # Character-level accuracy
    char_acc = char_accuracy(decoded_preds, decoded_labels)
    
    # BLEU metric
    references = [[l.split()] for l in decoded_labels]
    preds_tokens = [p.split() for p in decoded_preds]
    bleu_score = bleu.compute(predictions=preds_tokens, references=references)["bleu"]

    return {"exact_match": exact, "char_accuracy": char_acc, "bleu": bleu_score}

# -----------------------------
# Training Arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-hindi-grammar",
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,             # mixed precision for GPU
    logging_dir='./logs',
    logging_steps=100,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Evaluate on Test Set
# -----------------------------
metrics = trainer.evaluate()
print("\n=== Evaluation Metrics on Test Set ===")
print("Exact Match Accuracy :", metrics["eval_exact_match"])
print("Character-level Accuracy :", metrics["eval_char_accuracy"])
print("BLEU Score :", metrics["eval_bleu"])

# -----------------------------
# Test Example
# -----------------------------
def correct_grammar(sentence):
    inputs = tokenizer(TASK_PREFIX + sentence, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_sentence = "वह घर जा रही हूँ"
print("\nInput Sentence :", test_sentence)
print("Corrected Sentence :", correct_grammar(test_sentence))
