# ============================================================
# MT5 Hindi Grammar Correction Fine-Tuning 
# ============================================================

import pandas as pd
import torch
import gc
import nltk
nltk.download("punkt", quiet=True)

gc.collect()
torch.cuda.empty_cache()

from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

# ============================================================
# 1️ Load and clean CSVs
# ============================================================

train_df = pd.read_csv("/kaggle/input/hindi-train/hindi_train.csv")
test_df  = pd.read_csv("/kaggle/input/indicgec2025-dev/hindi_dev.csv")

def clean_dataframe(df):
    # Drop rows with missing values
    df = df.dropna(subset=["Input sentence", "Output sentence"])
    
    # Convert to string and strip extra spaces
    df["Input sentence"] = df["Input sentence"].astype(str).str.strip()
    df["Output sentence"] = df["Output sentence"].astype(str).str.strip()
    
    # Remove empty rows
    df = df[(df["Input sentence"] != "") & (df["Output sentence"] != "")]
    
    # Remove unwanted <extra_id_X> tokens if any
    df = df[~df["Output sentence"].str.contains(r"<extra_id_\d+>", regex=True)]
    
    # Remove super short sentences (optional)
    df = df[df["Input sentence"].str.len() > 1]
    df = df[df["Output sentence"].str.len() > 1]
    
    return df.reset_index(drop=True)


train_df = clean_dataframe(train_df)
test_df  = clean_dataframe(test_df)

print(f"Cleaned train samples: {len(train_df)}")
print(f"Cleaned test samples : {len(test_df)}")

# ============================================================
# 2️ DatasetDict
# ============================================================

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# ============================================================
# 3️ Download and load model/tokenizer locally
# ============================================================

# model_dir = "/kaggle/input/mt5-model-zip/kaggle/working/mt5-hindi-correction/checkpoint-4500"
model_name = "google/mt5-small"
model_dir = snapshot_download(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_dir)

max_input_length = 128
max_target_length = 128

# ============================================================
# 4️ Preprocess
# ============================================================

def preprocess_function(batch):
    # prompt in Hindi, more natural for Hindi data
    inputs = ["इस वाक्य को ग्रामर के हिसाब से सही करें: " + x for x in batch["Input sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    labels = tokenizer(
        batch["Output sentence"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )["input_ids"]

    # Mask padding tokens
    labels = [[(tok if tok != tokenizer.pad_token_id else -100) for tok in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(preprocess_function, batched=True)

# ============================================================
# 5️ Data collator
# ============================================================

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ============================================================
# 6️ Training arguments
# ============================================================

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_train_train_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    num_train_epochs=15,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=True,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True,
)

# ============================================================
# 7️ Debug callback: log 2-3 predictions each epoch
# ============================================================

class DebugPredictionCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        sample_df = test_df.sample(3, random_state=int(state.epoch))
        print(f"\nDebugging Predictions after epoch {int(state.epoch)}:")
        for idx, row in sample_df.iterrows():
            orig_input = row["Input sentence"]
            target_output = row["Output sentence"]

            tokenized_input = tokenizer(
                "इस वाक्य को ग्रामर के हिसाब से सही करें: " + orig_input,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_input_length
            ).to(model.device)

            generated_ids = model.generate(
                **tokenized_input,
                max_length=max_target_length,
                num_beams=5,
                repetition_penalty=2.0,
                length_penalty=1.2,
                early_stopping=True
            )

            decoded_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print(f"Input   : {orig_input}")
            print(f"Target  : {target_output}")
            print(f"Predict : {decoded_pred}")
            print("-" * 60)

# ============================================================
# 8️ Trainer
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[DebugPredictionCallback()],
)

torch.cuda.empty_cache()
trainer.train()

# ============================================================
# 9️ Final evaluation
# ============================================================

print("\nFinal Evaluation:")
results = trainer.evaluate()
print(results)

# ============================================================
#  Save fine-tuned model
# ============================================================

model.save_pretrained("./mt5_hindi_train")
tokenizer.save_pretrained("./mt5_hindi_train")
print("Model + tokenizer saved successfully.")

# ============================================================
# Test a prediction (Post-training)
# ============================================================

test_text = "इस वाक्य को ग्रामर के हिसाब से सही करें: ब्रावो ने कहा कि वह हिंदी फिल्में देखेंगे और कुछ हिंदी शब्द बोलेंगे।"
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=5,
    repetition_penalty=2.0,
    length_penalty=1.2,
    early_stopping=True
)
print("\nTest Prediction:", tokenizer.decode(outputs[0], skip_special_tokens=True))
