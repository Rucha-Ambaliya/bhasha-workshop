import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# -------------------------------
# Config
# -------------------------------
csv_file = "hindi_sentences_augmented_10k.csv"
model_name = "google/mt5-small"
max_length = 128
batch_size = 8
num_train_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(csv_file)

# Filter only 0 and 1 labels
df = df[df['Grammatical Error'].isin([0, 1])]

# Take 80% from each label for train
df_0 = df[df['Grammatical Error'] == 0]
df_1 = df[df['Grammatical Error'] == 1]

train_0, test_0 = train_test_split(df_0, test_size=0.2, random_state=42)
train_1, test_1 = train_test_split(df_1, test_size=0.2, random_state=42)

train_df = pd.concat([train_0, train_1]).sample(frac=1, random_state=42)
test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=42)

# -------------------------------
# Convert to HuggingFace Datasets
# -------------------------------
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = MT5Tokenizer.from_pretrained(model_name)

def preprocess(batch):
    # Encode input sentence
    inputs = tokenizer(batch["Input Sentences"], truncation=True, padding="max_length", max_length=max_length)
    # Labels: convert int to string for mT5 seq2seq
    labels = tokenizer([str(l) for l in batch["Grammatical Error"]], truncation=True, padding="max_length", max_length=2)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# -------------------------------
# Load model
# -------------------------------
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

# -------------------------------
# Data collator
# -------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -------------------------------
# Training arguments
# -------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_hindi_error",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
# Train
# -------------------------------
trainer.train()

# -------------------------------
# Evaluate and get predictions
# -------------------------------
preds = trainer.predict(test_dataset)
pred_ids = preds.predictions
decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

# Convert predictions to int
pred_labels = [int(p) if p in ["0","1"] else 0 for p in decoded_preds]
true_labels = test_df["Grammatical Error"].tolist()

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))
