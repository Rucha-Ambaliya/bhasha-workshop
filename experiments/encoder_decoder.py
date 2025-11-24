"""Run an encoder-decoder model using Huggingface (if running on Google Colab)."""
# 1. Install HuggingFace libraries
# !pip install transformers datasets

from datasets import load_dataset
from transformers import BertTokenizerFast, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# -------------------------------
# 0. Check PyTorch and GPU
# -------------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset = load_dataset(
    'csv', 
    data_files={
        'train': 'Hindi/train.csv',
        'validation': 'Hindi/dev.csv'
    }
)

# -------------------------------
# 2. Load model and tokenizer
# -------------------------------
model_name = 'google/muril-base-cased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_pretrained_model_name_or_path=model_name,
    decoder_pretrained_model_name_or_path=model_name
)

# Set special tokens
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# 3. Tokenize dataset
# -------------------------------
def tokenize(batch):
    input_enc = tokenizer(batch['Input Sentences'], max_length=64, truncation=True, padding='max_length')
    target_enc = tokenizer(batch['Output Sentences'], max_length=64, truncation=True, padding='max_length')
    input_enc['labels'] = target_enc['input_ids']
    return input_enc

tokenized = dataset.map(tokenize, batched=True)

# -------------------------------
# 4. Setup Trainer
# -------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=5,
    logging_steps=25,
    logging_first_step=True,
    per_device_train_batch_size=2,
    eval_strategy="epoch"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    tokenizer=tokenizer,
)

# -------------------------------
# 5. Train model
# -------------------------------
trainer.train()

# Save model and tokenizer
model.save_pretrained('final-encoder-decoder-model')
tokenizer.save_pretrained('final-encoder-decoder-model')

# -------------------------------
# 6. Inference
# -------------------------------
test_string = "हम सुनते हैं कि ग्लेशियर पिघलने से जलस्तर में बढ़ोतरी हुई है"
input_tokens = tokenizer(test_string, return_tensors="pt")

# Move input tensors to the same device as the model
input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

# Generate output
output_ids = model.generate(
    **input_tokens,
    decoder_start_token_id=model.config.decoder_start_token_id
)

# Decode generated ids to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated output:", generated_text)
