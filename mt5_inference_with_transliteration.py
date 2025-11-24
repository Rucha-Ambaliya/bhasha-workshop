# -------------------------------
# Imports
# -------------------------------
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import subprocess, sys
sys.path.append(r'../indic-trans')
from indictrans import Transliterator

# -------------------------------
# Paths
# -------------------------------
model_path = "./mt5_trained_models/hindi/without_augmented_data/epochs_15"
test_csv_path = "../IndicGEC2025/Hindi/dev.csv"
ssf_tokenizer_script = "../Tokenizer_for_Indian_Languages/tokenize_in_SSF_format_with_sentence_tokenization.py"
output_path = "./predictions/trained_without_augmented_data/indicgec2025_dev/transliterated/hindi0.csv"

# -------------------------------
# Load MT5 model
# -------------------------------
print("Loading MT5 model...")
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"MT5 model loaded on {device}")

# -------------------------------
# Load language identification model
# -------------------------------
print("Loading ILID model...")
tokenizer_model = "google/muril-base-cased"
lid_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
lid_device = 0 if torch.cuda.is_available() else -1

lid_model = AutoModelForSequenceClassification.from_pretrained("pruthwik/ilid-muril-model")
lid_pipe = TextClassificationPipeline(model=lid_model, tokenizer=lid_tokenizer, device=lid_device)

index_to_label_dict = {
    0: 'asm', 1: 'ben', 2: 'brx', 3: 'doi', 4: 'eng', 5: 'gom',
    6: 'guj', 7: 'hin', 8: 'kan', 9: 'kas', 10: 'mai', 11: 'mal',
    12: 'mar', 13: 'mni_Beng', 14: 'mni_Mtei', 15: 'npi', 16: 'ory',
    17: 'pan', 18: 'san', 19: 'sat', 20: 'snd_Arab', 21: 'snd_Deva',
    22: 'tam', 23: 'tel', 24: 'urd'
}

# -------------------------------
# Helper: SSF tokenizer
# -------------------------------
def tokenize_sentence_ssf(sentence):
    with open("temp_input.txt", "w", encoding="utf-8") as f:
        f.write(sentence.strip() + "\n")

    subprocess.run(
        [sys.executable, ssf_tokenizer_script,
         "--input", "temp_input.txt",
         "--output", "temp_output.txt",
         "--lang", "hi"],
        check=True
    )

    tokens = []
    with open("temp_output.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("*") or line.startswith("("):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                tokens.append(parts[1])
    return tokens

# -------------------------------
# Helper: Transliterate tokens to Hindi if needed
# -------------------------------
def transliterate_to_hindi(tokens):
    transliterated_tokens = []
    for t in tokens:
        # if it's ascii (English) → transliterate
        if all(ord(c) < 128 for c in t) and t.isalpha():
            try:
                trn = Transliterator(source='eng', target='hin', build_lookup=True)
                translit = trn.transform(t)
                transliterated_tokens.append(translit)
            except Exception:
                transliterated_tokens.append(t)
        else:
            transliterated_tokens.append(t)
    return transliterated_tokens

# -------------------------------
# Load Test CSV
# -------------------------------
test_df = pd.read_csv(test_csv_path)
if "Input sentence" not in test_df.columns:
    raise ValueError("Column 'Input sentence' not found in CSV.")

print(f"Loaded {len(test_df)} samples")

# -------------------------------
# Run inference + transliteration
# -------------------------------
predictions = []

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
    input_text = "इस वाक्य को व्याकरण की दृष्टि से सही करें: " + str(row["Input sentence"])

    # MT5 inference
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)

    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Step 1: detect language at sentence level
    lid_pred = lid_pipe([pred_text])[0]['label']
    lang_code = index_to_label_dict[int(lid_pred.split('_')[1])]

    # Step 2: if not Hindi or contains English words → token-level transliteration
    if lang_code != 'hin' or any(ord(c) < 128 for c in pred_text):
        tokens = tokenize_sentence_ssf(pred_text)
        transliterated_tokens = transliterate_to_hindi(tokens)
        pred_text = " ".join(transliterated_tokens)

    predictions.append(pred_text)

# -------------------------------
# Save predictions
# -------------------------------
output_df = pd.DataFrame({
    "Input sentence": test_df["Input sentence"],
    "Output sentence": predictions
})
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Final predictions with transliteration saved to {output_path}")
print(f"Total predictions: {len(predictions)}")
