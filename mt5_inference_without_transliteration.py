# -------------------------------
# Imports
# -------------------------------
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# -------------------------------
# Paths
# -------------------------------
model_path = "./mt5_trained_models/hindi/without_augmented_data/epochs_15"
test_csv_path = "../IndicGEC2025/Hindi/test.csv"
ssf_tokenizer_script = "../Tokenizer_for_Indian_Languages/tokenize_in_SSF_format_with_sentence_tokenization.py"
output_path = "./predictions/trained_without_augmented_data/indicgec2025_test/non_transliterated/hindi.csv"

# -------------------------------
# Load MT5 model
# -------------------------------
print("Loading MT5 model...")
model = MT5ForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True
)
tokenizer = MT5Tokenizer.from_pretrained(
    model_path,
    local_files_only=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"MT5 model loaded on {device}")

# -------------------------------
# Load Test CSV
# -------------------------------
test_df = pd.read_csv(test_csv_path)
if "Input sentence" not in test_df.columns:
    raise ValueError("Column 'Input sentence' not found in CSV.")

print(f"Loaded {len(test_df)} samples")

# -------------------------------
# Run inference ONLY
# -------------------------------
predictions = []

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="MT5 Inference"):
    input_text = "इस वाक्य को व्याकरण की दृष्टि से सही करें: " + str(row["Input sentence"])

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)

    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    predictions.append(pred_text)

# -------------------------------
# Save predictions
# -------------------------------
output_df = pd.DataFrame({
    "Input sentence": test_df["Input sentence"],
    "Output sentence": predictions
})

output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Predictions saved to {output_path}")
print(f"Total predictions: {len(predictions)}")
