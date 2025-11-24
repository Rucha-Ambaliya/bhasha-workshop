import pandas as pd
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from nltk.translate.gleu_score import sentence_gleu
from tqdm import tqdm
import nltk
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# 1️⃣ Load model & tokenizer
# -----------------------------
model_path = "checkpoint-21"
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# 2️⃣ Load dataset
# -----------------------------
dev_df = pd.read_csv("./Hindi/dev.csv")  
print(f"✅ Loaded {len(dev_df)} samples")

if "Input Sentences" not in dev_df.columns or "Output Sentences" not in dev_df.columns:
    raise ValueError("Columns 'Input Sentences' or 'Output Sentences' not found in CSV.")

# -----------------------------
# 3️⃣ Run inference
# -----------------------------
nltk.download('punkt', quiet=True)

predictions = []
correct_flags = []
gleu_scores = []

for i, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
    input_text = "fix grammar: " + str(row["Input Sentences"])
    target_text = str(row["Output Sentences"])

    # Tokenize and move to device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)

    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    predictions.append(pred_text)

    # Exact match check
    is_correct = pred_text == target_text
    correct_flags.append(is_correct)

    # Compute sentence-level GLEU
    reference_tokens = [nltk.word_tokenize(target_text)]
    pred_tokens = nltk.word_tokenize(pred_text)
    gleu = sentence_gleu(reference_tokens, pred_tokens)
    gleu_scores.append(gleu)

# -----------------------------
# 4️⃣ Metrics
# -----------------------------
avg_gleu = sum(gleu_scores) / len(gleu_scores)
num_correct = sum(correct_flags)
num_incorrect = len(correct_flags) - num_correct

print(f"✅ Average GLEU score: {avg_gleu:.4f}")
print(f"✅ Exact match count: {num_correct}")
print(f"✅ Non-match count: {num_incorrect}")

# -----------------------------
# 5️⃣ Save predictions CSV
# -----------------------------
output_df = pd.DataFrame({
    "Input sentence": dev_df["Input Sentences"],
    "Output sentence": dev_df["Output Sentences"],
    "Predicted sentence": predictions,
    "Correct?": correct_flags
})

output_path = "dev_predictions.csv"
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ Predictions with correctness saved to {output_path}")

# -----------------------------
# 6️⃣ Confusion Matrix (Simple Print)
# -----------------------------
# For exact matches, treat:
# True = exact match, False = not match
y_true = [True] * len(correct_flags)
y_pred = correct_flags

cm = confusion_matrix(y_true, y_pred, labels=[True, False])
print("\n✅ Confusion Matrix (Exact Matches):")
print(f"True Positives (Exact match): {cm[0,0]}")
print(f"False Negatives (Predicted wrong): {cm[0,1]}")
print(f"True Negatives: {cm[1,0]}")  # Should be 0
print(f"False Positives: {cm[1,1]}")  # Should be 0

# Optional: Classification report like f1-score summary
report = classification_report(y_true, y_pred, target_names=["Exact Match", "Not Match"])
print("\n✅ Classification Report:\n", report)
