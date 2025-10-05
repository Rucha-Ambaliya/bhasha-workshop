import pandas as pd

# ------------------------------
# File path
# ------------------------------
file_path = "hindi_sentences_augmented_10k.csv"  # your file

# ------------------------------
# Load CSV
# ------------------------------
df = pd.read_csv(file_path)

# ------------------------------
# Check columns exist
# ------------------------------
required_cols = ["Input Sentences", "Output Sentences"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV must contain column: {col}")

# ------------------------------
# Add grammatical error column
# ------------------------------
df["Grammatical Error"] = df.apply(
    lambda row: 0 if row["Input Sentences"] == row["Output Sentences"] else 1,
    axis=1
)

# ------------------------------
# Overwrite same file
# ------------------------------
df.to_csv(file_path, index=False, encoding="utf-8")
print(f"✅ Grammatical_Error column added directly to {file_path}")
