import pandas as pd

df = pd.read_csv("predictions.csv")

# Check for missing rows
print(df.isnull().sum())

# Fill missing output sentences (with input sentence, or blank, depending on competition spec)
df["Output Sentence"] = df["Output sentence"].fillna("")

# Check row count
assert len(df) == 107, f"Row count mismatch: {len(df)}"

# Save CSV
df.to_csv("predictions_fixed.csv", index=False, encoding="utf-8-sig")
