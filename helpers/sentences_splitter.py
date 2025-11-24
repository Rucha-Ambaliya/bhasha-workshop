import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# Config
# -------------------------------
input_csv = "data/tamil/augmented/5k_sentences.csv"
train_output = "data/telugu/augmented/train.csv"
test_output = "data/telugu/augmented/test.csv"
random_state = 42
test_size = 0.2  # 20% test, 80% train

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(input_csv)

# Keep only valid labels (0, 1)
df = df[df['Grammatical Error'].isin([0, 1])]

# -------------------------------
# Split by label
# -------------------------------
wrong = df[df['Grammatical Error'] == 1]  
correct = df[df['Grammatical Error'] == 0]  

# -------------------------------
# Split each label into train/test
# -------------------------------
wrong_train, wrong_test = train_test_split(
    wrong, test_size=test_size, random_state=random_state
)
correct_train, correct_test = train_test_split(
    correct, test_size=test_size, random_state=random_state
)

# -------------------------------
# Combine to get final train/test
# -------------------------------
train_df = pd.concat([wrong_train, correct_train]).sample(frac=1, random_state=random_state).reset_index(drop=True)
test_df  = pd.concat([wrong_test, correct_test]).sample(frac=1, random_state=random_state).reset_index(drop=True)

# -------------------------------
# Check ratios
# -------------------------------
def print_ratio(df, name):
    total = len(df)
    wrong_pct = 100 * (df['Grammatical Error'] == 1).sum() / total
    correct_pct = 100 * (df['Grammatical Error'] == 0).sum() / total
    print(f"{name}: {total} sentences | Wrong: {wrong_pct:.1f}% | Correct: {correct_pct:.1f}%")

print_ratio(train_df, "Train set")
print_ratio(test_df, "Test set")

# -------------------------------
# Save CSVs
# -------------------------------
train_df.to_csv(train_output, index=False, encoding="utf-8")
test_df.to_csv(test_output, index=False, encoding="utf-8")
print(f"Files saved:\n - {train_output}\n - {test_output}")
