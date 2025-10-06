import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# Config
# -------------------------------
input_csv = "hindi_sentences_augmented_10k.csv"
train_output = "hindi_sentences_train.csv"
test_output = "hindi_sentences_test.csv"
random_state = 42

# -------------------------------
# Load and clean dataset
# -------------------------------
df = pd.read_csv(input_csv)

# Ensure label column exists
if 'Grammatical Error' not in df.columns:
    raise ValueError("❌ 'Grammatical Error' column not found in the dataset!")

# Keep only valid labels (0, 1)
df = df[df['Grammatical Error'].isin([0, 1])]

# -------------------------------
# Balance labels
# -------------------------------
# Find minimum count between 0s and 1s
min_count = min(df['Grammatical Error'].value_counts())

# Take equal number of samples for both classes
balanced_df = pd.concat([
    df[df['Grammatical Error'] == 0].sample(n=min_count, random_state=random_state),
    df[df['Grammatical Error'] == 1].sample(n=min_count, random_state=random_state)
])

# Shuffle entire balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

# -------------------------------
# Split into train (80%) and test (20%)
# -------------------------------
train_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,
    stratify=balanced_df['Grammatical Error'],
    random_state=random_state
)

# -------------------------------
# Save results
# -------------------------------
train_df.to_csv(train_output, index=False, encoding="utf-8")
test_df.to_csv(test_output, index=False, encoding="utf-8")

print(f"✅ Total balanced samples: {len(balanced_df)}")
print(f"✅ Train set: {len(train_df)} sentences ({len(train_df)/len(balanced_df)*100:.1f}%)")
print(f"✅ Test set:  {len(test_df)} sentences ({len(test_df)/len(balanced_df)*100:.1f}%)")

print(f"✅ Files saved:\n - {train_output}\n - {test_output}")
