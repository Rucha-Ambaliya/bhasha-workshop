import pandas as pd
import random
import re
from tqdm import tqdm
from collections import Counter

# ------------------------------
# Load dataset (change filename as needed)
# ------------------------------
df = pd.read_csv("data/telugu/augmented/5k_sentences_clean.csv")

if "Output Sentences" not in df.columns:
    raise ValueError("CSV must contain a column named 'Output Sentences'")

# ------------------------------
# Pick 7000 unique random sentences
# ------------------------------
selected_indices = random.sample(range(len(df)), 3500)
df_subset = df.loc[selected_indices].copy()

# ------------------------------
# Counter for tracking changes
# ------------------------------
change_counter = Counter()

# ------------------------------
# Character-level augmentation
# ------------------------------
def char_level_augment(sentence):
    if len(sentence) < 2:
        return sentence

    aug_type = random.choice(["char_swap", "char_insert", "char_delete"])
    chars = list(sentence)

    if aug_type == "char_delete":
        idx = random.randint(0, len(chars) - 1)
        del chars[idx]

    elif aug_type == "char_swap" and len(chars) > 2:
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]

    elif aug_type == "char_insert":
        idx = random.randint(0, len(chars) - 1)
        chars.insert(idx, random.choice(chars))

    change_counter[aug_type] += 1
    return "".join(chars)

# ------------------------------
# Word-level augmentation
# ------------------------------
def word_level_augment(sentence):
    words = re.findall(r'\S+', sentence)
    if len(words) < 2:
        return sentence

    aug_type = random.choice(["word_swap", "word_insert", "word_delete"])

    if aug_type == "word_delete":
        del words[random.randint(0, len(words) - 1)]

    elif aug_type == "word_swap" and len(words) > 2:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx+1] = words[idx+1], words[idx]

    elif aug_type == "word_insert":
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, random.choice(words))

    change_counter[aug_type] += 1
    return " ".join(words)

# ------------------------------
# Introduce exactly one error
# ------------------------------
def introduce_error(sentence):
    if random.random() < 0.5:
        return char_level_augment(sentence)
    else:
        return word_level_augment(sentence)

# ------------------------------
# Apply augmentation
# ------------------------------
tqdm.pandas(desc="Augmenting Sentences")
df_subset["Input Sentences"] = df_subset["Output Sentences"].progress_apply(introduce_error)

# ------------------------------
# Merge augmented sentences back
# ------------------------------
df.loc[df_subset.index, "Input Sentences"] = df_subset["Input Sentences"]

# ------------------------------
# Save updated CSV
# ------------------------------
df.to_csv("data/telugu/augmented/5k_sentences.csv", index=False)

# ------------------------------
# Print statistics
# ------------------------------
print("\nIn-place augmentation complete!")
print("Augmentation Summary:")

total_changes = sum(change_counter.values())

for change, count in change_counter.items():
    percent = (count / total_changes) * 100 if total_changes else 0
    print(f"{change}: {count} sentences ({percent:.2f}%)")

print(f"\nTotal augmented sentences: {total_changes}")
print("Total possible error types: 6 (3 char + 3 word)")
