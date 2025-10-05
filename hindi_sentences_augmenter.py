import pandas as pd
import random
import re
from tqdm import tqdm

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("hindi_sentences_augmented_10k.csv")

# Ensure column name is correct
if "Output Sentences" not in df.columns:
    raise ValueError("CSV must contain a column named 'Output Sentences'")

# ------------------------------
# Pick 5k unique random sentences
# ------------------------------
selected_indices = random.sample(range(len(df)), 5000)
df_subset = df.loc[selected_indices].copy()

# ------------------------------
# Helper functions for augmentation
# ------------------------------

def char_level_augment(sentence):
    if len(sentence) < 2:
        return sentence
    aug_type = random.choice(["swap", "insert", "delete"])
    chars = list(sentence)
    
    if aug_type == "delete":
        idx = random.randint(0, len(chars)-1)
        del chars[idx]

    elif aug_type == "swap" and len(chars) > 2:
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]

    elif aug_type == "insert":
        idx = random.randint(0, len(chars)-1)
        insert_char = random.choice(chars)
        chars.insert(idx, insert_char)

    return "".join(chars)


def word_level_augment(sentence):
    words = re.findall(r'\S+', sentence)
    if len(words) < 2:
        return sentence
    aug_type = random.choice(["swap", "insert", "delete"])
    
    if aug_type == "delete":
        idx = random.randint(0, len(words)-1)
        del words[idx]

    elif aug_type == "swap" and len(words) > 2:
        idx = random.randint(0, len(words)-2)
        words[idx], words[idx+1] = words[idx+1], words[idx]

    elif aug_type == "insert":
        idx = random.randint(0, len(words)-1)
        insert_word = random.choice(words)
        words.insert(idx, insert_word)

    return " ".join(words)


def introduce_error(sentence):
    # Randomly choose char-level or word-level
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
# Merge augmented sentences back (same index)
# ------------------------------
df.loc[df_subset.index, "Input Sentences"] = df_subset["Input Sentences"]

# ------------------------------
# Save in-place to the same CSV
# ------------------------------
df.to_csv("hindi_sentences_augmented_10k.csv", index=False)
print("✅ In-place augmentation complete! File updated at hindi_sentences_10k.csv")
