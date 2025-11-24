from datasets import load_dataset
import csv
import re

output_file = "data/malayalam/augmented/5k_sentences_clean.csv"
max_sentences = 5_000
max_length = 15  # in words
min_length = 5

# Regex to match only Hindi characters, spaces, and common punctuation
# hindi_regex = re.compile(r'^[\u0900-\u097F\s।?!,;:“”\'"-]+$')

# Regex to match only Bangla characters, spaces, and common punctuation
# bangla_regex = re.compile(r'^[\u0980-\u09FF\s।?!,;:“”\'"-]+$')

# Regex to match only Malayalam characters, spaces, and common punctuation
malayalam_regex = re.compile(r'^[\u0D00-\u0D7F\s।?!,;:“”\'"-]+$')

# Regex to match only Tamil characters, spaces, and common punctuation
# tamil_regex = re.compile(r'^[\u0B80-\u0BFF\s।?!,;:“”\'"-]+$')

# Regex to match only Telugu characters, spaces, and common punctuation
# telugu_regex = re.compile(r'^[\u0C00-\u0C7F\s।?!,;:“”\'"-]+$')


try:
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Input Sentences", "Output Sentences"])

        dataset = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split="mal_Mlym",
            streaming=True
        )

        count = 0
        for sample in dataset:
            text = sample.get("text", "").strip()
            if not text:
                continue

            # Filter by word length
            if not (min_length <= len(text.split()) <= max_length):
                continue

            if not malayalam_regex.match(text):
                continue

            writer.writerow([text, text])
            count += 1

            if count % 500 == 0:
                print(f"{count} sentences written...")

            if count >= max_sentences:
                break

    print(f"\nSuccessfully saved {count} Hindi sentences to '{output_file}'")

except Exception as e:
    print(f"An error occurred: {e}")
