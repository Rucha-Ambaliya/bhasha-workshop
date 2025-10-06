from datasets import load_dataset
import csv

output_file = "hindi_sentences_augmented_10k.csv"
max_sentences = 10_000
max_length = 35  # or 300 depending on task, tokens not chars
min_length = 5   # optional: skip very short lines

try:
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Input Sentences", "Output Sentences"])

        dataset = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split="hin_Deva",
            streaming=True
        )

        count = 0
        for sample in dataset:
            text = sample.get("text", "").strip()
            if not text:
                continue

            # ✅ Filter by length (character-based for simplicity)
            if min_length <= len(text.split()) <= max_length:
                writer.writerow([text, text])
                count += 1

                if count % 500 == 0:
                    print(f"✅ {count} sentences written...")

                if count >= max_sentences:
                    break

    print(f"\n✅ Successfully saved {count} Hindi sentences to '{output_file}'")

except Exception as e:
    print(f"❌ An error occurred: {e}")
