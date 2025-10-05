from datasets import load_dataset
import csv

output_file = "hindi_sentences_augmented_10k.csv"
max_sentences = 10_000  # take 10k sentences

try:
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        # ✅ Write header with both columns
        writer.writerow(["Input Sentences", "Output Sentences"])

        # Stream the Hindi subset
        dataset = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split="hin_Deva",
            streaming=True
        )

        count = 0
        for sample in dataset:
            # Handle text depending on structure
            text = sample if isinstance(sample, str) else sample.get("text", "")
            text = text.strip()

            if text:
                # ✅ Write same sentence to both Input and Output
                writer.writerow([text, text])
                count += 1

                # Show progress every 500 sentences
                if count % 500 == 0:
                    print(f"✅ {count} sentences written...")

                # Stop after 10k sentences
                if count >= max_sentences:
                    break

    print(f"\n✅ Successfully saved {count} Hindi sentences to '{output_file}'")

except Exception as e:
    print(f"❌ An error occurred: {e}")
