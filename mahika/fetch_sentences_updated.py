"""Fetch sentences from the AI4Bharat IndicCorpV2 Hindi dataset and save them to a CSV file."""
from argparse import ArgumentParser
from datasets import load_dataset
import csv
from re import sub
from re import split as re_split
from random import choice


output_file = "hindi_sentences_augmented_10k.txt"
max_sentences = 10_000
max_length = 35  # or 300 depending on task, tokens not chars
min_length = 5   # optional: skip very short lines
selected_sentences = []
end_sentence_markers = ['।', '.', '!', '?', '॥']
# text = "डायबिटीज में किशमिश खाने के बहुत फायदे होते हैं। इसे खाने से ब्लड शुगर लेवल नियंत्रण में रहता है।"


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines) + '\n')


def tokenize_text_into_sentences(text):
    # Normalize spaces
    text = sub(r'\s+', ' ', text).strip()
    sentences = []
    for sentence in re_split(r'([।.!?॥])', text):
        sentence = sentence.strip()
        if sentence:
            # Ensure the sentence ends with an end marker
            if sentence[-1] not in end_sentence_markers:
                sentence += ' ।'  # Default to Hindi full stop if no marker present
            else:
                sentence += ' ' + sentence[-1]  # Append the marker
            sentence = sub(r'\s+', ' ', sentence).strip()
            sentences.append(sentence)
    return sentences


def fetch_indiccorp_sentences(language_code="hin_Deva"):
    """Fetch sentences from the IndicCorpV2 dataset."""
    selected_sentences = []
    try:
        dataset = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split=language_code,
            streaming=True
        )

        count = 0
        for sample in dataset:
            text = sample.get("text", "").strip()
            if not text:
                continue
            sentences = tokenize_text_into_sentences(text)
            random_sentence = choice(sentences)
            tries = 0
            if len(random_sentence.split()) < min_length:
                    continue
            else:
                count += 1
            if count <= max_sentences:
                selected_sentences.append(random_sentence)
            else:
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    return selected_sentences


def main():
    """Main function to fetch sentences and write to CSV."""
    parser = ArgumentParser(description="Fetch sentences from IndicCorpV2 dataset")
    parser.add_argument(
        "--lang",
        dest='lang',
        type=str,
        default="hin_Deva",
        help="Language code for the IndicCorpV2 dataset (default: hin_Deva for Hindi)"
    )
    parser.add_argument('--output', dest='out', type=str, help="Enter the output file path where the selected sentences will be written to")
    args = parser.parse_args()
    selected_sentences = fetch_indiccorp_sentences(args.lang)
    write_lines_to_file(selected_sentences, args.out)


if __name__ == '__main__':
    main()
