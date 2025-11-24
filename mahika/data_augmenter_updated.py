from argparse import ArgumentParser
import pandas as pd
import random
import re
from tqdm import tqdm


def read_lines_from_file(file_path):
    """Read lines from file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


# ------------------------------
# Helper functions for augmentation
# ------------------------------

def char_level_augment(sentence):
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



def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter input file path')
    parser.add_argument('--output', dest='out', help='Enter the output csv path')
    args = parser.parse_args()
    input_sentences = read_lines_from_file(args.inp)
    all_indexes = list(range(len(input_sentences)))
    half_size = int(0.5 * len(all_indexes))
    selected_indexes = random.sample(all_indexes, half_size)
    output_list = []
    columns = ['Input Sentences', 'Output Sentences', 'Grammatical Error']
    for index in all_indexes:
        input_sentence = input_sentences[index]
        if index in selected_indexes:
            output_sentence = introduce_error(input_sentence)
            label = 1
        else:
            output_sentence = input_sentence
            label = 0
        output_list.append([input_sentence, output_sentence, label])
    df = pd.DataFrame(output_list, columns=columns)
    df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
