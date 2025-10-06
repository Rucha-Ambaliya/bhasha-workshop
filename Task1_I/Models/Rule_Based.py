# Language_Rule-Based_Model.py
# Purpose: Run SentiWordNet sentiment analysis on all sentences in language TSVs (batch-wise)

import os
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Base directory containing language folders
base_dir = r"D:\svnit\shared task\IndicGEC2025"

# List of languages to process
languages = ["Hindi", "Telugu", "Bangla", "Malayalam", "Tamil"]

# Function: convert PENN POS to WordNet POS
def get_wordnet_pos(word):
    pos_tagged = pos_tag([word])[0][1]  # get POS tag
    pos_letter = pos_tagged[0].upper()  # first letter of POS
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}  # mapping
    return tag_dict.get(pos_letter, wn.NOUN)  # default noun

# Function: get sentiment score of a sentence
def get_sentiment_score(sentence, use_common=True):
    tokens = word_tokenize(str(sentence))  # tokenize sentence, ensure string
    lemmatizer = WordNetLemmatizer()
    sentiment_score = 0.0

    for token in tokens:  # iterate over tokens
        pos_tag_word = get_wordnet_pos(token)
        lemma_word = lemmatizer.lemmatize(token, pos_tag_word)
        if not lemma_word:
            continue

        synsets = wn.synsets(lemma_word, pos=pos_tag_word)
        if len(synsets) == 0:
            continue

        if use_common:
            swn_synset = swn.senti_synset(synsets[0].name())
            sentiment_score += swn_synset.pos_score() - swn_synset.neg_score()
        else:
            synset_scores = [swn.senti_synset(s.name()).pos_score() - swn.senti_synset(s.name()).neg_score() for s in synsets]
            sentiment_score += max(synset_scores)

    return sentiment_score

# Process each language TSV
for language in languages:
    language_path = os.path.join(base_dir, language)
    input_file = os.path.join(language_path, f"{language.lower()}_labels.tsv")

    if not os.path.exists(input_file):
        print(f"TSV file not found for {language}. Skipping.")
        continue

    df_sentences = pd.read_csv(input_file, sep="\t", encoding="utf-8")
    total_sentences = len(df_sentences)
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    # Calculate sentiment scores (batch-wise)
    for sentence_text in df_sentences['sentence']:
        score = get_sentiment_score(sentence_text, use_common=False)
        if score > 0:
            positive_count += 1
        elif score == 0:
            neutral_count += 1
        else:
            negative_count += 1

    # Print summary for the language
    print(f"Language: {language}")
    print(f"Total sentences: {total_sentences}")
    print(f"Positive: {positive_count}, Neutral: {neutral_count}, Negative: {negative_count}")
    print(f"Finished processing {language}\n")
