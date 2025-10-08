"""Split NER annotated data using stratification technique."""
from argparse import ArgumentParser
from random import shuffle
from math import ceil
import pandas as pd


def read_csv_file(file_path):
	"""Read a CSV file and return a DataFrame."""
	return pd.read_csv(file_path)


def stratify_based_on_labels(df, label_column):
	"""Stratify the DataFrame based on the specified label column."""
	grouped_by_labels = df.groupby(label_column)
	erroneous_sentences = []
	non_erroneous_sentences = []
	for label, group in grouped_by_labels:
		if label == 0:
			non_erroneous_sentences.extend(group.index.tolist())
		else:
			erroneous_sentences.extend(group.index.tolist())
	return erroneous_sentences, non_erroneous_sentences


def main():
	"""Pass arguments and call functions here."""
	parser = ArgumentParser()
	parser.add_argument('--input', dest='inp', help='Enter the input file in conll format.')
	parser.add_argument('--train', dest='inp', help='Enter the input file in conll format.')

	args = parser.parse_args()
	df = read_csv_file(args.inp)
	sentences = df['Input Sentences'].values
	erroneous_indexes, non_erroneous_indexes = stratify_based_on_labels(df, 'Grammatical Error')
	print(erroneous_indexes[: 10])
	print(f'Erroneous sentences: {len(erroneous_indexes)}')
	print(f'Non-erroneous sentences: {len(non_erroneous_indexes)}')
	# Shuffle the sentences to ensure randomness
	shuffle(erroneous_indexes)
	shuffle(non_erroneous_indexes)
	# Calculate split sizes
	train_size_erroneous = ceil(0.8 * len(erroneous_indexes))
	train_size_non_erroneous = ceil(0.8 * len(non_erroneous_indexes))
	# Split the sentences
	train_sentences = sentences[erroneous_indexes[: train_size_erroneous]].tolist() + sentences[non_erroneous_indexes[: train_size_non_erroneous]].tolist()
	test_sentences = sentences[erroneous_indexes[train_size_erroneous:]].tolist() + sentences[non_erroneous_indexes[train_size_non_erroneous:]].tolist()
	# Split the labels
	train_labels = [1 for i in range(train_size_erroneous)] + [0 for i in range(train_size_non_erroneous)]
	test_labels = [1 for i in range(len(erroneous_indexes) - train_size_erroneous)] + [0 for i in range(len(non_erroneous_indexes) - train_size_non_erroneous)]
	print(f'Training sentences: {len(train_sentences)}')
	print(f'Testing sentences: {len(test_sentences)}')
	print(f'Training labels: {len(train_labels)}')
	print(f'Testing sentences: {len(test_labels)}')
	train_sentences_with_labels = [[train_sentences[i], train_labels[i]] for i in range(len(train_sentences))]
	test_sentences_with_labels = [[test_sentences[i], test_labels[i]] for i in range(len(test_sentences))]
	print(train_sentences_with_labels[: 10])


if __name__ == '__main__':
	main()
