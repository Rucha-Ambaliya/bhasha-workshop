# -----------------------------
# Cell 2: Hindi Classification
# -----------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  # Commented, can be used later
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import classification_report
# Load Hindi data
# csv_file = '/content/drive/MyDrive/svnit_shared_task/shared_task/bhasha-workshop/Task1_I/Data/IndicCorp_fetch_aug_label/hindi/Data_fetch_aug_label.csv'
# csv_file = 'hindi_sentences_augmented_10k.csv'
csv_file = 'hindi_indic_corp_v2_10000_with_error.csv'
df = pd.read_csv(csv_file)
texts = df["Output Sentences"]
labels = df["Grammatical Error"]
total_sentences = len(labels)
created_errors = (labels == 1).sum()
skf = StratifiedKFold(n_splits=5)
classifiers = {
    "LogisticRegression": LogisticRegression(random_state=1),
    "LinearSVC": LinearSVC(random_state=1),
    "BernoulliNB": BernoulliNB(alpha=0.001),
    "MultinomialNB": MultinomialNB(alpha=0.001),
    "RandomForest": RandomForestClassifier(random_state=1, n_estimators=1000),  # Uncomment to use Random Forest
    "GradientBoosting": GradientBoostingClassifier(random_state=1, n_estimators=1000)
}   
splits = skf.split(texts, labels)
analyzers = ["word", "char"]
ngrams = [(1, 3), (3, 6)]

best_f1 = 0
best_model_info = None

print(f"\nTotal Sentences: {total_sentences}, Errors Created: {created_errors} ({created_errors/total_sentences*100:.2f}%)")
print(f"\n📘 Language: HINDI (hin_Deva)")
print("────────────────────────────────────────")
word_tf_idf_vect = TfidfVectorizer(analyzer=analyzers[0], ngram_range=ngrams[0], token_pattern='\S+')
char_tf_idf_vect = TfidfVectorizer(analyzer=analyzers[1], ngram_range=ngrams[1])
for i, (train_index, test_index) in enumerate(splits):
    for cls_name, cls_model in classifiers.items():

        print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        train_data, train_labels = texts[train_index], labels[train_index]
        test_data, test_labels = texts[test_index], labels[test_index]
        print(len(train_data), len(train_labels))
        print(len(test_data), len(test_labels))
        train_word_tf_idf = word_tf_idf_vect.fit_transform(train_data)
        test_word_tf_idf = word_tf_idf_vect.transform(test_data)
        train_char_tf_idf = char_tf_idf_vect.fit_transform(train_data)
        test_char_tf_idf = char_tf_idf_vect.transform(test_data)
        # train_word_and_char_tf_idf = hstack([train_word_tf_idf, train_char_tf_idf])
        # test_word_and_char_tf_idf = hstack([test_word_tf_idf, test_char_tf_idf])
        train_word_and_char_tf_idf = train_char_tf_idf
        test_word_and_char_tf_idf = test_char_tf_idf
        cls_model.fit(train_char_tf_idf, train_labels)
        pred_test_labels = cls_model.predict(test_char_tf_idf)
        print('Model Predictions of ', cls_name)
        class_report = classification_report(test_labels, pred_test_labels)
        print('---Begin of Report---')
        print(class_report)
        print('---End of Report---')
