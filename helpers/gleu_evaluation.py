# ------------------------------------------------------------
# Custom Evaluation Script
# Adapted for Personal File Set
# Reference: BhashaWorkshop IndicGEC2025 Framework
# ------------------------------------------------------------

import os
import csv
import json
import re
from collections import Counter

# ===============================
# ✅ SET YOUR PATHS HERE
# ===============================

predicted_path = "./predictions/trained_without_augmented_data/indicgec2025_dev/transliterated/hindi0.csv"
actual_path = "./../IndicGEC2025/Hindi/dev.csv"
output_dir = "gleu_results"

REQ_COL_OUT = "Output sentence"


# Utility functions

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def read_csv_rows_output_only(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if REQ_COL_OUT not in cols:
            raise ValueError(f"{os.path.basename(path)} must have column '{REQ_COL_OUT}'. Found: {cols}")
        return list(reader)


def corpus_gleu_1to4(ref_tokens_list, hyp_tokens_list):
    scores = []
    for n in range(1, 5):
        ref_total = hyp_total = matches = 0

        for rt, ht in zip(ref_tokens_list, hyp_tokens_list):
            r = Counter(tuple(rt[i:i+n]) for i in range(max(0, len(rt)-n+1)))
            h = Counter(tuple(ht[i:i+n]) for i in range(max(0, len(ht)-n+1)))

            ref_total += sum(r.values())
            hyp_total += sum(h.values())

            if r and h:
                matches += sum((r & h).values())

        prec = (matches / hyp_total) if hyp_total > 0 else 0.0
        rec  = (matches / ref_total) if ref_total > 0 else 0.0
        scores.append(min(prec, rec))

    return (sum(scores) / len(scores)) * 100.0 if scores else 0.0


# MAIN LOGIC

os.makedirs(output_dir, exist_ok=True)

if not os.path.isfile(predicted_path):
    raise FileNotFoundError(f"Predicted file not found: {predicted_path}")

if not os.path.isfile(actual_path):
    raise FileNotFoundError(f"Actual file not found: {actual_path}")

gold = read_csv_rows_output_only(actual_path)
pred = read_csv_rows_output_only(predicted_path)

if len(gold) != len(pred):
    raise ValueError(f"Row mismatch: gold={len(gold)}, pred={len(pred)}")

ref_tokens_list, hyp_tokens_list = [], []

for g, p in zip(gold, pred):
    ref_tokens_list.append(tokenize((g.get(REQ_COL_OUT, '') or '').strip()))
    hyp_tokens_list.append(tokenize((p.get(REQ_COL_OUT, '') or '').strip()))

gleu_score = corpus_gleu_1to4(ref_tokens_list, hyp_tokens_list)

# OUTPUT

print("GLEU Score (%):", round(gleu_score, 2))

with open(os.path.join(output_dir, "scores.txt"), "w", encoding="utf-8") as f:
    f.write(f"gleu: {gleu_score:.6f}\n")

with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
    json.dump({"gleu": gleu_score}, f, ensure_ascii=False)
