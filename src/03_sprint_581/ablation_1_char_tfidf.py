"""
File: ablation_1_char_tfidf.py
Author: Marco Wang
Date: 2026-04-12
Course: COLX 581 — Sprint 4 (Detailed Analysis)

Ablation 1 — Subword robustness: fastText vs. character-level TF-IDF
=====================================================================
Research question:
    The Sprint-2 TF-IDF + fastText model achieved Test Macro-F1 = 0.7566, a
    +0.1161 improvement over the TF-IDF-only Sprint-1 baseline (0.6405).
    fastText provides TWO things simultaneously:
        a) semantic geometry (word meaning, cross-lingual neighbourhood)
        b) subword coverage  (char n-grams handle OOV / misspelled tokens)

    This ablation replaces fastText with a character-level TF-IDF
    (analyzer='char_wb', n=(2,4)) to disentangle those contributions.
    If char-TF-IDF matches fastText's gains, the improvement was mostly
    about lexical coverage.  If it falls short, semantic geometry matters.

Configurations evaluated
  A. Sprint-1 baseline:  word TF-IDF (1-2 grams)  + LinearSVC
  B. Ablation target:    word TF-IDF + char TF-IDF + LinearSVC
  C. Sprint-2 (reference): word TF-IDF + fastText  + LinearSVC

Both (A) and (C) are reproduced inline for direct comparison.
"""

import json
import os
import re
import sys
import warnings

import jieba
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 581
LABEL_NAMES = ["Ham", "Phish", "Spam"]
LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

np.random.seed(SEED)


# ── I/O ────────────────────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bilingual_tokenize(text: str) -> list[str]:
    """Bilingual tokeniser: jieba for Chinese, regex for English."""
    if re.search(r"[\u4e00-\u9fff]", text):
        tokens = []
        for seg in jieba.cut(text):
            seg = seg.strip()
            if seg:
                tokens.append(seg)
        return tokens
    return re.findall(r"\b\w+\b", text.lower())


def combined_text(record: dict) -> str:
    return f"{record.get('text', '')} {record.get('text_zh', '')}".strip()


# ── Feature builders ───────────────────────────────────────────────────────────
def word_tfidf() -> TfidfVectorizer:
    """Standard word-level TF-IDF (1-2 grams), identical to Sprint-1."""
    return TfidfVectorizer(
        tokenizer=bilingual_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
    )


def char_tfidf() -> TfidfVectorizer:
    """Character-level TF-IDF (2-4 grams, char_wb mode).

    char_wb: pads token boundaries with whitespace so n-grams do not
    span across word boundaries — a natural match for the kind of
    character-level misspellings found in spam/phish emails.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_df=0.95,
        min_df=2,
        sublinear_tf=True,   # log-scale TF, standard for char n-gram models
    )


def make_clf() -> LinearSVC:
    return LinearSVC(random_state=SEED, class_weight="balanced", max_iter=2000)


# ── Eval helper ────────────────────────────────────────────────────────────────
def print_results(title: str, true_labels, pred_labels) -> dict:
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_NAMES)
    per_class = f1_score(true_labels, pred_labels, average=None, labels=LABEL_NAMES)

    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    for name, s in zip(LABEL_NAMES, per_class):
        print(f"    {name:<6}: {s:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_NAMES)
    header = "          " + "  ".join(f"{n:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    return {"accuracy": acc, "macro_f1": f1,
            "ham_f1": per_class[0], "phish_f1": per_class[1], "spam_f1": per_class[2]}


# ── Configurations ─────────────────────────────────────────────────────────────
def run_config_A(train_texts, train_labels, eval_texts, eval_labels, split_name):
    """Config A — Sprint-1 baseline: word TF-IDF only."""
    print(f"\n{'─'*64}")
    print(f"Config A: Word TF-IDF only (Sprint-1 baseline reproduced)")
    print(f"{'─'*64}")
    pipe = Pipeline([("tfidf", word_tfidf()), ("clf", make_clf())])
    pipe.fit(train_texts, train_labels)
    preds = pipe.predict(eval_texts)
    return print_results(f"Config A — {split_name}", eval_labels, preds)


def run_config_B(train_texts, train_labels, eval_texts, eval_labels, split_name):
    """Config B — Word TF-IDF + Character TF-IDF (this ablation)."""
    print(f"\n{'─'*64}")
    print(f"Config B: Word TF-IDF + Char TF-IDF (ablation — no fastText)")
    print(f"{'─'*64}")

    wtf = word_tfidf()
    ctf = char_tfidf()

    X_train_word = wtf.fit_transform(train_texts)
    X_train_char = ctf.fit_transform(train_texts)
    X_train = hstack([X_train_word, X_train_char])
    print(f"  Train feature shape: {X_train.shape}")

    clf = make_clf()
    clf.fit(X_train, train_labels)

    X_eval_word = wtf.transform(eval_texts)
    X_eval_char = ctf.transform(eval_texts)
    X_eval = hstack([X_eval_word, X_eval_char])

    preds = clf.predict(X_eval)
    return print_results(f"Config B — {split_name}", eval_labels, preds)


def run_config_C_reference(train_texts, train_labels, eval_texts, eval_labels, split_name):
    """
    Config C — Sprint-2 TF-IDF + fastText reference.
    This is a stub that prints the known results from the docs since
    loading 300d fastText vectors is optional for the ablation script.
    Set env-var FASTTEXT_EN_VEC / FASTTEXT_ZH_VEC to enable live reproduction.
    """
    en_vec_path = os.environ.get(
        "FASTTEXT_EN_VEC",
        os.path.join(PROJECT_ROOT, "models", "fasttext", "cc.en.300.vec"),
    )
    zh_vec_path = os.environ.get(
        "FASTTEXT_ZH_VEC",
        os.path.join(PROJECT_ROOT, "models", "fasttext", "cc.zh.300.vec"),
    )

    if not (os.path.isfile(en_vec_path) or os.path.isfile(zh_vec_path)):
        print(f"\n{'─'*64}")
        print("Config C: TF-IDF + fastText (Sprint-2 reference) — SKIPPED")
        print("  [Set FASTTEXT_EN_VEC / FASTTEXT_ZH_VEC to reproduce live]")
        print("  Reference result from Sprint-2 docs:")
        print("  Test Macro-F1 = 0.7566  |  Ham=0.8889  Phish=0.7143  Spam=0.6667")
        print(f"{'─'*64}")
        return None

    # Live reproduction if vectors are available
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "02_sprint_581"))
    from transfer_traditional import (
        load_fasttext_vec, build_embedding_lookup,
        mean_embed, build_tfidf_pipeline, build_feature_matrix,
        FASTTEXT_EN_URL, FASTTEXT_ZH_URL,
    )
    en_vecs = load_fasttext_vec(en_vec_path, url=FASTTEXT_EN_URL)
    zh_vecs = load_fasttext_vec(zh_vec_path, url=FASTTEXT_ZH_URL)
    lookup = build_embedding_lookup(en_vecs, zh_vecs)

    tfidf = build_tfidf_pipeline()
    X_train = build_feature_matrix(tfidf, train_texts, train_texts, lookup, fit=True)
    X_eval  = build_feature_matrix(tfidf, eval_texts, eval_texts, lookup, fit=False)

    clf = make_clf()
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_eval)
    return print_results(f"Config C (fastText live) — {split_name}", eval_labels, preds)


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(results: dict) -> None:
    print("\n" + "=" * 80)
    print("  ABLATION 1 — SUMMARY TABLE")
    print("  Research question: Is fastText's gain from semantic geometry or subword?")
    print("=" * 80)
    header = f"{'Configuration':<42} {'Acc':>6} {'MacroF1':>8} {'Ham':>6} {'Phish':>7} {'Spam':>7}"
    print(header)
    print("─" * 80)

    rows = [
        ("A: Word TF-IDF only (Sprint-1 baseline)", 0.9167, 0.9221, 1.00, 0.83, 0.80, "val (documented)"),
        ("C: Word TF-IDF + fastText (Sprint-2 ref.)", 0.8110, 0.8110, 1.00, 0.80, 0.60, "val (documented)"),
    ]
    for name, acc, f1, h, p, s, note in rows:
        print(f"{name:<42} {acc:>6.4f} {f1:>8.4f} {h:>6.4f} {p:>7.4f} {s:>7.4f}  [{note}]")

    # Print live results
    for key, r in results.items():
        if r is not None:
            cfg_label = {
                "A_val": "A: Word TF-IDF only — VAL (live)",
                "A_test": "A: Word TF-IDF only — TEST (live)",
                "B_val": "B: Word TF-IDF + Char TF-IDF — VAL (live)",
                "B_test": "B: Word TF-IDF + Char TF-IDF — TEST (live)",
            }.get(key, key)
            print(f"{cfg_label:<42} {r['accuracy']:>6.4f} {r['macro_f1']:>8.4f} "
                  f"{r['ham_f1']:>6.4f} {r['phish_f1']:>7.4f} {r['spam_f1']:>7.4f}  [live]")
    print("─" * 80)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print(" ABLATION 1 — Subword Robustness: fastText vs. Char TF-IDF")
    print("=" * 64)

    print("\nLoading data …")
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    train_texts = [combined_text(r) for r in train_records]
    val_texts   = [combined_text(r) for r in val_records]
    test_texts  = [combined_text(r) for r in test_records]
    train_labels = [r["label"] for r in train_records]
    val_labels   = [r["label"] for r in val_records]
    test_labels  = [r["label"] for r in test_records]

    print(f"  Train={len(train_records)}  Val={len(val_records)}  Test={len(test_records)}")

    results = {}

    # — Validation —
    print("\n\n>>> VALIDATION SET <<<")
    results["A_val"] = run_config_A(train_texts, train_labels, val_texts, val_labels, "Validation")
    results["B_val"] = run_config_B(train_texts, train_labels, val_texts, val_labels, "Validation")
    run_config_C_reference(train_texts, train_labels, val_texts, val_labels, "Validation")

    # — Test —
    print("\n\n>>> TEST SET <<<")
    results["A_test"] = run_config_A(train_texts, train_labels, test_texts, test_labels, "Test")
    results["B_test"] = run_config_B(train_texts, train_labels, test_texts, test_labels, "Test")
    run_config_C_reference(train_texts, train_labels, test_texts, test_labels, "Test")

    print_summary(results)


if __name__ == "__main__":
    main()
