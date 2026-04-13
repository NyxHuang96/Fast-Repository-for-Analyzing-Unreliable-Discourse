"""
File: ablation_4_bilingual_fasttext.py
Author: Darwin Zhang
Date: 2026-04-12
Course: COLX 581 — Sprint 4 (Detailed Analysis)

Ablation 4 — Bilingual fastText: Do Chinese vectors add signal?
===============================================================
Research question:
    Sprint-2 loaded BOTH cc.en.300.vec (English) AND cc.zh.300.vec
    (Chinese) fastText Common Crawl vectors and merged them into one
    lookup table. Our dataset is bilingual: each email has an English
    body (`text`) and a Chinese translation (`text_zh`).

    Did the Chinese vectors meaningfully contribute, or was the gain
    driven entirely by the English fastText embeddings?

Configurations evaluated
  A. TF-IDF + EN fastText only  (300-d)
  B. TF-IDF + ZH fastText only  (300-d)
  C. TF-IDF + EN + ZH fastText  (300-d, Sprint-2 configuration)
  D. TF-IDF only               (Sprint-1 baseline, no embeddings)

All four share the same TF-IDF features, SVC hyperparameters, and seed.
fastText vectors are optionally auto-downloaded if not already present.
"""

import gzip
import json
import os
import re
import sys
import urllib.request
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
EMBEDDING_DIM = 300
MAX_VOCAB_LOAD = 200_000

LABEL_NAMES = ["Ham", "Phish", "Spam"]
LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

FASTTEXT_EN_VEC = os.environ.get(
    "FASTTEXT_EN_VEC",
    os.path.join(PROJECT_ROOT, "models", "fasttext", "cc.en.300.vec"),
)
FASTTEXT_ZH_VEC = os.environ.get(
    "FASTTEXT_ZH_VEC",
    os.path.join(PROJECT_ROOT, "models", "fasttext", "cc.zh.300.vec"),
)
FASTTEXT_EN_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
FASTTEXT_ZH_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"

np.random.seed(SEED)


# ── I/O helpers ────────────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def combined_text(record: dict) -> str:
    return f"{record.get('text', '')} {record.get('text_zh', '')}".strip()


# ── Bilingual tokeniser ────────────────────────────────────────────────────────
def bilingual_tokenize(text: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", text):
        tokens = []
        for seg in jieba.cut(text):
            seg = seg.strip()
            if seg:
                tokens.append(seg)
        return tokens
    return re.findall(r"\b\w+\b", text.lower())


# ── fastText loader ────────────────────────────────────────────────────────────
def _maybe_download_vec(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gz_path = dest_path + ".gz"
    print(f"  Downloading {url} …")
    urllib.request.urlretrieve(url, gz_path)
    print(f"  Decompressing → {dest_path} …")
    with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(gz_path)


def load_fasttext_vec(
    vec_path: str,
    max_words: int = MAX_VOCAB_LOAD,
    url: str | None = None,
) -> dict[str, np.ndarray] | None:
    if not os.path.isfile(vec_path):
        if url is None:
            print(f"  [WARN] fastText file not found: {vec_path}  (skipping)")
            return None
        try:
            _maybe_download_vec(url, vec_path)
        except Exception as exc:
            print(f"  [WARN] Auto-download failed ({exc}). Skipping.")
            return None

    print(f"  Loading {vec_path} (up to {max_words:,} words) …")
    embeddings: dict[str, np.ndarray] = {}
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
        vocab_total, dim = int(header[0]), int(header[1])
        print(f"    Vocab: {vocab_total:,}  |  Dim: {dim}")
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == dim:
                    embeddings[word] = vec
            except ValueError:
                continue
    print(f"    Loaded {len(embeddings):,} word vectors.")
    return embeddings


# ── Feature builders ───────────────────────────────────────────────────────────
def mean_embed(texts: list[str], lookup: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = bilingual_tokenize(text)
        vecs = [lookup[t] for t in tokens if t in lookup]
        if vecs:
            matrix[i] = np.mean(vecs, axis=0)
    return matrix


def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=bilingual_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
    )


def make_clf() -> LinearSVC:
    return LinearSVC(random_state=SEED, class_weight="balanced", max_iter=2000)


# ── Train / evaluate one configuration ────────────────────────────────────────
def run_config(
    config_name: str,
    lookup: dict[str, np.ndarray] | None,
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
) -> dict:
    print(f"\n{'='*64}")
    print(f"  Config: {config_name}")
    print(f"{'='*64}")

    tfidf = build_tfidf()

    if lookup is not None:
        # TF-IDF + embedding concatenation
        X_tr_tfidf = tfidf.fit_transform(train_texts)
        X_tr_emb   = csr_matrix(mean_embed(train_texts, lookup))
        X_train    = hstack([X_tr_tfidf, X_tr_emb])

        X_val_tfidf = tfidf.transform(val_texts)
        X_val_emb   = csr_matrix(mean_embed(val_texts, lookup))
        X_val       = hstack([X_val_tfidf, X_val_emb])

        X_te_tfidf = tfidf.transform(test_texts)
        X_te_emb   = csr_matrix(mean_embed(test_texts, lookup))
        X_test     = hstack([X_te_tfidf, X_te_emb])
    else:
        # TF-IDF only
        X_train = tfidf.fit_transform(train_texts)
        X_val   = tfidf.transform(val_texts)
        X_test  = tfidf.transform(test_texts)

    print(f"  Feature shape: {X_train.shape}")

    clf = make_clf()
    clf.fit(X_train, train_labels)

    val_preds  = clf.predict(X_val)
    test_preds = clf.predict(X_test)

    print("\n  --- Validation ---")
    val_result  = _print_eval("Validation", val_labels, val_preds)
    print("\n  --- Test ---")
    test_result = _print_eval("Test", test_labels, test_preds)
    return {"val": val_result, "test": test_result}


def _print_eval(split: str, true_labels, pred_labels) -> dict:
    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_NAMES)
    per_class = f1_score(true_labels, pred_labels, average=None, labels=LABEL_NAMES)

    print(f"  {split}  Accuracy={acc:.4f}  Macro-F1={f1:.4f}  "
          f"Ham={per_class[0]:.4f}  Phish={per_class[1]:.4f}  Spam={per_class[2]:.4f}")
    print(classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_NAMES)
    header = "          " + "  ".join(f"{n:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    return {
        "accuracy": acc, "macro_f1": f1,
        "ham_f1": per_class[0], "phish_f1": per_class[1], "spam_f1": per_class[2],
    }


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(all_configs: list[tuple]) -> None:
    print("\n\n" + "=" * 84)
    print("  ABLATION 4 — BILINGUAL FASTTEXT SUMMARY TABLE")
    print("  Research question: Do Chinese fastText vectors add signal?")
    print("=" * 84)

    for split in ("val", "test"):
        print(f"\n  --- {split.upper()} SET ---")
        header = (f"{'Configuration':<38} {'Acc':>6} {'MacroF1':>8} "
                  f"{'Ham':>6} {'Phish':>7} {'Spam':>7}")
        print(header)
        print("─" * 76)
        for name, results in all_configs:
            if results is None:
                print(f"{name:<38}  (SKIPPED — fastText not available)")
                continue
            r = results[split]
            print(f"{name:<38} {r['accuracy']:>6.4f} {r['macro_f1']:>8.4f} "
                  f"{r['ham_f1']:>6.4f} {r['phish_f1']:>7.4f} {r['spam_f1']:>7.4f}")
        print("─" * 76)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print(" ABLATION 4 — Bilingual fastText Contribution")
    print(" EN-only vs ZH-only vs EN+ZH vs TF-IDF-only")
    print("=" * 64)

    print("\nLoading data …")
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    train_texts  = [combined_text(r) for r in train_records]
    val_texts    = [combined_text(r) for r in val_records]
    test_texts   = [combined_text(r) for r in test_records]
    train_labels = [r["label"] for r in train_records]
    val_labels   = [r["label"] for r in val_records]
    test_labels  = [r["label"] for r in test_records]

    print(f"  Train={len(train_records)}  Val={len(val_records)}  Test={len(test_records)}")

    # ── Load vectors ─────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("Loading fastText vectors …")
    en_vecs = load_fasttext_vec(FASTTEXT_EN_VEC, url=FASTTEXT_EN_URL)
    zh_vecs = load_fasttext_vec(FASTTEXT_ZH_VEC, url=FASTTEXT_ZH_URL)

    if en_vecs is None and zh_vecs is None:
        print("\n[ERROR] No fastText vectors available — cannot run ablation.")
        print("  Set FASTTEXT_EN_VEC / FASTTEXT_ZH_VEC to point at .vec files,")
        print("  or allow auto-download (requires internet connection).")
        sys.exit(1)

    # Merged lookup for EN+ZH
    en_zh_lookup: dict[str, np.ndarray] = {}
    if en_vecs:
        en_zh_lookup.update(en_vecs)
    if zh_vecs:
        en_zh_lookup.update(zh_vecs)
    print(f"  Combined lookup size: {len(en_zh_lookup):,}")

    # ── Run all configs ───────────────────────────────────────────────────────
    all_configs = []

    # Config D: TF-IDF only (Sprint-1 baseline)
    r_D = run_config(
        "D: TF-IDF only (Sprint-1 baseline)",
        lookup=None,
        train_texts=train_texts, train_labels=train_labels,
        val_texts=val_texts, val_labels=val_labels,
        test_texts=test_texts, test_labels=test_labels,
    )
    all_configs.append(("D: TF-IDF only", r_D))

    # Config A: TF-IDF + EN fastText
    if en_vecs is not None:
        r_A = run_config(
            "A: TF-IDF + EN fastText only",
            lookup=en_vecs,
            train_texts=train_texts, train_labels=train_labels,
            val_texts=val_texts, val_labels=val_labels,
            test_texts=test_texts, test_labels=test_labels,
        )
        all_configs.append(("A: TF-IDF + EN fastText only", r_A))
    else:
        print("\n[SKIP] Config A: English fastText not available.")
        all_configs.append(("A: TF-IDF + EN fastText only", None))

    # Config B: TF-IDF + ZH fastText
    if zh_vecs is not None:
        r_B = run_config(
            "B: TF-IDF + ZH fastText only",
            lookup=zh_vecs,
            train_texts=train_texts, train_labels=train_labels,
            val_texts=val_texts, val_labels=val_labels,
            test_texts=test_texts, test_labels=test_labels,
        )
        all_configs.append(("B: TF-IDF + ZH fastText only", r_B))
    else:
        print("\n[SKIP] Config B: Chinese fastText not available.")
        all_configs.append(("B: TF-IDF + ZH fastText only", None))

    # Config C: TF-IDF + EN + ZH fastText (Sprint-2 full)
    if en_zh_lookup:
        r_C = run_config(
            "C: TF-IDF + EN+ZH fastText (Sprint-2)",
            lookup=en_zh_lookup,
            train_texts=train_texts, train_labels=train_labels,
            val_texts=val_texts, val_labels=val_labels,
            test_texts=test_texts, test_labels=test_labels,
        )
        all_configs.append(("C: TF-IDF + EN+ZH fastText (Sprint-2)", r_C))
    else:
        all_configs.append(("C: TF-IDF + EN+ZH fastText (Sprint-2)", None))

    print_summary(all_configs)


if __name__ == "__main__":
    main()
