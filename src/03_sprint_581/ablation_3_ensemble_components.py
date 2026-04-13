"""
File: ablation_3_ensemble_components.py
Author: Yusen Huang
Date: 2026-04-12
Course: COLX 581 — Sprint 4 (Detailed Analysis)

Ablation 3 — Motivated Ensemble Component Ablation
===================================================
Research question:
    The Sprint-2 motivated ensemble has three layers on top of naive
    soft voting:
        Layer 1 — Per-class bias correction (precision scaling + renorm)
        Layer 2 — Entropy-based per-sample confidence weighting
        Layer 3 — Validation-tuned global base weight (grid search)

    We remove each layer individually to measure its contribution:

    Config A  Full motivated ensemble (all three layers)        [baseline]
    Config B  No bias correction  (Layers 2 + 3 only)
    Config C  No entropy weighting (Layers 1 + 3 only)
    Config D  Fixed weight 0.3/0.7  (Layers 1 + 2, no grid search)
              — uses the Sprint-2 hand-picked weight that was best in that doc

    All configs share the same BERT and SVC predictions so differences
    are *entirely* attributable to the ablated component.
"""

import json
import os
import re
import warnings

import jieba
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 581
LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["Ham", "Phish", "Spam"]
NUM_LABELS = len(LABEL_NAMES)
MAX_LENGTH = 256
BATCH_SIZE = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
NEURAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "baseline_neural")

# Fixed weight used as the "no grid search" fallback (Sprint-2 best hand-picked)
FIXED_W_BERT = 0.3

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bilingual_tokenizer(text: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(jieba.cut(text))
    return re.findall(r"\b\w+\b", text.lower())


def combined_text(record: dict) -> str:
    return f"{record.get('text', '')} {record.get('text_zh', '')}".strip()


# ── Model inference ────────────────────────────────────────────────────────────
def get_bert_predictions(texts: list[str], device) -> tuple[np.ndarray, np.ndarray]:
    print("  Loading DistilBERT checkpoint …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(NEURAL_MODEL_DIR, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        NEURAL_MODEL_DIR, local_files_only=True
    ).to(device)
    model.eval()

    all_probs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    probs = np.vstack(all_probs)
    return probs.argmax(axis=1), probs


def train_svc(texts: list[str], labels: list[str]):
    print("  Training TF-IDF + LinearSVC …")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=bilingual_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
        )),
        ("clf", LinearSVC(random_state=42, class_weight="balanced")),
    ])
    pipe.fit(texts, labels)
    return pipe


def get_svc_predictions(pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pred_labels = pipeline.predict(texts)
    pred_ids = np.array([LABEL2ID[l] for l in pred_labels])
    scores = pipeline.decision_function(texts)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return pred_ids, probs


# ── Ensemble primitives ────────────────────────────────────────────────────────
def compute_class_reliabilities(true_labels: list[str], pred_ids: np.ndarray) -> np.ndarray:
    report = classification_report(
        true_labels,
        [ID2LABEL[p] for p in pred_ids],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return np.array([report[cls]["precision"] for cls in LABEL_NAMES])


def apply_reliability_correction(probs: np.ndarray, reliabilities: np.ndarray) -> np.ndarray:
    corrected = probs * reliabilities[np.newaxis, :]
    row_sums = corrected.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
    return corrected / row_sums


def entropy_confidence(probs: np.ndarray) -> np.ndarray:
    max_H = np.log(NUM_LABELS)
    clipped = np.clip(probs, 1e-9, 1.0)
    H = -np.sum(clipped * np.log(clipped), axis=1)
    return 1.0 - H / max_H


def combine(bert_probs, svc_probs, eff_bert, eff_svc) -> np.ndarray:
    """Normalize per-sample weights and return weighted combination."""
    total = eff_bert + eff_svc
    total = np.where(total < 1e-9, 1.0, total)
    w_bert = eff_bert / total
    w_svc  = eff_svc  / total
    return w_bert[:, np.newaxis] * bert_probs + w_svc[:, np.newaxis] * svc_probs


def grid_search_weights(
    bert_probs, svc_probs, true_labels, bert_rel, svc_rel,
    use_bias_correction=True, use_entropy=True, steps=21,
) -> float:
    """Parameterised grid search that respects ablation flags."""
    best_f1, best_w = -1.0, 0.5
    for w_bert in np.linspace(0.0, 1.0, steps):
        preds = _ensemble_predict(
            bert_probs, svc_probs, bert_rel, svc_rel, w_bert,
            use_bias_correction=use_bias_correction,
            use_entropy=use_entropy,
        )
        f1 = f1_score(
            true_labels, [ID2LABEL[p] for p in preds],
            average="macro", labels=LABEL_NAMES, zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_w = f1, w_bert
    print(f"    Grid search → best w_bert = {best_w:.2f}  (val macro-F1 = {best_f1:.4f})")
    return best_w


def _ensemble_predict(
    bert_probs, svc_probs, bert_rel, svc_rel, base_w_bert,
    use_bias_correction=True, use_entropy=True,
) -> np.ndarray:
    """Core ensemble prediction, parameterised by ablation flags."""
    base_w_svc = 1.0 - base_w_bert

    # Layer 1: bias correction
    if use_bias_correction:
        bp = apply_reliability_correction(bert_probs, bert_rel)
        sp = apply_reliability_correction(svc_probs, svc_rel)
    else:
        bp = bert_probs.copy()
        sp = svc_probs.copy()

    # Layer 2: entropy weighting
    if use_entropy:
        conf_bert = entropy_confidence(bp) * base_w_bert
        conf_svc  = entropy_confidence(sp) * base_w_svc
    else:
        conf_bert = np.full(len(bp), base_w_bert)
        conf_svc  = np.full(len(sp), base_w_svc)

    combined = combine(bp, sp, conf_bert, conf_svc)
    return combined.argmax(axis=1)


# ── Eval ───────────────────────────────────────────────────────────────────────
def print_results(title: str, true_labels: list[str], pred_ids: np.ndarray) -> dict:
    pred_labels = [ID2LABEL[p] for p in pred_ids]
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

    return {
        "accuracy": acc, "macro_f1": f1,
        "ham_f1": per_class[0], "phish_f1": per_class[1], "spam_f1": per_class[2],
    }


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(configs: list[tuple]) -> None:
    print("\n\n" + "=" * 80)
    print("  ABLATION 3 — ENSEMBLE COMPONENT ABLATION SUMMARY")
    print("  Research question: Which motivated ensemble layer contributes most?")
    print("=" * 80)
    header = (f"{'Configuration':<45} {'Acc':>6} {'MacroF1':>8} "
              f"{'Ham':>6} {'Phish':>7} {'Spam':>7}")
    print(header)
    print("─" * 80)

    # Sprint-2 documented reference
    print(f"{'Naive ensemble (0.5/0.5, documented)':<45} {'0.5625':>6} {'0.5574':>8} "
          f"{'0.8889':>6} {'0.5333':>7} {'0.2500':>7}  [ref]")
    print(f"{'Naive best fixed 0.3/0.7 (documented)':<45} {'0.6250':>6} {'0.6349':>8} "
          f"{'0.8889':>6} {'0.5714':>7} {'0.4444':>7}  [ref]")

    for label, result in configs:
        if result is not None:
            print(f"{label:<45} {result['accuracy']:>6.4f} {result['macro_f1']:>8.4f} "
                  f"{result['ham_f1']:>6.4f} {result['phish_f1']:>7.4f} {result['spam_f1']:>7.4f}  [live]")
    print("─" * 80)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print(" ABLATION 3 — Motivated Ensemble Component Ablation")
    print(" Configs: Full | No-BiasCorr | No-Entropy | Fixed-Weight")
    print("=" * 64)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # Load data
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    val_labels  = [r["label"] for r in val_records]
    test_labels = [r["label"] for r in test_records]

    val_en    = [r["text"] for r in val_records]
    test_en   = [r["text"] for r in test_records]
    val_bi    = [combined_text(r) for r in val_records]
    test_bi   = [combined_text(r) for r in test_records]
    train_bi  = [combined_text(r) for r in train_records]
    train_labels_str = [r["label"] for r in train_records]

    # ── Shared inference (run once) ────────────────────────────────────────────
    print("\n[BERT] Running inference on val + test …")
    all_en = val_en + test_en
    _, all_bert_probs = get_bert_predictions(all_en, device)
    bert_val_probs  = all_bert_probs[:len(val_en)]
    bert_test_probs = all_bert_probs[len(val_en):]
    bert_val_preds  = bert_val_probs.argmax(axis=1)
    bert_test_preds = bert_test_probs.argmax(axis=1)

    print("\n[SVC] Training and predicting …")
    svc = train_svc(train_bi, train_labels_str)
    svc_val_preds,  svc_val_probs  = get_svc_predictions(svc, val_bi)
    svc_test_preds, svc_test_probs = get_svc_predictions(svc, test_bi)

    # Per-class reliabilities from validation
    print("\n[Reliabilities] Computing from validation …")
    bert_rel = compute_class_reliabilities(val_labels, bert_val_preds)
    svc_rel  = compute_class_reliabilities(val_labels, svc_val_preds)
    for name, br, sr in zip(LABEL_NAMES, bert_rel, svc_rel):
        print(f"  {name:<6}: BERT={br:.4f}  SVC={sr:.4f}")

    results_by_config = []

    # ── Standalone baselines for reference ───────────────────────────────────
    print_results("DistilBERT standalone (test)", test_labels, bert_test_preds)
    print_results("TF-IDF + SVC standalone (test)", test_labels, svc_test_preds)

    # ── Config A: Full motivated ensemble ─────────────────────────────────────
    print("\n\n>>> Config A: FULL Motivated Ensemble (Layers 1+2+3) <<<")
    opt_w = grid_search_weights(
        bert_val_probs, svc_val_probs, val_labels, bert_rel, svc_rel,
        use_bias_correction=True, use_entropy=True,
    )
    preds_A = _ensemble_predict(
        bert_test_probs, svc_test_probs, bert_rel, svc_rel, opt_w,
        use_bias_correction=True, use_entropy=True,
    )
    r_A = print_results("Config A: Full Motivated Ensemble (test)", test_labels, preds_A)
    results_by_config.append(("A: Full (bias corr + entropy + grid search)", r_A))

    # ── Config B: No bias correction  ─────────────────────────────────────────
    print("\n\n>>> Config B: No Bias Correction (Layers 2+3 only) <<<")
    opt_w_B = grid_search_weights(
        bert_val_probs, svc_val_probs, val_labels, bert_rel, svc_rel,
        use_bias_correction=False, use_entropy=True,
    )
    preds_B = _ensemble_predict(
        bert_test_probs, svc_test_probs, bert_rel, svc_rel, opt_w_B,
        use_bias_correction=False, use_entropy=True,
    )
    r_B = print_results("Config B: No Bias Correction (test)", test_labels, preds_B)
    results_by_config.append(("B: No bias correction", r_B))

    # ── Config C: No entropy weighting ────────────────────────────────────────
    print("\n\n>>> Config C: No Entropy Weighting (Layers 1+3 only) <<<")
    opt_w_C = grid_search_weights(
        bert_val_probs, svc_val_probs, val_labels, bert_rel, svc_rel,
        use_bias_correction=True, use_entropy=False,
    )
    preds_C = _ensemble_predict(
        bert_test_probs, svc_test_probs, bert_rel, svc_rel, opt_w_C,
        use_bias_correction=True, use_entropy=False,
    )
    r_C = print_results("Config C: No Entropy Weighting (test)", test_labels, preds_C)
    results_by_config.append(("C: No entropy weighting", r_C))

    # ── Config D: Fixed weight 0.3/0.7, Layers 1+2 only ──────────────────────
    print(f"\n\n>>> Config D: Fixed w_bert={FIXED_W_BERT} + Layers 1+2 (no grid search) <<<")
    preds_D = _ensemble_predict(
        bert_test_probs, svc_test_probs, bert_rel, svc_rel, FIXED_W_BERT,
        use_bias_correction=True, use_entropy=True,
    )
    r_D = print_results(
        f"Config D: Fixed w_bert={FIXED_W_BERT} (no grid search) (test)",
        test_labels, preds_D,
    )
    results_by_config.append((f"D: Fixed w_bert={FIXED_W_BERT} (bias+entropy, no grid)", r_D))

    print_summary(results_by_config)


if __name__ == "__main__":
    main()
