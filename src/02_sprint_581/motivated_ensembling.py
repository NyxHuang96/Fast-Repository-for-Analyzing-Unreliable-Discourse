"""
Motivated Ensemble — three-layer improvement over naive soft voting:

  1. Per-class bias correction
     Each model's probability for class c is scaled by that model's precision
     on class c (measured on the validation set). This down-weights classes the
     model habitually over-predicts. Probabilities are renormalised after scaling.

  2. Entropy-based per-sample confidence weighting
     A model that is uncertain (high-entropy distribution) contributes less on
     that specific sample than a confident model. The per-sample effective weight
     is proportional to (1 - normalised_entropy).

  3. Validation-tuned global base weight
     A grid search over base_w_bert ∈ [0, 1] finds the split that maximises
     macro-F1 on the held-out validation set. This base ratio is then modulated
     by the per-sample entropy weights above.
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

LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["Ham", "Phish", "Spam"]
NUM_LABELS = len(LABEL_NAMES)
MAX_LENGTH = 256
BATCH_SIZE = 16
SEED = 581

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
NEURAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "baseline_neural")


# ─── Data helpers ────────────────────────────────────────────────────────────

def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bilingual_tokenizer(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(jieba.cut(text))
    return re.findall(r"\b\w+\b", text.lower())


def combined_text(record):
    return f"{record.get('text', '')} {record.get('text_zh', '')}".strip()


# ─── Model inference ─────────────────────────────────────────────────────────

def get_bert_predictions(texts, device):
    """Return (pred_ids, softmax_probs) for a list of texts."""
    print("Loading DistilBERT model …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        NEURAL_MODEL_DIR, local_files_only=True
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        NEURAL_MODEL_DIR, local_files_only=True
    ).to(device)
    model.eval()

    all_probs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)
    return preds, probs


def train_svc(texts, labels):
    print("Training TF-IDF + LinearSVC …")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=bilingual_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
        )),
        ("clf", LinearSVC(random_state=42, class_weight="balanced")),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def get_svc_predictions(pipeline, texts):
    """Return (pred_ids, softmax'd_decision_probs)."""
    pred_labels = pipeline.predict(texts)
    pred_ids = np.array([LABEL2ID[l] for l in pred_labels])
    scores = pipeline.decision_function(texts)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return pred_ids, probs


# ─── Motivated ensemble components ───────────────────────────────────────────

def compute_class_reliabilities(true_labels, pred_ids):
    """
    Per-class precision on a labelled dataset.
    Returns shape (n_classes,): reliability[c] = precision for class c.
    A model that habitually over-predicts class c will have a low precision
    for that class, and its probabilities for that class will be scaled down.
    """
    report = classification_report(
        true_labels,
        [ID2LABEL[p] for p in pred_ids],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    rel = np.array([report[cls]["precision"] for cls in LABEL_NAMES])
    return rel


def apply_reliability_correction(probs, reliabilities):
    """
    Scale each class column by the model's precision for that class,
    then renormalise rows to sum to 1.
    """
    corrected = probs * reliabilities[np.newaxis, :]
    row_sums = corrected.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
    return corrected / row_sums


def entropy_confidence(probs):
    """
    Per-sample confidence = 1 - H(p) / H_max, where H_max = log(n_classes).
    Returns shape (n_samples,) in [0, 1]: 1 = perfectly confident, 0 = uniform.
    """
    max_H = np.log(NUM_LABELS)
    clipped = np.clip(probs, 1e-9, 1.0)
    H = -np.sum(clipped * np.log(clipped), axis=1)
    return 1.0 - H / max_H


def motivated_ensemble_core(
    bert_probs, svc_probs, bert_rel, svc_rel, base_w_bert
):
    """
    Full three-layer motivated ensemble. Returns (pred_ids, combined_probs).

    Layer 1 — bias correction:
        Scale class probabilities by per-class precision, renormalise.
    Layer 2 — entropy weighting:
        Per-sample weight ∝ base_global_weight × confidence.
    Layer 3 — weighted combination:
        Normalised per-sample weights applied to corrected distributions.
    """
    base_w_svc = 1.0 - base_w_bert

    # Layer 1
    bert_cal = apply_reliability_correction(bert_probs, bert_rel)
    svc_cal  = apply_reliability_correction(svc_probs,  svc_rel)

    # Layer 2
    conf_bert = entropy_confidence(bert_cal)
    conf_svc  = entropy_confidence(svc_cal)

    # Layer 3
    eff_bert = base_w_bert * conf_bert
    eff_svc  = base_w_svc  * conf_svc
    total = eff_bert + eff_svc
    total = np.where(total < 1e-9, 1.0, total)
    eff_bert /= total
    eff_svc  /= total

    combined = eff_bert[:, np.newaxis] * bert_cal + eff_svc[:, np.newaxis] * svc_cal
    return combined.argmax(axis=1), combined


def grid_search_weights(
    bert_probs, svc_probs, true_labels, bert_rel, svc_rel, steps=21
):
    """
    Search base_w_bert ∈ linspace(0, 1, steps) and return the value that
    maximises macro-F1 on the validation set.
    """
    best_f1, best_w = -1.0, 0.5
    for w in np.linspace(0.0, 1.0, steps):
        preds, _ = motivated_ensemble_core(
            bert_probs, svc_probs, bert_rel, svc_rel, base_w_bert=w
        )
        f1 = f1_score(
            true_labels,
            [ID2LABEL[p] for p in preds],
            average="macro",
            labels=LABEL_NAMES,
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print(f"  Grid search → best base_w_bert = {best_w:.2f}  (val macro-F1 = {best_f1:.4f})")
    return best_w


# ─── Evaluation ──────────────────────────────────────────────────────────────

def print_results(title, true_labels, pred_ids):
    pred_labels = [ID2LABEL[p] for p in pred_ids]
    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_NAMES)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                                target_names=LABEL_NAMES, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_NAMES)
    header = "          " + "  ".join(f"{n:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    val_labels  = [r["label"] for r in val_records]
    test_labels = [r["label"] for r in test_records]

    # English text for BERT; bilingual for SVC
    val_texts_en   = [r["text"] for r in val_records]
    test_texts_en  = [r["text"] for r in test_records]
    val_texts_bi   = [combined_text(r) for r in val_records]
    test_texts_bi  = [combined_text(r) for r in test_records]
    train_texts_bi = [combined_text(r) for r in train_records]
    train_labels   = [r["label"] for r in train_records]

    print(f"\nDataset sizes  — train: {len(train_records)}  "
          f"val: {len(val_records)}  test: {len(test_records)}")

    # ── BERT: run once over val + test together to save load time ──────────────
    print("\n[BERT] Running inference on val + test …")
    all_en = val_texts_en + test_texts_en
    _, all_bert_probs = get_bert_predictions(all_en, device)
    bert_val_probs  = all_bert_probs[:len(val_texts_en)]
    bert_test_probs = all_bert_probs[len(val_texts_en):]
    bert_val_preds  = bert_val_probs.argmax(axis=1)
    bert_test_preds = bert_test_probs.argmax(axis=1)

    # ── SVC: train on full training set, predict val + test ───────────────────
    svc = train_svc(train_texts_bi, train_labels)
    svc_val_preds,  svc_val_probs  = get_svc_predictions(svc, val_texts_bi)
    svc_test_preds, svc_test_probs = get_svc_predictions(svc, test_texts_bi)

    # ── Compute per-class reliabilities from validation set ───────────────────
    print("\n[Calibration] Computing per-class reliabilities on validation set …")
    bert_rel = compute_class_reliabilities(val_labels, bert_val_preds)
    svc_rel  = compute_class_reliabilities(val_labels, svc_val_preds)

    print(f"\n  BERT reliabilities (precision per class):")
    for cls, r in zip(LABEL_NAMES, bert_rel):
        print(f"    {cls:<6}: {r:.4f}")
    print(f"\n  SVC reliabilities (precision per class):")
    for cls, r in zip(LABEL_NAMES, svc_rel):
        print(f"    {cls:<6}: {r:.4f}")

    # ── Grid search for optimal base weight ───────────────────────────────────
    print("\n[Grid Search] Tuning base_w_bert on validation set …")
    opt_w_bert = grid_search_weights(
        bert_val_probs, svc_val_probs, val_labels, bert_rel, svc_rel
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  TEST SET RESULTS")
    print("="*60)

    print_results("DistilBERT (standalone)", test_labels, bert_test_preds)
    print_results("TF-IDF + LinearSVC (standalone)", test_labels, svc_test_preds)

    # Naive ensemble baseline (equal weights, no correction)
    naive_preds = (0.5 * bert_test_probs + 0.5 * svc_test_probs).argmax(axis=1)
    print_results("Naive Ensemble — equal weights (0.5 / 0.5)", test_labels, naive_preds)

    # Motivated ensemble
    motivated_preds, _ = motivated_ensemble_core(
        bert_test_probs, svc_test_probs, bert_rel, svc_rel, base_w_bert=opt_w_bert
    )
    print_results(
        f"Motivated Ensemble  [bias-corrected · entropy-weighted · base_w_bert={opt_w_bert:.2f}]",
        test_labels,
        motivated_preds,
    )


if __name__ == "__main__":
    main()
