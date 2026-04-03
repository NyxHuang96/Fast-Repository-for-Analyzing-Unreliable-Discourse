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

def load_jsonl(filepath):
    """Load a JSONL file and return raw records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bilingual_tokenizer(text):
    """Jieba for Chinese text, regex word tokenizer for English."""
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(jieba.cut(text))
    return re.findall(r"\b\w+\b", text.lower())


def minmax_normalise(scores):
    """Min-max normalise an array to [0, 1]."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        return np.full_like(scores, 0.5)
    return (scores - s_min) / (s_max - s_min)


# Neural Baseline Model
def get_bert_predictions(test_texts, device):
    """Return predicted labels and full softmax probability distributions."""
    print("Loading DistilBERT model …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(NEURAL_MODEL_DIR, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(NEURAL_MODEL_DIR, local_files_only=True).to(device)
    model.eval()

    all_preds = []
    all_probs = []

    # Process in batches
    for start in range(0, len(test_texts), BATCH_SIZE):
        batch_texts = test_texts[start : start + BATCH_SIZE]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)

        preds = probs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.array(all_preds), np.vstack(all_probs)


# LinearSVC baseline model

def train_svc(train_texts, train_labels):
    """Train the SVC pipeline and return it."""
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
    pipeline.fit(train_texts, train_labels)
    return pipeline


def get_svc_predictions(pipeline, test_texts):
    """Return predicted label ids and softmax'd decision-function probabilities."""
    pred_labels = pipeline.predict(test_texts)
    pred_ids = np.array([LABEL2ID[l] for l in pred_labels])

    # decision_function: shape (n_samples, n_classes) for multi-class
    decision_scores = pipeline.decision_function(test_texts)

    # Convert to probability-like scores via softmax so they are on the same scale as DistilBERT's softmax output
    exp_scores = np.exp(decision_scores - decision_scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return pred_ids, probs


# Ensemble

def ensemble_predictions(bert_probs, svc_probs, w_bert=0.5, w_svc=0.5):
    """
    Weighted soft voting: average the full probability distributions from
    both models, then take argmax.

    Args:
        bert_probs: (n_samples, n_classes) softmax probabilities from DistilBERT
        svc_probs:  (n_samples, n_classes) softmax'd decision-function from SVC
        w_bert:     weight for DistilBERT (default 0.5)
        w_svc:      weight for SVC (default 0.5)
    """
    # Weighted average of probability distributions
    combined_probs = w_bert * bert_probs + w_svc * svc_probs
    ensemble_preds = combined_probs.argmax(axis=1)

    print(f"\nEnsemble weights: DistilBERT={w_bert:.2f}, SVC={w_svc:.2f}")
    pred_counts = {ID2LABEL[i]: (ensemble_preds == i).sum() for i in range(NUM_LABELS)}
    print(f"Prediction distribution: {pred_counts}")

    return ensemble_preds


# Evaluation

def print_results(title, true_labels, pred_ids):
    """Print accuracy, macro-F1, classification report, and confusion matrix."""
    pred_labels = [ID2LABEL[p] for p in pred_ids]
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro",
                  labels=LABEL_NAMES)

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



def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    test_records = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    test_texts_en = [r["text"] for r in test_records]
    test_labels = [r["label"] for r in test_records]

    # SVC uses combined en+zh text (matches its training setup)
    test_texts_combined = [
        f"{r.get('text', '')} {r.get('text_zh', '')}".strip()
        for r in test_records
    ]
    train_texts_combined = [
        f"{r.get('text', '')} {r.get('text_zh', '')}".strip()
        for r in train_records
    ]
    train_labels = [r["label"] for r in train_records]

    print(f"Test samples: {len(test_records)}")

    # Get predictions from both models
    bert_preds, bert_probs = get_bert_predictions(test_texts_en, device)
    svc_pipeline = train_svc(train_texts_combined, train_labels)
    svc_preds, svc_probs = get_svc_predictions(svc_pipeline, test_texts_combined)

    # Individual model results 
    print_results("DistilBERT (standalone)", test_labels, bert_preds)
    print_results("TF-IDF + LinearSVC (standalone)", test_labels, svc_preds)

    # Ensemble results with different weight configs
    weight_configs = [
        (0.5, 0.5, "Equal weights (0.5 / 0.5)"),
        (0.4, 0.6, "SVC-favoured (0.4 / 0.6)"),
        (0.3, 0.7, "SVC-heavy (0.3 / 0.7)"),
    ]

    for w_bert, w_svc, desc in weight_configs:
        ensemble_preds = ensemble_predictions(bert_probs, svc_probs,
                                              w_bert=w_bert, w_svc=w_svc)
        print_results(f"ENSEMBLE — {desc}", test_labels, ensemble_preds)


if __name__ == "__main__":
    main()
