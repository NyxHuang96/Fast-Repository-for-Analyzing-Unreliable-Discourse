"""
File: ablation_2_ner_lambda.py
Author: Tianhao Cao
Date: 2026-04-12
Course: COLX 581 — Sprint 4 (Detailed Analysis)

Ablation 2 — NER auxiliary loss weight λ in Neural MTL
=======================================================
Research question:
    Sprint-3 fixed λ=0.3 (i.e., L = L_cls + 0.3 * L_ner).
    This ablation sweeps λ ∈ {0.0, 0.3, 1.0} to measure sensitivity.

    λ=0.0  → pure single-task DistilBERT (same backbone, NER head computed
              but not back-propagated). This is NOT the same as the Sprint-1
              baseline — the NER head is still present, just its loss is
              zeroed. Use this as the "no MTL" control.
    λ=0.3  → Sprint-3 setting (reproduced).
    λ=1.0  → equal-weight MTL (NER loss as strong as classification loss).

Architecture, tokeniser, training loop, hyperparameters, and early-stopping
are identical to mtl_neural.py (Sprint-3).
"""

import json
import os
import random
import sys
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

# Add sprint-3 src to path so we can import silver_ner
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "03_sprint_581"))

from silver_ner import align_bio_to_wordpieces, NUM_NER_LABELS, PAD_NER_ID  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 581
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
PATIENCE = 3

LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
LABEL_NAMES = ["Ham", "Phish", "Spam"]

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# λ values to sweep
LAMBDA_VALUES = [0.0, 0.3, 1.0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data ───────────────────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list[dict]:
    out = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class MTLEmailDataset(Dataset):
    """Same as Sprint-3 MTLEmailDataset — tokenises + silver-labels in __init__."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        offset_mapping = enc["offset_mapping"].tolist()
        special_tokens_mask = enc["special_tokens_mask"].tolist()

        ner_label_lists = [
            align_bio_to_wordpieces(text, offsets, specials)
            for text, offsets, specials in zip(texts, offset_mapping, special_tokens_mask)
        ]
        self.ner_labels = torch.tensor(ner_label_lists, dtype=torch.long)
        self.cls_labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "cls_labels": self.cls_labels[idx],
            "ner_labels": self.ner_labels[idx],
        }


def prepare_split(records):
    texts = [r["text"] for r in records]
    labels = [LABEL2ID[r["label"]] for r in records]
    return texts, labels


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────
class DistilBertMTL(nn.Module):
    def __init__(self, model_name, num_cls_labels, num_ner_labels, dropout=0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, num_cls_labels)
        self.ner_head = nn.Linear(hidden, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        pooled = hidden[:, 0]
        cls_logits = self.cls_head(self.dropout(pooled))
        ner_logits = self.ner_head(self.dropout(hidden))
        return cls_logits, ner_logits


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler,
                    cls_criterion, ner_criterion, device, lambda_ner):
    model.train()
    total, n = 0.0, 0
    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls_labels = batch["cls_labels"].to(device)
        ner_labels = batch["ner_labels"].to(device)

        optimizer.zero_grad()
        cls_logits, ner_logits = model(input_ids, attention_mask)
        loss_cls = cls_criterion(cls_logits, cls_labels)
        loss_ner = ner_criterion(
            ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1)
        )
        loss = loss_cls + lambda_ner * loss_ner
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        bs = cls_labels.size(0)
        total += loss.item() * bs
        n += bs
        preds_all.extend(cls_logits.argmax(-1).cpu().numpy())
        labels_all.extend(cls_labels.cpu().numpy())
    return total / n, accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro")


@torch.no_grad()
def evaluate(model, loader, cls_criterion, device):
    model.eval()
    total, n = 0.0, 0
    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls_labels = batch["cls_labels"].to(device)

        cls_logits, _ = model(input_ids, attention_mask)
        loss = cls_criterion(cls_logits, cls_labels)
        bs = cls_labels.size(0)
        total += loss.item() * bs
        n += bs
        preds_all.extend(cls_logits.argmax(-1).cpu().numpy())
        labels_all.extend(cls_labels.cpu().numpy())
    return (
        total / n,
        accuracy_score(labels_all, preds_all),
        f1_score(labels_all, preds_all, average="macro"),
        np.array(preds_all),
        np.array(labels_all),
    )


def print_results(title, preds, true, label_names):
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")
    per_class = f1_score(true, preds, average=None, labels=list(range(NUM_LABELS)))

    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=label_names, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true, preds, labels=list(range(NUM_LABELS)))
    print("          " + "  ".join(f"{n:>6}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    return {
        "accuracy": acc, "macro_f1": f1,
        "ham_f1": per_class[0], "phish_f1": per_class[1], "spam_f1": per_class[2],
    }


# ── Run one λ configuration ────────────────────────────────────────────────────
def run_lambda(
    lambda_ner: float,
    train_loader, val_loader, test_loader,
    train_labels, val_records, test_records,
    device,
) -> dict:
    set_seed(SEED)
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    tag = "no-MTL" if lambda_ner == 0.0 else f"λ={lambda_ner}"
    print(f"\n\n{'#'*64}")
    print(f"  RUNNING: DistilBERT MTL  ({tag})")
    print(f"{'#'*64}")

    model = DistilBertMTL(MODEL_NAME, NUM_LABELS, NUM_NER_LABELS).to(device)

    cls_weights = compute_class_weights(train_labels, NUM_LABELS).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)
    ner_criterion = nn.CrossEntropyLoss(ignore_index=PAD_NER_ID)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print(f"\n{'Epoch':>5}  {'Train L':>8}  {'Train Acc':>9}  {'Train F1':>8}  "
          f"{'Val L':>7}  {'Val Acc':>7}  {'Val F1':>6}")
    print("-" * 64)

    best_f1, patience_count = -1.0, 0
    best_weights = None

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            cls_criterion, ner_criterion, device, lambda_ner,
        )
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, cls_criterion, device)
        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>9.4f}  {tr_f1:>8.4f}  "
              f"{v_loss:>7.4f}  {v_acc:>7.4f}  {v_f1:>6.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            patience_count = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest validation macro-F1: {best_f1:.4f}")
    model.load_state_dict(best_weights)

    _, test_acc, test_f1, preds, true = evaluate(model, test_loader, cls_criterion, device)
    result = print_results(f"TEST — MTL ({tag})", preds, true, label_names)
    result["best_val_f1"] = best_f1
    result["lambda_ner"] = lambda_ner
    return result


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(all_results: list[dict]) -> None:
    print("\n\n" + "=" * 80)
    print("  ABLATION 2 — SUMMARY TABLE")
    print("  Research question: How sensitive is Neural MTL to the NER loss weight λ?")
    print("=" * 80)
    header = (f"{'Configuration':<36} {'λ':>5} {'Best ValF1':>10} "
              f"{'TestAcc':>8} {'MacroF1':>8} {'Ham':>6} {'Phish':>7} {'Spam':>7}")
    print(header)
    print("─" * 80)

    # S3 documented result for reference
    print(f"{'Sprint-3 (documented)':<36} {'0.30':>5} {'0.7231':>10} "
          f"{'0.5625':>8} {'0.4353':>8} {'0.6000':>6} {'0.7059':>7} {'0.0000':>7}  [ref]")

    for r in all_results:
        tag = "no-MTL" if r["lambda_ner"] == 0.0 else f"{r['lambda_ner']:.2f}"
        label = f"DistilBERT MTL (λ={tag})"
        print(f"{label:<36} {r['lambda_ner']:>5.2f} {r['best_val_f1']:>10.4f} "
              f"{r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
              f"{r['ham_f1']:>6.4f} {r['phish_f1']:>7.4f} {r['spam_f1']:>7.4f}  [live]")
    print("─" * 80)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print(" ABLATION 2 — NER λ Sweep (λ ∈ {0.0, 0.3, 1.0})")
    print("=" * 64)

    print("\nLoading data …")
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    train_texts, train_label_ids = prepare_split(train_records)
    val_texts, val_label_ids = prepare_split(val_records)
    test_texts, test_label_ids = prepare_split(test_records)

    print(f"  Train={len(train_records)}  Val={len(val_records)}  Test={len(test_records)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print("\nBuilding datasets (running spaCy silver NER) …")
    train_ds = MTLEmailDataset(train_texts, train_label_ids, tokenizer, MAX_LENGTH)
    val_ds = MTLEmailDataset(val_texts, val_label_ids, tokenizer, MAX_LENGTH)
    test_ds = MTLEmailDataset(test_texts, test_label_ids, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")
    print(f"λ values to test: {LAMBDA_VALUES}")

    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_lambda(
            lam,
            train_loader, val_loader, test_loader,
            train_label_ids, val_records, test_records, device,
        )
        all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
