"""
File: baseline_neural.py
Author: Marco Wang
Date: 2026-03-28
Last Updated: 2026-03-28
Course: COLX 523
Description:
    DistilBERT baseline for 3-class email classification (Ham / Phish / Spam).
    - Loads train/validation/test splits from data/processed/*.jsonl
    - Fine-tunes distilbert-base-uncased with class-weighted cross-entropy
    - Reports accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix
"""

import json
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Configuration
SEED = 581
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
PATIENCE = 3  # early-stopping patience (epochs without val improvement)

LABEL2ID = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Reproducibility

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data helpers

def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class EmailDataset(Dataset):
    """Simple PyTorch dataset for tokenised email texts."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def prepare_split(records: list[dict]):
    """Extract parallel lists of texts and integer labels from raw records."""
    texts = [r["text"] for r in records]
    labels = [LABEL2ID[r["label"]] for r in records]
    return texts, labels


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (normalised so they sum to num_classes)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# Training & evaluation loops

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


# Main

def main():
    set_seed(SEED)

    # ---- Resolve paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data", "processed"))
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "models", "baseline_neural"))
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print("DistilBERT Baseline — Email Classification (Ham/Phish/Spam)")
    print("=" * 60)

    train_records = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_records = load_jsonl(os.path.join(data_dir, "validation.jsonl"))
    test_records = load_jsonl(os.path.join(data_dir, "test.jsonl"))

    print(f"\nDataset sizes  →  Train: {len(train_records)}  |  Val: {len(val_records)}  |  Test: {len(test_records)}")

    train_texts, train_labels = prepare_split(train_records)
    val_texts, val_labels = prepare_split(val_records)
    test_texts, test_labels = prepare_split(test_records)

    # ---- Tokeniser & datasets ----
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = EmailDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = EmailDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ---- Model ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # ---- Class-weighted loss ----
    class_weights = compute_class_weights(train_labels, NUM_LABELS).to(device)
    print(f"Class weights:  {', '.join(f'{ID2LABEL[i]}: {w:.3f}' for i, w in enumerate(class_weights))}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimiser & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ---- Training loop with early stopping ----
    print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE}) …\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Train F1':>8}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}")
    print("-" * 72)

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  {train_f1:>8.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_f1:>6.4f}")

        # Early stopping on validation macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # ---- Reload best model & evaluate on test ----
    print(f"\nBest validation macro-F1: {best_val_f1:.4f}")
    print(f"Loading best model from {output_dir} …")
    model = DistilBertForSequenceClassification.from_pretrained(output_dir).to(device)

    test_loss, test_acc, test_f1, test_preds, test_true = evaluate(
        model, test_loader, criterion, device
    )

    # ---- Report ----
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Macro-F1  : {test_f1:.4f}")
    print(f"  Loss      : {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=label_names, digits=4))

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(test_true, test_preds, labels=list(range(NUM_LABELS)))
    # Pretty print
    header = "          " + "  ".join(f"{name:>6}" for name in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>6}" for val in row)
        print(f"  {label_names[i]:<6}  {row_str}")

    # ---- Save predictions ----
    predictions_path = os.path.join(output_dir, "test_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(test_records):
            result = {
                "text": record["text"],
                "true_label": record["label"],
                "predicted_label": ID2LABEL[int(test_preds[i])],
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nTest predictions saved to {predictions_path}")
    print("Done.")


if __name__ == "__main__":
    main()
