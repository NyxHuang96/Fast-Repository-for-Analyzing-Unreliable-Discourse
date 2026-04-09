"""
File: mtl_neural.py
Author: Yusen Huang
Date: 2026-04-09
"""

import argparse
import json
import os
import random
import warnings

import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# Configuration
# Setting SEED for reproducibility
SEED = 581

TRANSFER_MODEL = "distilbert-base-multilingual-cased"

MAX_LENGTH    = 256
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 10
PATIENCE      = 3
FREEZE_LAYERS = 3
NER_WEIGHT    = 0.3   # λ for auxiliary loss

LABEL2ID    = {"Ham": 0, "Phish": 1, "Spam": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["Ham", "Phish", "Spam"]
NUM_LABELS  = len(LABEL_NAMES)

# BIO NER tag set (spaCy entity types + O tag)
# We use a compact set covering the most relevant entity types for email
NER_TAGS = [
    "O",           # Outside any entity
    "B-PERSON",  "I-PERSON",
    "B-ORG",     "I-ORG",
    "B-GPE",     "I-GPE",       # Geo-political entity (countries, cities)
    "B-MONEY",   "I-MONEY",
    "B-DATE",    "I-DATE",
    "B-PRODUCT", "I-PRODUCT",
    "B-MISC",    "I-MISC",      # Catch-all for other entity types
]
NER_TAG2ID  = {t: i for i, t in enumerate(NER_TAGS)}
NER_ID2TAG  = {i: t for t, i in NER_TAG2ID.items()}
NUM_NER_TAGS = len(NER_TAGS)
NER_IGNORE_INDEX = -100  # PyTorch CE ignore index for subword / pad tokens

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")


# Reproducibility 
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Jsonl File Loading
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


# Silver NER annotation via spaCy

# Map spaCy entity labels to our compact tag set
SPACY_TO_TAG = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "GPE",       # fold LOC into GPE
    "MONEY": "MONEY",
    "DATE": "DATE",
    "TIME": "DATE",      # fold TIME into DATE
    "PRODUCT": "PRODUCT",
}


def generate_silver_ner(texts: list[str], nlp) -> list[list[str]]:
    """
    Use spaCy to produce character-level NER spans, then convert to
    word-level BIO tags aligned with whitespace tokenisation.

    Returns a list of tag sequences (one per text), where each tag
    sequence has the same length as text.split().
    """
    all_tags = []
    for doc in nlp.pipe(texts, batch_size=64):
        words = doc.text.split()
        tags = ["O"] * len(words)

        # Build character offset - word index mapping
        char2word = {}
        pos = 0
        for wi, word in enumerate(words):
            start = doc.text.find(word, pos)
            for ci in range(start, start + len(word)):
                char2word[ci] = wi
            pos = start + len(word)

        for ent in doc.ents:
            tag_type = SPACY_TO_TAG.get(ent.label_, "MISC")
            # Find which words this entity spans
            word_indices = sorted(set(
                char2word[ci]
                for ci in range(ent.start_char, ent.end_char)
                if ci in char2word
            ))
            for i, wi in enumerate(word_indices):
                prefix = "B" if i == 0 else "I"
                tags[wi] = f"{prefix}-{tag_type}"

        all_tags.append(tags)
    return all_tags


def align_ner_to_tokens(
    text: str,
    word_tags: list[str],
    tokenizer,
    max_length: int,
) -> list[int]:
    """
    Align word-level BIO tags to wordpiece tokens produced by the
    HuggingFace tokenizer. Only the first subword of each word gets the
    real tag; continuation subwords and special tokens get IGNORE_INDEX.
    """
    words = text.split()
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"][0].tolist()

    # Map each character offset to its word index
    char2word = {}
    pos = 0
    for wi, word in enumerate(words):
        start = text.find(word, pos)
        for ci in range(start, start + len(word)):
            char2word[ci] = wi
        pos = start + len(word)

    token_tags = []
    prev_word = None
    for (start, end) in offsets:
        if start == 0 and end == 0:
            # Special token ([CLS], [SEP], [PAD])
            token_tags.append(NER_IGNORE_INDEX)
            prev_word = None
            continue

        # Find which word this subword belongs to
        word_idx = char2word.get(start, None)
        if word_idx is None:
            token_tags.append(NER_IGNORE_INDEX)
            prev_word = None
            continue

        if word_idx == prev_word:
            # Continuation subword → ignore
            token_tags.append(NER_IGNORE_INDEX)
        else:
            # First subword of a new word → assign the word's tag
            if word_idx < len(word_tags):
                token_tags.append(NER_TAG2ID.get(word_tags[word_idx], 0))
            else:
                token_tags.append(0)  # O tag
            prev_word = word_idx

    return token_tags


# Dataset
class MTLEmailDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        ner_tags: list[list[str]],
        tokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Align NER tags to wordpiece tokens
        self.ner_labels = []
        for text, tags in zip(texts, ner_tags):
            aligned = align_ner_to_tokens(text, tags, tokenizer, max_length)
            self.ner_labels.append(aligned)
        self.ner_labels = torch.tensor(self.ner_labels, dtype=torch.long)

        # Remove offset mapping (not needed during training)
        del self.encodings["offset_mapping"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
            "ner_labels":     self.ner_labels[idx],
        }


# MTL Model
class DistilBertMTL(nn.Module):
    """
    Multi-task DistilBERT with two heads:
      - cls_head: [CLS] → email class (Ham/Phish/Spam)
      - ner_head: token hidden states → BIO-NER tags
    """

    def __init__(self, model_name: str, num_cls_labels: int, num_ner_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.cls_dropout = nn.Dropout(0.1)
        self.cls_head = nn.Linear(hidden_size, num_cls_labels)

        self.ner_dropout = nn.Dropout(0.1)
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden)

        # Head 1: classification from [CLS]
        cls_hidden = hidden_states[:, 0, :]  # (B, hidden)
        cls_logits = self.cls_head(self.cls_dropout(cls_hidden))  # (B, num_cls)

        # Head 2: NER from all tokens
        ner_logits = self.ner_head(self.ner_dropout(hidden_states))  # (B, seq_len, num_ner)

        return cls_logits, ner_logits


# Layer freezing 
def freeze_lower_layers(model: DistilBertMTL, n_freeze: int) -> None:
    if n_freeze == 0:
        print("  Layer freezing: DISABLED")
        return

    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False
    print("  Frozen: embedding layer")

    n_layers = len(model.encoder.transformer.layer)
    for i, block in enumerate(model.encoder.transformer.layer):
        if i < min(n_freeze, n_layers):
            for param in block.parameters():
                param.requires_grad = False
            print(f"  Frozen: transformer block {i}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# Training loop 
def train_one_epoch(model, loader, optimizer, scheduler, cls_criterion,
                    ner_criterion, ner_weight, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        ner_labels     = batch["ner_labels"].to(device)

        optimizer.zero_grad()
        cls_logits, ner_logits = model(input_ids, attention_mask)

        # Primary loss
        loss_cls = cls_criterion(cls_logits, labels)

        # Auxiliary NER loss (flatten tokens)
        loss_ner = ner_criterion(
            ner_logits.view(-1, NUM_NER_TAGS),
            ner_labels.view(-1),
        )

        loss = loss_cls + ner_weight * loss_ner
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = cls_logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, cls_criterion, ner_criterion, ner_weight, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        ner_labels     = batch["ner_labels"].to(device)

        cls_logits, ner_logits = model(input_ids, attention_mask)

        loss_cls = cls_criterion(cls_logits, labels)
        loss_ner = ner_criterion(
            ner_logits.view(-1, NUM_NER_TAGS),
            ner_labels.view(-1),
        )
        loss = loss_cls + ner_weight * loss_ner

        total_loss += loss.item() * labels.size(0)
        preds = cls_logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


# Experiment Excuetion
def run_experiment(
    model_name: str,
    experiment_label: str,
    train_texts: list[str],
    train_labels: list[int],
    train_ner: list[list[str]],
    val_texts: list[str],
    val_labels: list[int],
    val_ner: list[list[str]],
    test_texts: list[str],
    test_labels: list[int],
    test_ner: list[list[str]],
    test_records: list[dict],
    device: torch.device,
    freeze_n: int = 0,
    ner_weight: float = 0.3,
    output_subdir: str = "mtl_neural",
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_label}")
    print(f"  Backbone:   {model_name}")
    print(f"  Freeze:     {freeze_n} layers")
    print(f"  NER weight: {ner_weight}")
    print(f"{'='*60}")

    output_dir = os.path.join(PROJECT_ROOT, "models", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Tokeniser & datasets
    print(f"\nLoading tokeniser from {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Building datasets (with NER alignment) …")
    train_dataset = MTLEmailDataset(train_texts, train_labels, train_ner, tokenizer, MAX_LENGTH)
    val_dataset   = MTLEmailDataset(val_texts,   val_labels,   val_ner,   tokenizer, MAX_LENGTH)
    test_dataset  = MTLEmailDataset(test_texts,  test_labels,  test_ner,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    print(f"Loading model {model_name} …")
    model = DistilBertMTL(model_name, NUM_LABELS, NUM_NER_TAGS).to(device)

    if freeze_n > 0:
        print(f"\nFreezing bottom {freeze_n} transformer blocks …")
        freeze_lower_layers(model, freeze_n)

    # Loss functions
    class_weights = compute_class_weights(train_labels, NUM_LABELS).to(device)
    print(f"\nClass weights: " +
          ", ".join(f"{ID2LABEL[i]}={w:.3f}" for i, w in enumerate(class_weights)))

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    ner_criterion = nn.CrossEntropyLoss(ignore_index=NER_IGNORE_INDEX)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training
    print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE}) …\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Train F1':>8}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>6}")

    best_val_f1 = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            cls_criterion, ner_criterion, ner_weight, device,
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, cls_criterion, ner_criterion, ner_weight, device,
        )

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  {train_f1:>8.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_f1:>6.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Reload best & evaluate on test
    print(f"\nBest validation macro-F1: {best_val_f1:.4f}")
    model.load_state_dict(best_state)
    model.to(device)

    # Save best model
    torch.save(best_state, os.path.join(output_dir, "best_model.pt"))
    tokenizer.save_pretrained(output_dir)

    test_loss, test_acc, test_f1, test_preds, test_true = evaluate(
        model, test_loader, cls_criterion, ner_criterion, ner_weight, device,
    )

    # Report
    pred_labels = [ID2LABEL[p] for p in test_preds]
    true_labels_str = [ID2LABEL[t] for t in test_true]

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS — {experiment_label}")
    print(f"{'='*60}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Macro-F1 : {test_f1:.4f}")
    print(f"  Loss     : {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels_str, pred_labels,
                                target_names=LABEL_NAMES, digits=4))

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels_str, pred_labels, labels=LABEL_NAMES)
    header = "          " + "  ".join(f"{n:>6}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<6}  " + "  ".join(f"{v:>6}" for v in row))

    # ── Save predictions ──
    pred_path = os.path.join(output_dir, "test_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(test_records):
            row = {
                "text":            record.get("text", ""),
                "true_label":      record["label"],
                "predicted_label": pred_labels[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved → {pred_path}")

    return {
        "experiment":  experiment_label,
        "model":       model_name,
        "freeze_n":    freeze_n,
        "ner_weight":  ner_weight,
        "best_val_f1": best_val_f1,
        "test_acc":    test_acc,
        "test_f1":     test_f1,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTL DistilBERT: email classification + NER auxiliary task"
    )
    parser.add_argument(
        "--freeze", type=int, default=FREEZE_LAYERS,
        help=f"Transformer blocks to freeze (default: {FREEZE_LAYERS})."
    )
    parser.add_argument(
        "--ner-weight", type=float, default=NER_WEIGHT,
        help=f"Weight λ for NER auxiliary loss (default: {NER_WEIGHT})."
    )
    parser.add_argument(
        "--no-mtl", action="store_true",
        help="Ablation: disable MTL (set NER weight to 0)."
    )
    return parser.parse_args()


# Main
def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ner_weight = 0.0 if args.no_mtl else args.ner_weight

    print("=" * 60)
    print("  Multi-Task Learning | Neural Baseline")
    print("  mDistilBERT + NER Auxiliary Head")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading data splits …")
    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records   = load_jsonl(os.path.join(DATA_DIR, "validation.jsonl"))
    test_records  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}  |  Test: {len(test_records)}")

    # Bilingual text
    train_texts = [combined_text(r) for r in train_records]
    val_texts   = [combined_text(r) for r in val_records]
    test_texts  = [combined_text(r) for r in test_records]

    train_labels = [LABEL2ID[r["label"]] for r in train_records]
    val_labels   = [LABEL2ID[r["label"]] for r in val_records]
    test_labels  = [LABEL2ID[r["label"]] for r in test_records]

    # Generate silver NER labels via spaCy
    print("\nGenerating silver NER annotations via spaCy …")
    nlp = spacy.load("en_core_web_sm")
    train_ner = generate_silver_ner(train_texts, nlp)
    val_ner   = generate_silver_ner(val_texts, nlp)
    test_ner  = generate_silver_ner(test_texts, nlp)

    # Report NER statistics
    def ner_stats(tags_list):
        total = sum(len(t) for t in tags_list)
        entities = sum(1 for tags in tags_list for t in tags if t != "O")
        return total, entities

    for name, tags in [("Train", train_ner), ("Val", val_ner), ("Test", test_ner)]:
        total, ents = ner_stats(tags)
        print(f"  {name}: {total} tokens, {ents} entity tokens "
              f"({100*ents/max(total,1):.1f}%)")

    results = []

    # Experiment 1: Ablation — no MTL
    print("\n" + "─" * 60)
    print("  ABLATION: Classification only (no MTL)")
    print("─" * 60)
    r0 = run_experiment(
        model_name=TRANSFER_MODEL,
        experiment_label="mDistilBERT — no MTL (ablation)",
        train_texts=train_texts, train_labels=train_labels, train_ner=train_ner,
        val_texts=val_texts, val_labels=val_labels, val_ner=val_ner,
        test_texts=test_texts, test_labels=test_labels, test_ner=test_ner,
        test_records=test_records,
        device=device,
        freeze_n=args.freeze,
        ner_weight=0.0,
        output_subdir="mtl_neural_ablation",
    )
    results.append(r0)

    # Experiment 2: MTL with NER
    if not args.no_mtl:
        print("\n" + "─" * 60)
        print(f"  MTL: Classification + NER (λ={ner_weight})")
        print("─" * 60)
        r1 = run_experiment(
            model_name=TRANSFER_MODEL,
            experiment_label=f"mDistilBERT + NER MTL (λ={ner_weight})",
            train_texts=train_texts, train_labels=train_labels, train_ner=train_ner,
            val_texts=val_texts, val_labels=val_labels, val_ner=val_ner,
            test_texts=test_texts, test_labels=test_labels, test_ner=test_ner,
            test_records=test_records,
            device=device,
            freeze_n=args.freeze,
            ner_weight=ner_weight,
            output_subdir="mtl_neural_ner",
        )
        results.append(r1)

        # Experiment 3: MTL with higher NER weight
        high_w = min(ner_weight * 2, 1.0)
        print("\n" + "─" * 60)
        print(f"  MTL: Classification + NER (λ={high_w})")
        print("─" * 60)
        r2 = run_experiment(
            model_name=TRANSFER_MODEL,
            experiment_label=f"mDistilBERT + NER MTL (λ={high_w})",
            train_texts=train_texts, train_labels=train_labels, train_ner=train_ner,
            val_texts=val_texts, val_labels=val_labels, val_ner=val_ner,
            test_texts=test_texts, test_labels=test_labels, test_ner=test_ner,
            test_records=test_records,
            device=device,
            freeze_n=args.freeze,
            ner_weight=high_w,
            output_subdir=f"mtl_neural_ner_w{high_w:.1f}",
        )
        results.append(r2)

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<45}  {'λ':>4}  {'Val F1':>7}  {'Test Acc':>8}  {'Test F1':>7}")
    print("  " + "─" * 75)
    for r in results:
        print(f"  {r['experiment']:<45}  "
              f"{r['ner_weight']:>4.1f}  "
              f"{r['best_val_f1']:>7.4f}  "
              f"{r['test_acc']:>8.4f}  "
              f"{r['test_f1']:>7.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
