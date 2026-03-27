"""
File: build_subcorpora.py
Author: Yusen Huang
Date: 2026-03-02
Course: COLX 523
Description: Extract two subsets of corpus from One Huge corpus.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from nltk.tokenize import word_tokenize

script_dir    = Path(__file__).resolve().parent
processed_dir = script_dir.parent.parent / "data" / "processed"
file_path     = processed_dir / "kaggle_corpus.json"

# Extracting two subsets of corpus for future annotation

CORPUS_CONFIGS = [
    {"filename": "english_sub_corpus.json", "per_class": 16_667},
    {"filename": "chinese_sub_corpus.json", "per_class":  3_333},
]

TOTAL_PER_CLASS_NEEDED = sum(c["per_class"] for c in CORPUS_CONFIGS)  # 20,000

# Setting seed to ensure the file is reproducible
SEED = 523

# Load Source json files
def load_jsonl(path: Path) -> dict[str, list[dict]]:
    """Load a JSONL file and bucket records by their 'label' field, filtering by length."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                if "label" not in record or "text" not in record:
                    raise ValueError("Missing 'label' or 'text' key")
                
                if len(record["text"]) > 2000:
                    continue
                
                buckets[record["label"]].append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"  Line {lineno} skipped — {exc}")
                skipped += 1
        
    return dict(buckets)

# Write Json files
def write_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Main function
def main():
    rng = random.Random(SEED)

    # Load the corpus
    buckets = load_jsonl(file_path)

    print("class distribution in source corpus:")
    for label, records in sorted(buckets.items()):
        print(f"    {label:10s}: {len(records):,}")
    print(f"    {'TOTAL':10s}: {sum(len(v) for v in buckets.values()):,}")

    corpus_buckets: list[dict[str, list[dict]]] = [
        defaultdict(list) for _ in CORPUS_CONFIGS
    ]

    for label, records in sorted(buckets.items()):
        available = len(records)

        if available < TOTAL_PER_CLASS_NEEDED:
            print(f"Class '{label}': only {available:,} records available, "
                  f"need {TOTAL_PER_CLASS_NEEDED:,}.  Corpora sizes will be reduced proportionally.")

        pool   = rng.sample(records, min(TOTAL_PER_CLASS_NEEDED, available))
        cursor = 0
        for i, cfg in enumerate(CORPUS_CONFIGS):
            n      = min(cfg["per_class"], len(pool) - cursor)
            slice_ = pool[cursor: cursor + n]
            corpus_buckets[i][label].extend(slice_)
            cursor += n

    # Write out each corpus
    all_ids: list[set[int]] = []

    for i, cfg in enumerate(CORPUS_CONFIGS):
        out_path = processed_dir / cfg["filename"]

        records: list[dict] = [
            rec
            for label in sorted(corpus_buckets[i])
            for rec in corpus_buckets[i][label]
        ]
        rng.shuffle(records)

        write_jsonl(records, out_path)

        label_counts: dict[str, int] = defaultdict(int)
        for rec in records:
            label_counts[rec["label"]] += 1

        print(f"'{cfg['filename']}'  →  {out_path}")
        print(f" Total records : {len(records):,}")
        for label in sorted(label_counts):
            print(f"{label:10s}  : {label_counts[label]:,}")

        all_ids.append({id(rec) for rec in records})


if __name__ == "__main__":
    main()
