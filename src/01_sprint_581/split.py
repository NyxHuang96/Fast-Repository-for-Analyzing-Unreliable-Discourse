"""
File: split.py
Author: Tianhao Cao
Date: 2026-03-26
Last Updated: 2026-03-26
Course: COLX 523
Description: Split the dataset into train, validation, and test sets.
"""

import json
import random
import os
from collections import defaultdict


def load_data(file_path):
    """
    Robust data loader for JSON/JSONL formats.
    It can handle standard JSONL or the indented comma-separated format found in our annotations_best.jsonl.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("{") and (
            content.endswith("}") or content.endswith("},")
        ):
            content = "[" + content.rstrip(",") + "]"
        return json.loads(content)

    # Fallback to standard JSON Lines
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    # seed
    random_seed = 581
    random.seed(random_seed)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "04_sprint_523",
            "web",
            "backend",
            "corpus_data",
            "annotated",
            "annotations_best.jsonl",
        )
    )
    output_dir = os.path.normpath(
        os.path.join(script_dir, "..", "..", "data", "processed")
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {input_file}")
    data = load_data(input_file)

    if not data:
        print("Error: Failed to load data or the dataset is empty.")
        return

    print(f"Total instances loaded: {len(data)}")

    # 2. Deduplicate by text to prevent data leakage
    seen_texts = set()
    unique_data = []
    for item in data:
        text = item.get("text", "")
        if text not in seen_texts:
            seen_texts.add(text)
            unique_data.append(item)

    num_duplicates = len(data) - len(unique_data)
    if num_duplicates > 0:
        print(f"Removed {num_duplicates} duplicate texts to prevent data leakage.")
    print(f"Unique instances for splitting: {len(unique_data)}")

    # 3. Consistent class distributions
    label_groups = defaultdict(list)
    for item in unique_data:
        label = item.get("label", "Unknown")
        label_groups[label].append(item)

    train_data = []
    dev_data = []
    test_data = []

    # Splitting 80-10-10 inside each label group
    for label, items in label_groups.items():
        # Shuffle within each class with the fixed seed
        random.shuffle(items)

        n = len(items)
        n_train = int(n * 0.8)
        n_dev = int(n * 0.1)

        train_data.extend(items[:n_train])
        dev_data.extend(items[n_train : n_train + n_dev])
        test_data.extend(items[n_train + n_dev :])

    # Shuffle the final subsets to interleave the labels
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    # 3. Validating the splits
    def check_leakage(train, dev, test):
        train_texts = set(item.get("text", "") for item in train)
        dev_texts = set(item.get("text", "") for item in dev)
        test_texts = set(item.get("text", "") for item in test)

        train_dev_overlap = train_texts.intersection(dev_texts)
        train_test_overlap = train_texts.intersection(test_texts)
        dev_test_overlap = dev_texts.intersection(test_texts)

        if train_dev_overlap or train_test_overlap or dev_test_overlap:
            print("\nWARNING: Data leakage detected (duplicate texts across splits)!")
            print(f"Train/Dev overlaps: {len(train_dev_overlap)}")
            print(f"Train/Test overlaps: {len(train_test_overlap)}")
            print(f"Dev/Test overlaps: {len(dev_test_overlap)}")
        else:
            print(
                "\nLeakage check passed: No exact text overlaps across train/dev/test sets."
            )

    check_leakage(train_data, dev_data, test_data)

    # Save to Disk
    def save_jsonl(filename, dataset):
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(dataset)} items to {output_path}")

    print("\n--- Saving datasets ---")
    save_jsonl("train.jsonl", train_data)
    save_jsonl("validation.jsonl", dev_data)
    save_jsonl("test.jsonl", test_data)

    # Output Split Statistics
    def print_distribution(name, dataset):
        counts = defaultdict(int)
        for item in dataset:
            counts[item.get("label", "Unknown")] += 1
        print(f"\n{name} split distribution ({len(dataset)} items):")
        for label, count in counts.items():
            percentage = (count / len(dataset) * 100) if len(dataset) > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")

    print_distribution("Train", train_data)
    print_distribution("Validation", dev_data)
    print_distribution("Test", test_data)

    print("\nData splitting completed successfully.")


if __name__ == "__main__":
    main()
