---
title: "split.md"
author: "Yusen Huang"
date: "2026-03-28"
Disclaimer: This documentation is generated with the help of Claude
---

## Overview

This document describes the data splitting algorithm implemented in `split.py`. The script is designed to process an annotated JSONL corpus and produce three disjoint subsets—`train.jsonl`, `validation.jsonl`, and `test.jsonl`—for use in downstream model training and evaluation tasks.

## Prerequisites

-   **json**: For parsing the annotated corpus and serializing the output splits.

-   **random**: For reproducible, seeded shuffling and splitting of records.

-   **os**: For file path resolution and directory creation.

-   **collections.defaultdict**: For efficient grouping of records by class label.

## Step-by-Step Algorithm

### 1. Configuration & Path Resolution

The script sets a fixed random seed (`581`) to ensure reproducibility across runs. It dynamically resolves the input file path to `04_sprint_523/web/backend/corpus_data/annotated/annotations_best.jsonl` and the output directory to `data/processed/`, both relative to the script location. The output directory is created if it does not already exist.

### 2. Robust Data Loading

The `load_data` function reads the source file and handles two possible formats:

-   **Comma-separated JSON objects**: If the file content starts with `{` and ends with `}` or `},`, the loader wraps the content in brackets and parses it as a JSON array.

-   **Standard JSON Lines**: As a fallback, the function reads line-by-line, parsing each non-empty line as an individual JSON object.

### 3. Deduplication

Before splitting, the script removes duplicate records based on the `text` field to prevent data leakage between splits. It iterates through all loaded records, retaining only the first occurrence of each unique text. The number of removed duplicates is logged to the console.

### 4. Stratified Splitting

To maintain consistent class distributions across all three subsets, the script groups records by their `label` field. Within each label group:

-   Records are shuffled using the fixed seed.

-   **80%** of records are allocated to the training set.

-   **10%** are allocated to the validation (dev) set.

-   The remaining **10%** are allocated to the test set.

After allocation, each of the three final subsets is shuffled again to interleave records from different classes.

### 5. Leakage Validation

The `check_leakage` function performs a post-split integrity check by computing the text-level intersection between all three split pairs (train/dev, train/test, dev/test). If any overlapping texts are detected, a warning is printed with the overlap counts. Otherwise, a confirmation message is logged.

### 6. Serialization & Logging

Finally, the script:

-   Writes each subset to disk in JSONL format with `ensure_ascii=False` to support non-ASCII characters.

-   Logs the number of items saved and the output file path for each split.

-   Prints the per-class distribution (count and percentage) for each split to the console for verification.
