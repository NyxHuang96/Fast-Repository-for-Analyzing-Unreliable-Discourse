---
title: "build_subcorpora.md"
author: "Yusen Huang"
date: "2026-03-02"
Disclaimer: This documentation is generated with the help of Gemini3
---

## Overview

This document describes the sampling and extraction algorithm implemented in `src/03_sprint/build_subcorpora.py`. The script is designed to process a large-scale JSONL corpus and extract two specific, disjoint subsets—`english_sub_corpus.json` and `chinese_sub_corpus.json`—for future annotation tasks.

## Prerequisites

-   **json**: For parsing the primary corpus and serializing the extracted subsets.

-   **random**: For reproducible, seeded sampling of records.

-   **collections.defaultdict**: For efficient bucketing of records by label.

-   **pathlib**: For robust file path resolutionTop Tool Bar

## Step-by-Step Algorithm

### 1. Configuration & Path Resolution

The script defines two target sub-corpora with a stratified sampling goal:

-   **English Subset**: 16,667 records per class.

-   **Chinese Subset**: 3,333 records per class. It dynamically locates the source file `kaggle_corpus.json` within the `data/processed/` directory relative to the script location.

### 2. Streamed Loading & Filtering

The `load_jsonl` function reads the source file line-by-line to maintain a low memory footprint. It implements two critical checks:

-   **Integrity Check**: Verifies each JSON object contains the required `label` and `text` keys.

-   **Length Filter**: Automatically discards any record where the `text` field exceeds **2,000 characters** to ensure the subsets are suitable for human annotation.

### 3. Stratified Random Sampling

The script uses a fixed seed (`523`) to ensure reproducibility. For each label (Ham, Spam, Phish):

-   It pools all available records that passed the length filter.

-   It uses `rng.sample` to extract exactly 20,000 records (the sum of both sub-corpora needs).

-   If a class has fewer than 20,000 records, it proportionally reduces the slice sizes for both output files.

### 4. Disjoint Allocation

The extracted pool is sliced into two distinct segments:

-   The first **16,667** records are assigned to the English bucket.

-   The subsequent **3,333** records are assigned to the Chinese bucket. This slicing method ensures that no single record from the source corpus appears in both output files.

### 5. Serialization & Logging

Finally, the script:

-   Shuffles the final record lists to remove any ordering bias from the source file.

-   Writes the records to disk in JSONL format with `ensure_ascii=False` to support non-ASCII characters.

-   Logs the final class distribution and total record counts to the console for verification.
