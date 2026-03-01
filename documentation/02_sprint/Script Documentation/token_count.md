# Token Counting Script Documentation

**Author:** Yusen Huang

**Date:** 2026-02-25

**Course:** COLX 523

## Overview

This document describes the tokenization and counting algorithm implemented in `src/02_sprint/token_count.py`. The script processes the cleaned dataset to calculate the total number of linguistic units (tokens) using a professional-grade tokenizer, providing a more accurate metric for text volume than simple character or whitespace counts.

## Prerequisites

-   **json**: Used for parsing the processed JSONL corpus.

-   **nltk**: The Natural Language Toolkit, specifically the `punkt` tokenizer models, used for sophisticated word segmentation.

-   **pathlib**: Used for robust, cross-platform file path management.

## Step-by-Step Algorithm

### 1. Resource Initialization

The script ensures the necessary linguistic resources are available by calling `nltk.download('punkt')`. This downloads the pre-trained unsupervised machine learning models required for NLTK's tokenization logic.

### 2. Path Resolution

The script uses `pathlib.Path` to dynamically locate the data relative to the script's execution point:

-   **`script_dir`**: Identifies the current directory of the script.

-   **`processed_dir`**: Navigates to the `data/processed` directory relative to the repository root.

-   **`file_path`**: Targets the specific corpus file (`kaggle_corpus.json`) using the `/` operator for path joining.

### 3. Streamed File Processing

To maintain memory efficiency while handling a potentially large Kaggle corpus, the script opens the file and iterates through it line-by-line. This avoids loading the entire dataset into RAM simultaneously.

### 4. JSON Extraction

For every line in the file, the script:

-   Parses the line into a Python dictionary using `json.loads()`.

-   Extracts the content of the `"text"` field.

### 5. NLTK Tokenization

The script applies `nltk.word_tokenize(text)` to the extracted content. Unlike a basic whitespace split, this algorithm:

-   Separates punctuation from words (e.g., "end." becomes `["end", "."]`).

-   Handles contractions and currency symbols intelligently.

-   Converts the raw string into a list of individual tokens.

### 6. Aggregate Calculation

The script maintains a running total, `total_nltk_tokens`, incrementing it by the length of the token list generated for each record.

## Output

Executing the `token_count.py` script yields:

-   A download status message for the `punkt` package.

-   A final console output displaying the **Total Tokens (NLTK)**, formatted with commas for readability.
