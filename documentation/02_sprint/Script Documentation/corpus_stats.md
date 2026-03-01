Corpus Statistical Analysis Documentation

**Author:** Yusen Huang

**Date:** February 28, 2026

**Course:** COLX 523

## Overview

This document describes the statistical analysis algorithm implemented in `src/02_sprint/corpus_stats.py`. The script performs a deep quantitative dive into the cleaned email corpus, calculating lexical diversity, document length metrics, and N-gram distributions across **Ham**, **Spam**, and **Phish** categories.

## Prerequisites

-   **pandas**: Used for high-level data manipulation and group-based aggregation.

-   **nltk**: Used for tokenization (`word_tokenize`) and N-gram generation (`ngrams`).

-   **collections.Counter**: Utilized for memory-efficient frequency tracking of word sequences.

-   **pathlib**: Used for managing cross-platform file paths within the repository.

## Step-by-Step Algorithm

### 1. Environment & Path Setup

The script initializes the NLTK `punkt` resource and resolves the absolute path to the processed dataset. It navigates from the script's location to `data/processed/kaggle_corpus.json` using `pathlib.Path` to ensure compatibility with directory structure.

### 2. Dataset Loading

The corpus is loaded into a pandas DataFrame using `pd.read_json(file_path, lines=True)`. This loads the JSONL format into a structured table, where each row represents an email and columns represent the `label` and `text`.

### 3. Feature Engineering (Tokenization & Length)

The script performs a global tokenization pass:

-   **Lowercasing**: All text is normalized to lowercase to ensure statistical consistency.

-   **Tokenization**: Uses `nltk.word_tokenize` to segment text into linguistic units.

-   **Length Calculation**: Computes the token count for every document, storing it in a new `doc_len` column.

### 4. Lexical Diversity (Type-Token Ratio)

The `get_ttr` function calculates the **Type-Token Ratio (TTR)** for each label group:

-   It aggregates all tokens for a specific category.

-   **Formula**: $TTR = \frac{\text{Unique Tokens (Types)}}{\text{Total Tokens}}$

-   This metric determines the vocabulary richness of each email category.

### 5. Central Tendency (Average Length)

The script groups the data by the `label` metadata and calculates the mean of the `doc_len` column. This provides insights into the typical message volume for legitimate vs. malicious correspondence.

### 6. Memory-Efficient N-gram Analysis

To handle the massive **72.6 million tokens** without exhausting the M4 chip's unified memory, the script implements an iterative counting approach using `collections.Counter`:

-   **Bigrams (n=2)**: Identifies the top 5 most frequent two-word sequences per label.

-   **Trigrams (n=3)**: Identifies the top 5 most frequent three-word sequences per label.

-   The iterative `update` method ensures that the script only stores the frequency counts rather than a massive list of all generated N-grams.

## Output

Executing the script provides a comprehensive statistical summary in the console:

-   **Type-Token Ratio Table**: A comparison of lexical diversity across Ham, Phish, and Spam.

-   **Average Text Length**: The mean token count for each category.

-   **Categorical N-gram Lists**: The most common phrases defining the linguistic "signatures" of each email type.
