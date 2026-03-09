---
title: "annotation_explanation.md"
author: "Darwin Zhang"
date: "2026-03-08"
Disclaimer: This documentation is generated with the help of Gemini 3
---

## Overview

This document outlines the end-to-end workflow for transforming the raw Kaggle fraud corpus into our final, adjudicated set of "best" annotations. It details the sampling strategy, the diverse annotation formats used by the team, and the automated merging process, while addressing the specific technical and linguistic challenges encountered during the sprint.

## Step-by-Step Workflow

### 1. Data Acquisition and Preprocessing
We began by downloading the raw email dataset from Kaggle.
* **Action**: Initial cleaning and structure mapping.
* **Reference**: `src/01_sprint/preprocess.py`

### 2. Sub-Corpora Extraction
To make the vast amount of data manageable for human annotation, we extracted specific subsets from the primary corpus.
* **Action**: Implemented a stratified sampling algorithm to create balanced English and Chinese subsets.
* **Reference**: `src/03_sprint/build_subcorpora.py`

### 3. Annotation Phase
Each team member annotated a minimum of **50 records** following the guidelines established in our project documentation.
* **Action**: Manual labeling of `label`, `scenario`, `tactic_primary`, and `entities` (using character offsets).
* **Reference**: `documentation/03_sprint/annotation_tutorial.md`

### 4. Consolidation and Adjudication
We developed a robust Python utility to pool diverse annotation files into a single master record.
* **Action**: Parsed multiple formats (CSV, JSON, JSONL, and MD-embedded JSON), handled character encoding for Chinese text, and performed deduplication based on unique text strings.
* **Reference**: `src/03_sprint/best_annotations.py`
* **Final Output**: `documentation/03_sprint/annotation/final/annotations_best.jsonl`

## Challenges and Discussion

### Linguistic Quality of Chinese Translations
During the extraction of the Chinese sub-corpus, we encountered significant quality issues with the machine-translated text from the original Kaggle dataset.
* **Impact**: Many records contained nonsensical or poorly structured Chinese translations.
* **Resolution**: We opted to disregard a substantial portion of the Chinese records that did not meet a minimum threshold of readability to ensure the quality of our annotated gold standard remained high.

### Identifier Consistency
We initially intended to include unique `id` fields for every record in our JSONL files to simplify tracking across various versions of the corpus.
* **Challenge**: Several records lacked pre-existing IDs. Manually generating and maintaining consistent IDs across all individual annotation files proved to be an unexpected bottleneck.
* **Resolution**: To prioritize the completion of the linguistic annotations (entities and tactics), we decided to proceed without mandatory IDs for this sprint.

### Annotation Format Variability
A major hurdle in the final consolidation was the lack of a single, rigid annotation format across the team; members used a mix of CSVs and JSON structures.
* **Challenge**: The parser in `src/03_sprint/best_annotations.py` initially struggled with "jumps" in formatting, such as Markdown headers and missing JSON brackets.
* **Resolution**: We iterated on the script with AI assistance to create a more flexible parser. The final script uses a "greedy" JSON locator that finds array boundaries `[` and `]` within files and standardizes CSV columns into a uniform JSONL schema, capturing approximately 100+ unique entries.

### Storage Logic
To prevent the script from entering an infinite loop, where it would read its own output and flag every record as a duplicate, we established a dedicated `/final/` subdirectory.
* **Action**: The output is directed to `documentation/03_sprint/annotation/final/`, ensuring it is isolated from the raw source files in the parent directory.