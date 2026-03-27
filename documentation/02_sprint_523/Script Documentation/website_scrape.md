# Reddit Website Scraping

**Author:** Darwin Zhang
**Date:** 2026-02-27
**Course:** COLX 523

## Overview

This document outlines the tri-modal data extraction and heuristic filtering algorithm implemented in `src/website_scrape.py` during Sprint 2. The script is designed to politely query Reddit's JSON API, extract multi-modal fraud data (Title, Body Text, and Image OCR), and apply a heuristic regex filter to remove user meta-noise before saving the clean data for manual annotation.

## Prerequisites

- **requests**: Used for making HTTP requests to Reddit's JSON endpoints and downloading high-resolution image bytes.
- **pytesseract**: An optical character recognition (OCR) wrapper for Python. Used to extract raw fraud text from image attachments. *Note: Requires the Tesseract engine to be installed on the host operating system.*
- **Pillow (PIL)**: Used to open and manipulate downloaded image bytes in-memory before passing them to the OCR engine.
- **jsonlines**: Used for efficient, memory-safe line-by-line writing of the JSONL corpus format.
- **re**: The standard regular expressions library, used to identify structural patterns of meta-noise and cries for help in the text.

## Step-by-Step Algorithm

### 1. API Initialization & Rate Limiting
The script initializes the `RedditScraper` with a custom, professional `User-Agent` to comply with Reddit's API guidelines and prevent immediate 429 (Too Many Requests) blocks. It implements a polite delay using `time.sleep(2)` between network calls.

### 2. JSON Feed Extraction
Rather than parsing brittle HTML, the script targets specific subreddits (e.g., `r/phishing`) by appending `.json` to the URL. It iterates through the returned nested dictionary structure to isolate the `data` payload of each individual post.

### 3. Tri-Modal Data Mapping
For every post in the payload, the script:
- Maps the `title` to the `victim_title` field.
- Maps the `selftext` to the `victim_body` field.
- Detects if a direct image URL exists (ending in `.jpg`, `.jpeg`, or `.png`).

### 4. In-Memory OCR Processing
If an image URL is detected, the script fetches the image into a `BytesIO` stream. It uses `Pillow` to open the image and `pytesseract.image_to_string()` to extract the text dynamically. This happens entirely in memory, avoiding the need to write temporary image files to the local hard drive. The extracted string is saved to the `ocr_text` field.

### 5. Heuristic Noise Filtering
The script instantiates the `NoiseFilter` class, which uses compiled regex patterns to scan the title and the first 200 characters of the body text for meta-questions (e.g., "Is this a scam?", "What do I do?"). If a match is found, the script categorizes the post as a "cry for help" rather than raw fraud data and drops the record to save downstream annotation time.

### 6. JSONL Export
The script uses `jsonlines` to stream the surviving, cleaned records directly into the `data/raw/reddit_processed.jsonl` file, ensuring the output directory is created if it does not already exist.

## Output

Executing the `website_scrape.py` script yields:
- Console status updates indicating the number of raw posts fetched, the progress of the OCR processing, and the heuristic filtering statistics (Clean records retained vs. Meta-noise dropped).
- A final `reddit_processed.jsonl` file containing the structured, tri-modal fraud data ready for manual team review.