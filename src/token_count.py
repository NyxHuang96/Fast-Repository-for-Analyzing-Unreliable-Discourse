"""
File: token_count.py
Author: Yusen Huang
Date: 2026-02-25
Course: COLX 523
Description: Calculate the tokens after processing text with nltk tokenizer
"""
import json
import nltk
from pathlib import Path

# You must download the 'punkt' resource the first time you use NLTK tokenizers
nltk.download('punkt')

script_dir = Path(__file__).resolve().parent
processed_dir = script_dir.parent.parent / "data" / "processed"
file_path = processed_dir / 'kaggle_corpus.json'

total_nltk_tokens = 0

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Load the JSON object
        data = json.loads(line)
        text = data.get("text", "")
        
        # Use NLTK's recommended word tokenizer
        tokens = nltk.word_tokenize(text)
        
        # Add the count of tokens for this line to the total
        total_nltk_tokens += len(tokens)

print(f"Total Tokens (NLTK): {total_nltk_tokens:,}")