import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from pathlib import Path

# Ensure NLTK resources are available
nltk.download('punkt')

# 1. Setup paths relative to your repo structure
script_dir = Path(__file__).resolve().parent
processed_dir = script_dir.parent.parent / "data" / "processed"
file_path = processed_dir / 'kaggle_corpus.json'

# 2. Load the dataset
print("Loading dataset...")
df = pd.read_json(file_path, lines=True)

# 3. Tokenize and calculate lengths
print("Tokenizing texts (this may take a minute for 72M tokens)...")
df['tokens'] = df['text'].apply(lambda x: word_tokenize(str(x).lower()))
df['doc_len'] = df['tokens'].apply(len)

# TTR analysis
def get_ttr(text_series):
    """Calculates Type-Token Ratio for a collection of texts."""
    all_tokens = []
    for text in text_series:
        # Tokenize and normalize to lowercase
        all_tokens.extend(word_tokenize(str(text).lower()))
    
    if not all_tokens:
        return 0
    # TTR = Unique Words / Total Words
    return len(set(all_tokens)) / len(all_tokens)

# 3. Apply the analysis to each label group
stats = df.groupby('label')['text'].apply(get_ttr)

print("\n--- Type-Token Ratio (Lexical Diversity) ---")
print(stats)

# Average Text Length 
avg_lens = df.groupby('label')['doc_len'].mean()
print("\n--- Part B: Average Text Length (in tokens) ---")
print(avg_lens)

#  N-gram Analysis
def get_top_ngrams_efficient(token_series, n, top_k=5):
    """Counts n-grams iteratively to save memory on M4 Mac."""
    counts = Counter()
    for tokens in token_series:
        # Generate n-grams and update the counter
        counts.update(ngrams(tokens, n))
    return counts.most_common(top_k)

print("\n--- Part D: Top 5 N-grams per Category ---")
for label in df['label'].unique():
    subset = df[df['label'] == label]['tokens']
    print(f"\n>>> Label: {label}")
    
    # Bigrams (n=2)
    bigrams_top = get_top_ngrams_efficient(subset, 2)
    print(f"  Top Bigrams: {bigrams_top}")
    
    # Trigrams (n=3)
    trigrams_top = get_top_ngrams_efficient(subset, 3)
    print(f"  Top Trigrams: {trigrams_top}")