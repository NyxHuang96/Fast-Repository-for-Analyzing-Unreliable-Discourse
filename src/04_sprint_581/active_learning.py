import json
import math
import random
import re
import numpy as np
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def bilingual_tokenizer(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return list(jieba.cut(text))
    return re.findall(r'\b\w+\b', text.lower())


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=bilingual_tokenizer, ngram_range=(1, 2))),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])


def train_and_eval(train_data, dev_data, label_key='label'):
    X_train = [d['text'] for d in train_data]
    y_train = [d[label_key].capitalize() for d in train_data]
    X_dev = [d['text'] for d in dev_data]
    y_dev = [d['label'].capitalize() for d in dev_data]

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)

    acc = accuracy_score(y_dev, y_pred)
    report = classification_report(y_dev, y_pred, target_names=["Ham", "Phish", "Spam"])
    return model, acc, report


# ── Query-by-Uncertainty (margin sampling) ──────────────────────────────────

def margin_scores(model, texts):
    """Lower margin = more uncertain = higher AL priority."""
    decision = model.decision_function(texts)
    sorted_scores = np.sort(decision, axis=1)[:, ::-1]
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    return margins


# ── Query-by-Committee (entropy of disagreement) ────────────────────────────

def committee_entropy_scores(base_train, texts, n_models=5, sample_rate=0.9, seed=42):
    """Higher entropy = more disagreement = higher AL priority."""
    rng = random.Random(seed)
    all_preds = []

    for i in range(n_models):
        k = max(1, int(len(base_train) * sample_rate))
        subset = rng.sample(base_train, k)
        X = [d['text'] for d in subset]
        y = [d['label'].capitalize() for d in subset]
        m = build_pipeline()
        m.fit(X, y)
        all_preds.append(m.predict(texts))

    classes = ["Ham", "Phish", "Spam"]
    entropies = []
    for i in range(len(texts)):
        votes = [pred[i] for pred in all_preds]
        counts = Counter(votes)
        probs = np.array([counts.get(c, 0) / n_models for c in classes])
        # add smoothing to avoid log(0)
        probs = (probs + 1e-9) / probs.sum()
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)

    return np.array(entropies)


def accumulation_test(base_train, al_ranked, dev_data, percentages):
    """
    Retrain with top p% of AL-ranked samples (with gold labels) + base_train.
    al_ranked: list of bootstrapped entries sorted by AL priority (highest first),
               with 'label' already set to gold label.
    """
    results = {}
    for pct in percentages:
        n = max(1, math.ceil(len(al_ranked) * pct))
        selected = al_ranked[:n]
        combined = base_train + selected
        _, acc, report = train_and_eval(combined, dev_data)
        results[pct] = {'n_selected': n, 'accuracy': acc, 'report': report}
    return results


def print_section(title):
    print('\n' + '=' * 50)
    print(f'  {title}')
    print('=' * 50)


def main():
    random.seed(42)
    np.random.seed(42)

    bootstrapped_path = 'documentation/04_sprint_581/bootstrapped_train.jsonl'
    dev_path = 'data/processed/validation.jsonl'
    al_output_path = 'documentation/04_sprint_581/al_ranked.jsonl'

    all_train = load_jsonl(bootstrapped_path)
    dev_data = load_jsonl(dev_path)

    # Split into original train and bootstrapped samples
    base_train = [d for d in all_train if d.get('source') != 'bootstrapped_model_guess']
    bootstrapped = [d for d in all_train if d.get('source') == 'bootstrapped_model_guess']

    print(f'Base training set: {len(base_train)} samples')
    print(f'Bootstrapped samples: {len(bootstrapped)} samples')

    # ── Step 1: train model on base data ────────────────────────────────────
    print_section('BASELINE (base train only)')
    _, base_acc, base_report = train_and_eval(base_train, dev_data)
    print(f'Accuracy: {base_acc:.4f}')
    print(base_report)

    # re-train to get model object for scoring
    base_model = build_pipeline()
    base_model.fit(
        [d['text'] for d in base_train],
        [d['label'].capitalize() for d in base_train]
    )

    # ── Step 2: AL scoring ───────────────────────────────────────────────────
    X_boot = [d['text'] for d in bootstrapped]

    # Method 1: Query-by-Uncertainty (margin)
    margins = margin_scores(base_model, X_boot)
    qbu_order = np.argsort(margins)           # ascending: smallest margin first

    # Method 2: Query-by-Committee (entropy)
    entropies = committee_entropy_scores(base_train, X_boot)
    qbc_order = np.argsort(-entropies)        # descending: highest entropy first

    print_section('AL SCORES (bootstrapped samples)')
    print(f"{'#':<4} {'Margin':>8} {'Entropy':>8}  {'Pseudo':>6} {'Gold':>6}  Text[:60]")
    print('-' * 80)
    for idx in range(len(bootstrapped)):
        d = bootstrapped[idx]
        text_preview = d['text'][:60].replace('\n', ' ')
        print(f"{idx:<4} {margins[idx]:>8.4f} {entropies[idx]:>8.4f}  "
              f"{d['label']:>6} {d['original_yh_label']:>6}  {text_preview}")

    # ── Step 3: build gold-labeled ranked lists ──────────────────────────────
    def make_gold(entry):
        """Return copy with pseudo label replaced by gold label."""
        gold = dict(entry)
        gold['label'] = entry['original_yh_label']
        gold['source'] = 'active_learning_gold'
        return gold

    qbu_ranked = [make_gold(bootstrapped[i]) for i in qbu_order]
    qbc_ranked = [make_gold(bootstrapped[i]) for i in qbc_order]

    # Save QBU ranked list
    save_jsonl(qbu_ranked, al_output_path)
    print(f'\nAL-ranked samples saved to {al_output_path}')

    # ── Step 4: accumulation test ────────────────────────────────────────────
    percentages = [0.05, 0.10, 0.15, 0.20]

    print_section('ACCUMULATION TEST — Query-by-Uncertainty (Margin Sampling)')
    print(f"{'%':<6} {'N':>4}  {'Accuracy':>9}")
    print('-' * 25)
    qbu_results = accumulation_test(base_train, qbu_ranked, dev_data, percentages)
    for pct, res in qbu_results.items():
        print(f"{pct*100:>5.0f}% {res['n_selected']:>4}  {res['accuracy']:>9.4f}")

    print()
    for pct, res in qbu_results.items():
        print(f'\n--- QBU Top {pct*100:.0f}% (n={res["n_selected"]}) ---')
        print(res['report'])

    print_section('ACCUMULATION TEST — Query-by-Committee (Entropy of Disagreement)')
    print(f"{'%':<6} {'N':>4}  {'Accuracy':>9}")
    print('-' * 25)
    qbc_results = accumulation_test(base_train, qbc_ranked, dev_data, percentages)
    for pct, res in qbc_results.items():
        print(f"{pct*100:>5.0f}% {res['n_selected']:>4}  {res['accuracy']:>9.4f}")

    print()
    for pct, res in qbc_results.items():
        print(f'\n--- QBC Top {pct*100:.0f}% (n={res["n_selected"]}) ---')
        print(res['report'])

    # ── Step 5: full bootstrapped baseline (pseudo labels) ──────────────────
    print_section('FULL BOOTSTRAPPED BASELINE (all 26 pseudo labels)')
    _, boot_acc, boot_report = train_and_eval(all_train, dev_data)
    print(f'Accuracy: {boot_acc:.4f}')
    print(boot_report)


if __name__ == '__main__':
    main()
