import json
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


def load_jsonl(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Combine English and Chinese text if both exist, or just grab what's there
            en_text = data.get("text", "")
            zh_text = data.get("text_zh", "")
            combined_text = f"{en_text} {zh_text}".strip()
            
            texts.append(combined_text)
            labels.append(data.get("label", "Unknown").capitalize())
            
    return texts, labels


def bilingual_tokenizer(text):
    """
    Routes text through Jieba if Chinese characters are detected, 
    otherwise uses standard English word tokenization.
    """
    # Regex to check for Chinese characters
    if re.search(r'[\u4e00-\u9fff]', text):
        # Use Jieba for Chinese (it also handles mixed English decently)
        return list(jieba.cut(text))
    else:
        # Lowercase and extract English words/numbers
        return re.findall(r'\b\w+\b', text.lower())


def main():
    print("Loading data splits...")
    # train_path = "data/processed/train.jsonl"

    # Edited for Sprint 2 and testing with silver labels 
    train_path = "data/processed/transfer_train.jsonl"
    dev_path = "data/processed/validation.jsonl"
    
    # Code for Sprint 1 
    X_train, y_train = load_jsonl(train_path)
    X_dev, y_dev = load_jsonl(dev_path)

    # Sprint 2: Add the Gold switch to the evaluation data
    X_dev = [f"{text} _source_gold_" for text in X_dev]
    
    print(f"Training on {len(X_train)} samples, Evaluating on {len(X_dev)} samples.\n")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=bilingual_tokenizer, 
            token_pattern=None,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )),
        ('clf', LinearSVC(random_state=42, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)    
    y_pred = pipeline.predict(X_dev)
    
    print("Classificaiton")
    print(classification_report(y_dev, y_pred, target_names=["Ham", "Phish", "Spam"]))
    
    print("Confusion Matrix")
    cm = confusion_matrix(y_dev, y_pred, labels=["Ham", "Phish", "Spam"])
    print(f"       Pred_Ham Pred_Phish Pred_Spam")
    print(f"True_Ham   {cm[0][0]:2d}       {cm[0][1]:2d}         {cm[0][2]:2d}")
    print(f"True_Phish {cm[1][0]:2d}       {cm[1][1]:2d}         {cm[1][2]:2d}")
    print(f"True_Spam  {cm[2][0]:2d}       {cm[2][1]:2d}         {cm[2][2]:2d}")

if __name__ == "__main__":
    main()