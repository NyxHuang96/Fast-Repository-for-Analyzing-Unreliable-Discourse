import json
import random
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Setup tokenizers 
def bilingual_tokenizer(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return list(jieba.cut(text))
    else:
        return re.findall(r'\b\w+\b', text.lower())

# Helper functions to load jsonl files 
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    random.seed(581)
    
    sprint2_train_path = "data/processed/transfer_train.jsonl" # The 1104 samples from last sprint
    yh_path = "documentation/04_sprint_581/new_annotations_YH.jsonl"
    dev_path = "data/processed/validation.jsonl"
    output_path = "data/processed/bootstrapped_train.jsonl"
    
    sprint2_data = load_jsonl(sprint2_train_path)
    
    # Load data and handle to hide annotations (subset)
    yh_pool = []

    with open(yh_path, 'r', encoding='utf-8') as f:
        yh_raw = json.load(f)
        for key in yh_raw:
            yh_pool.extend(yh_raw[key])

    # 26 samples (25% of the 105 Gold annotated data)
    sample_size = 26
    bootstrap_subset = random.sample(yh_pool, sample_size)

    # Using best existing model (Traditional baseline)
    X_teacher = [d['text'] for d in sprint2_data]
    y_teacher = [d['label'].capitalize() for d in sprint2_data]
    
    teacher_model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=bilingual_tokenizer, ngram_range=(1, 2))),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])

    teacher_model.fit(X_teacher, y_teacher)

    # Inference and bootstrap (only extracting the texts)
    X_unseen = [d['text'] for d in bootstrap_subset] 
    pseudo_labels = teacher_model.predict(X_unseen)

    # Augmenting the dataset 
    augmented_train = list(sprint2_data)
    
    for i, text in enumerate(X_unseen):
        augmented_train.append({
            "text": f"{text} _source_gold_", 
            "label": pseudo_labels[i],
            "source": "bootstrapped_model_guess",
            "original_yh_label": bootstrap_subset[i]['label']
        })

    print(f"New training set size: {len(augmented_train)} records.")
    
    # Save the new dataset
    save_jsonl(augmented_train, output_path)

    # Re-train and evaluate the model 
    X_aug = [d['text'] for d in augmented_train]
    y_aug = [d['label'].capitalize() for d in augmented_train]
    
    student_model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=bilingual_tokenizer, ngram_range=(1, 2))),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])

    student_model.fit(X_aug, y_aug)

    # Evaluate on Dev set
    dev_data = load_jsonl(dev_path)
    
    # Add the gold switch to dev data for evaluation consistency
    X_dev = [f"{d['text']} _source_gold_" for d in dev_data] 
    y_dev = [d['label'].capitalize() for d in dev_data]
    
    y_pred = student_model.predict(X_dev)
    
    print("\n" + "="*40)
    print(" BOOTSTRAPPING EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_dev, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_dev, y_pred, target_names=["Ham", "Phish", "Spam"]))

if __name__ == "__main__":
    main()