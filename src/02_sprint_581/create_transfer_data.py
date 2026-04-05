import json
import random
from collections import defaultdict

def load_gold_train(filepath):
    """Loads the Gold training samples and adds the gold switch."""
    gold_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Grab text, handling potential missing keys
            en_text = data.get("text", "")
            zh_text = data.get("text_zh", "")
            
            # Adding _source_gold_ 
            combined_text = f"{en_text} {zh_text} _source_gold_".strip()
            
            gold_data.append({
                "text": combined_text,
                "label": data.get("label", "Unknown").capitalize()
            })

    return gold_data

def sample_silver_jsonl(filepath, sample_size=999):
    """Loads JSONL corpus (working on English sub corpus which had majority of te annotations),
    groups by class, samples evenly, and adds the silver switch."""
    records_by_class = defaultdict(list)
    
    # Load corpus 
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            label = data.get("label", "Unknown").capitalize()
            text = data.get("text", "")
            
            # Add if it has text 
            # Fail safe as we have 50k entries and may have some not in UTF-8 
            if text:
                records_by_class[label].append(text)
                
    # Sample evenly across Ham, Spam, and Phish
    samples_per_class = sample_size // 3
    silver_data = []
    
    for target_label in ["Ham", "Spam", "Phish"]:
        population = records_by_class.get(target_label, [])
        
        # If we do not have enough, which we have 50k instances, then we sample with replacement to ensure balance
        if len(population) >= samples_per_class:
            chosen_texts = random.sample(population, samples_per_class)
        else:
            print(f"Warning: Only found {len(population)} items for {target_label}. Sampling with replacement.")
            chosen_texts = random.choices(population, k=samples_per_class)
            
        # Add the Magic Switch: _source_silver_
        for text in chosen_texts:
            silver_data.append({
                "text": f"{text} _source_silver_".strip(),
                "label": target_label
            })
            
    return silver_data

def main():
    # Paths
    gold_path = "data/processed/train.jsonl"
    silver_path = "data/processed/english_sub_corpus.json"
    output_path = "data/processed/transfer_train.jsonl"
    
    random.seed(42)
    
    # Process Data
    gold_records = load_gold_train(gold_path)
    print(f"Loaded {len(gold_records)} Gold records.")
    
    silver_records = sample_silver_jsonl(silver_path, sample_size=999)
    print(f"Loaded {len(silver_records)} Silver records.")
    
    # Combine and Shuffle
    transfer_dataset = gold_records + silver_records
    random.shuffle(transfer_dataset)
    
    # Save to new JSONL
    print(f"Saving {len(transfer_dataset)} combined records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in transfer_dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()