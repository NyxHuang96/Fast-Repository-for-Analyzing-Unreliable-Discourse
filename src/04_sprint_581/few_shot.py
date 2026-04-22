import json
import numpy as np
import random
from active_learning import load_jsonl, save_jsonl, train_and_eval, build_pipeline, print_section

def load_json_array(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    bootstrapped_path = 'documentation/04_sprint_581/bootstrapped_train.jsonl'
    al_ranked_path = 'documentation/04_sprint_581/al_ranked.jsonl'
    augmented_path = 'documentation/04_sprint_581/augmented_data.jsonl'
    dev_path = 'data/processed/validation.jsonl'
    few_shot_output_path = 'documentation/04_sprint_581/few_shot_train.jsonl'

    # Load base train (excluding bootstrapped model guesses)
    all_bootstrapped_train = load_jsonl(bootstrapped_path)
    base_train = [d for d in all_bootstrapped_train if d.get('source') != 'bootstrapped_model_guess']

    # Load AL ranked data (the 26 samples with gold labels)
    al_data = load_jsonl(al_ranked_path)

    # Combine to form the dataset that is "25% larger than last block"
    extended_dataset = base_train + al_data
    print(f"Base training set: {len(base_train)} samples")
    print(f"Active Learning additions: {len(al_data)} samples")
    print(f"Current extended dataset size: {len(extended_dataset)} samples")

    # Load few-shot augmented data
    # Note: augmented_data.jsonl is formatted as a JSON array
    try:
        few_shot_data = load_json_array(augmented_path)
    except json.JSONDecodeError:
        # Fallback if it was reformatted to JSONL
        few_shot_data = load_jsonl(augmented_path)
    
    # Add a source tag
    for item in few_shot_data:
        item['source'] = 'few_shot_augmented'

    print(f"Few-shot augmented samples: {len(few_shot_data)} samples")

    # Final combined dataset
    final_dataset = extended_dataset + few_shot_data
    print(f"Final few-shot dataset size: {len(final_dataset)} samples")

    # Save to few_shot_train.jsonl
    save_jsonl(final_dataset, few_shot_output_path)
    print(f"Saved final dataset to {few_shot_output_path}")

    # Evaluate
    dev_data = load_jsonl(dev_path)

    print_section('EVALUATION: EXTENDED DATASET (Before Few-Shot)')
    _, ext_acc, ext_report = train_and_eval(extended_dataset, dev_data)
    print(f'Accuracy: {ext_acc:.4f}')
    print(ext_report)

    print_section('EVALUATION: FINAL FEW-SHOT DATASET (After Few-Shot)')
    _, fs_acc, fs_report = train_and_eval(final_dataset, dev_data)
    print(f'Accuracy: {fs_acc:.4f}')
    print(fs_report)

if __name__ == '__main__':
    main()
