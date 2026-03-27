import json
import os
import csv

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # The files were somewhat manually edited. Let's see if json.loads works
            return json.loads(content)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

files = {
    'mw': '/Users/marco/Documents/GitHub/523GroupRepo/documentation/03_sprint/Annotation/annotation_mw.jsonl',
    'yh': '/Users/marco/Documents/GitHub/523GroupRepo/documentation/03_sprint/Annotation/annotation_yh.jsonl',
    'tc': '/Users/marco/Documents/GitHub/523GroupRepo/documentation/03_sprint/Annotation/annotation_tc.jsonl'
}

data = {}
for annotator, filepath in files.items():
    d = load_json(filepath)
    data[annotator] = d

# We need to extract the first 10 entries for Chinese and first 10 for English.
# The user said "the first 10 entries of each language (Chinese, Mandarin)". They probably meant Chinese and English.
# Let's extract first 10 for both and combine them (total 20 entries) or do it separately.
# I'll just combine Chinese (1-10) and English (11-20).

fields = ['scenario', 'tactic_primary', 'entities']

# Prepare data structure to hold values:
# values[field][entry_id] = {annotator: value}
extracted = {field: {} for field in fields}

languages = ['Chinese', 'English']

for lang in languages:
    for idx in range(10):
        # entry_id from 1 to 20
        entry_id = f"{lang[:2]}_{idx+1:02d}"  # Ch_01, En_01 etc.
        for field in fields:
            extracted[field][entry_id] = {}
        for annotator in ['mw', 'yh', 'tc']:
            try:
                # Some files might not have exact 10, but let's assume they do.
                ann_data = data[annotator].get(lang, [])
                if idx < len(ann_data):
                    val = ann_data[idx].get(field)
                    # Handle null values
                    if val is None:
                        val = 'null'
                else:
                    val = 'MISSING'
                
                for field in fields:
                    extracted[field][entry_id][annotator] = data[annotator][lang][idx].get(field, 'null') if idx < len(data[annotator][lang]) else 'MISSING'
            except Exception as e:
                for field in fields:
                    extracted[field][entry_id][annotator] = 'ERROR'

# Output to iaa_study
output_file = '/Users/marco/Documents/GitHub/523GroupRepo/documentation/03_sprint/Annotation/iaa_study.raw'

with open(output_file, 'w', encoding='utf-8') as f:
    for field in fields:
        f.write(f"--- {field.upper()} ---\n")
        f.write("id\tannotator_1\tannotator_2\tannotator_3\n")
        for lang in languages:
            for idx in range(10):
                entry_id = f"{lang[:2]}_{idx+1:02d}"
                mw_val = str(extracted[field][entry_id]['mw']).replace('\n', ' ')
                yh_val = str(extracted[field][entry_id]['yh']).replace('\n', ' ')
                tc_val = str(extracted[field][entry_id]['tc']).replace('\n', ' ')
                f.write(f"{entry_id}\t{mw_val}\t{yh_val}\t{tc_val}\n")
        f.write("\n")

print(f"Successfully generated {output_file}")
