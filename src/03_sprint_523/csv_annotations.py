"""
File used to create .csv annotations for undergraduate annotator
"""
import pandas as pd
import json
from itertools import islice

input_file = "data/processed/english_sub_corpus.json"
output_csv = "annotation_task_4500_4525.csv"

start_line = 4500
end_line = 4525 

data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in islice(f, start_line, end_line):
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)

df['scenario'] = ""
df['tactic_primary'] = ""
df['entities'] = ""
df['notes'] = ""

columns_order = ['label', 'text', 'scenario', 'tactic_primary', 'entities', 'notes']
df = df[columns_order]

# If want csv/xlsx
# df.to_csv(output_csv, index=False)
df.to_excel("annotation_task_4500_4525.xlsx", index=False)