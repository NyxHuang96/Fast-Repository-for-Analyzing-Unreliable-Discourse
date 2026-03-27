## This script here is used for annotators who have 3 sheets present in their xlsx 
import pandas as pd
import json

def convert_excel_to_jsonl(excel_path, jsonl_path, sheet_name="Annotations"):
    # Read specifically sheet 3 (named "Annotations")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Replace all NaN/empty cells with empty strings to avoid errors
    df = df.fillna('')
    data = []

    for index, row in df.iterrows():
        tactic = str(row.get('tactic_primary', '')).strip()
        if not tactic or tactic.lower() == 'null':
            tactic = None
        
        entities = str(row.get('entities', '')).strip()
        if entities.lower() == 'null':
            entities = ""
        
        notes = str(row.get('notes', '')).strip()
        if notes.lower() == 'null':
            notes = ""

        obj = {
            "label": str(row.get('label', '')).strip(),
            "text": str(row.get('text', '')).strip(),
            "text_zh": "", # English-only source
            "scenario": str(row.get('scenario', '')).strip(),
            "tactic_primary": tactic,
            "entities": entities,
            "notes": notes
        }
        data.append(obj)
    
    # Write using your preferred array format
    with open(jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
        jsonl_file.write("[\n")
        for i, item in enumerate(data):
            json_str = json.dumps(item, ensure_ascii=False, indent=2)
            indented_json = "\n".join("  " + line for line in json_str.split("\n"))
            jsonl_file.write(indented_json)
            if i < len(data) - 1:
                jsonl_file.write(",\n")
            else:
                jsonl_file.write("\n")
        jsonl_file.write("]\n")

if __name__ == "__main__":
    # Point this to Cassie's Excel file
    
    excel_input = "documentation/03_sprint/annotation/raw/cassie-annotations-return.xlsx"
    # Output it alongside the other annotation files so the merge script catches it
    jsonl_output = "documentation/03_sprint/annotation/cassie_gold.jsonl"
    
    convert_excel_to_jsonl(excel_input, jsonl_output)
    print(f"Converted {excel_input} to {jsonl_output}")