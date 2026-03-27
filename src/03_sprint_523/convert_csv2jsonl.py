import csv
import json
import os

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
    
    with open(jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
        # Based on annotation_tc.jsonl, it's a JSON array with trailing comma or just a list
        # Let's output it as a JSON array of objects to be safe and clear.
        # However, "JSONL" usually means Newline Delimited JSON.
        # The existing files are a bit weird (some are arrays, some are sections).
        # Let's check if the user wants JSONL (one per line) or the Array format seen in tc.
        # User said "jsonl format", but tc.jsonl is actually an array across lines.
        # I'll follow the tc.jsonl style since it's an individual annotator file.
        
        jsonl_file.write("[\n")
        for i, item in enumerate(data):
            json_str = json.dumps(item, ensure_ascii=False, indent=2)
            # Add indentation to the json_str
            indented_json = "\n".join("  " + line for line in json_str.split("\n"))
            jsonl_file.write(indented_json)
            if i < len(data) - 1:
                jsonl_file.write(",\n")
            else:
                jsonl_file.write("\n")
        jsonl_file.write("]\n")

if __name__ == "__main__":
    csv_input = "/Users/marco/Documents/GitHub/523GroupRepo/documentation/03_sprint/annotation/annotation_dz.csv"
    jsonl_output = "/Users/marco/Documents/GitHub/523GroupRepo/documentation/03_sprint/annotation/annotation_dz.jsonl"
    convert_csv_to_jsonl(csv_input, jsonl_output)
    print(f"Converted {csv_input} to {jsonl_output}")
