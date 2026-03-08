
"""
File: best_annotations.py
Author: Darwin Zhang
Date: 2026-03-08
Course: COLX 523
Description: Script to merge annotation files from CSV and JSONL formats into a single JSONL file.
The output will be properly formatted with indentation for readability.

Note: assisted with free tier of Claude to find parses that work for our structure.
"""

import json
import csv
import os
from pathlib import Path


def read_csv_annotations(csv_file):
    """
    Read annotations from a CSV file and convert to JSON format.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotation = {
                "label": row.get("label", ""),
                "text": row.get("text", ""),
                "text_zh": row.get("text_zh", ""),
                "scenario": row.get("scenario", ""),
                "tactic_primary": row.get("tactic_primary") if row.get("tactic_primary") else None,
                "entities": row.get("entities", ""),
                "notes": row.get("notes", "")
            }
            annotations.append(annotation)
    
    return annotations


def read_jsonl_annotations(jsonl_file):
    """
    Read annotations from a JSONL file.
    Handles various formats:
    - Standard JSONL (one compact JSON per line)
    - Pretty-formatted JSON with indentation
    - JSON arrays
    - Comma-separated objects
    
    Args:
        jsonl_file: Path to the JSONL file
        
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        if not content:
            return annotations
        
        # Strategy 1: Try parsing as a JSON array (wrapped in [])
        if content.startswith('[') and content.endswith(']'):
            try:
                annotations = json.loads(content)
                return annotations
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse as JSON array: {e}")
        
        # Strategy 2: Try wrapping comma-separated objects in brackets
        # This handles pretty-formatted JSON objects separated by commas
        # Look for pattern: } followed by comma and newline and {
        if '},\n' in content or '},\r\n' in content:
            # Wrap in array brackets
            content_array = '[' + content + ']'
            try:
                annotations = json.loads(content_array)
                print(f"  Successfully parsed as comma-separated pretty-formatted JSON objects")
                return annotations
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse as comma-separated objects: {e}")
        
        # Strategy 3: Try parsing as standard JSONL (one compact JSON object per line)
        # Split into lines and try to parse each as JSON
        lines = content.split('\n')
        current_obj = ""
        brace_count = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines when not building an object
            if not stripped and brace_count == 0:
                continue
            
            # Add line to current object being built
            current_obj += line + '\n'
            
            # Count braces to determine when we have a complete object
            for char in stripped:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            # When brace count returns to 0, we have a complete object
            if brace_count == 0 and current_obj.strip():
                # Remove trailing comma if present
                obj_to_parse = current_obj.strip()
                if obj_to_parse.endswith(','):
                    obj_to_parse = obj_to_parse[:-1]
                
                try:
                    annotation = json.loads(obj_to_parse)
                    annotations.append(annotation)
                    current_obj = ""
                except json.JSONDecodeError as e:
                    print(f"  Warning: Could not parse object ending at line {line_num}: {e}")
                    print(f"  Content: {obj_to_parse[:200]}...")
                    current_obj = ""
    
    return annotations


def write_formatted_jsonl(annotations, output_file):
    """
    Write annotations to a JSONL file with proper formatting (indentation).
    
    Args:
        annotations: List of annotation dictionaries
        output_file: Path to the output JSONL file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, annotation in enumerate(annotations):
            # Write each annotation as a formatted JSON object (with indentation)
            json_str = json.dumps(annotation, ensure_ascii=False, indent=2)
            f.write(json_str)
            
            # Add a comma and newline between entries (except for the last one)
            if i < len(annotations) - 1:
                f.write(',\n')
            else:
                f.write('\n')


def merge_annotations(input_dir, output_file):
    """
    Merge all annotation files (CSV and JSONL) from input directory into a single JSONL file.
    
    Args:
        input_dir: Directory containing annotation files
        output_file: Path to the output merged JSONL file
    """
    input_path = Path(input_dir)
    all_annotations = []
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist!")
        return
    
    # Process all CSV files
    csv_files = list(input_path.glob('*.csv'))
    for csv_file in csv_files:
        print(f"Reading CSV file: {csv_file.name}")
        annotations = read_csv_annotations(csv_file)
        all_annotations.extend(annotations)
        print(f"  Added {len(annotations)} annotations from {csv_file.name}")
    
    # Process all JSONL files
    jsonl_files = list(input_path.glob('*.jsonl'))
    for jsonl_file in jsonl_files:
        print(f"Reading JSONL file: {jsonl_file.name}")
        annotations = read_jsonl_annotations(jsonl_file)
        all_annotations.extend(annotations)
        print(f"  Added {len(annotations)} annotations from {jsonl_file.name}")
    
    # Write merged annotations to output file
    if all_annotations:
        write_formatted_jsonl(all_annotations, output_file)
        print(f"\nSuccessfully merged {len(all_annotations)} annotations into '{output_file}'")
    else:
        print("\nNo annotations found to merge!")


if __name__ == "__main__":
    INPUT_DIR = "documentation/03_sprint/annotation"
    OUTPUT_FILE = "documentation/03_sprint/annotation/annotations_best.jsonl"
    
    output_path = Path(OUTPUT_FILE).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Starting annotation merge process...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 60)
    
    merge_annotations(INPUT_DIR, OUTPUT_FILE)
    
    print("-" * 60)
    print("Process completed!")