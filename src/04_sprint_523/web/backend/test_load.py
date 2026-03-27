import json, glob, os
def _load_json_file(filepath: str) -> list:
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content: return docs
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list): return parsed
        elif isinstance(parsed, dict): return [parsed]
    except json.JSONDecodeError: pass

    try:
        trimmed = content.rstrip()
        while trimmed.endswith(","): trimmed = trimmed[:-1].rstrip()
        wrapped = "[" + trimmed + "]"
        parsed = json.loads(wrapped)
        if isinstance(parsed, list): return parsed
    except json.JSONDecodeError: pass

    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)
    try:
        while idx < length:
            while idx < length and content[idx] in " \t\n\r,": idx += 1
            if idx >= length: break
            obj, end_idx = decoder.raw_decode(content, idx)
            docs.append(obj)
            idx = end_idx
        if docs: return docs
    except json.JSONDecodeError as e:
        # print("raw_decode error:", e)
        docs = []

    for line in content.splitlines():
        line = line.strip().rstrip(",")
        if line:
            try: docs.append(json.loads(line))
            except json.JSONDecodeError: continue
    return docs

print("chinese:", len(_load_json_file("corpus_data/raw/chinese_sub_corpus_translated_api.json")))
print("english:", len(_load_json_file("corpus_data/raw/english_sub_corpus.json")))
print("annotated:", len(_load_json_file("corpus_data/annotated/annotations_best.jsonl")))
