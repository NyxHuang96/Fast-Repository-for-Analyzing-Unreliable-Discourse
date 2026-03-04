import json
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
script_dir      = Path(__file__).resolve().parent
processed_dir   = script_dir.parent.parent / "data" / "processed"
file_path       = processed_dir / "chinese_sub_corpus.json"
output_path     = processed_dir / "chinese_sub_corpus_translated.json"
checkpoint_path = processed_dir / "chinese_sub_corpus_translated_checkpoint.json"

# ── Device selection: CUDA > Apple MPS > CPU ──────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")     # Apple Silicon GPU
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Helsinki-NLP/opus-mt-en-zh"
BATCH_SIZE       = 64    # raise to 128 if memory allows; lower to 32 if OOM
NUM_BEAMS        = 4     # set to 1 for greedy decoding (fastest, slight quality drop)
CHECKPOINT_EVERY = 10    # save progress every N batches (crash recovery)

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)

# fp16 halves memory and roughly doubles throughput on GPU
if DEVICE.type in ("cuda", "mps"):
    model = model.half()

model = model.to(DEVICE)
model.eval()
print(f"Model ready on {DEVICE}.\n")


# ── Translation ────────────────────────────────────────────────────────────────
def translate_all(texts: list[str]) -> list[str]:
    """
    Translate all texts with checkpoint recovery support.
    If a checkpoint file exists from a previous interrupted run,
    translation resumes from where it left off.
    """
    # Resume from checkpoint if available
    results     = []
    start_index = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_index = len(results)
        print(f"Resuming from checkpoint: {start_index}/{len(texts)} texts already done.\n")

    remaining = texts[start_index:]
    batches   = list(range(0, len(remaining), BATCH_SIZE))

    with torch.inference_mode():
        for batch_num, i in enumerate(tqdm(batches, desc="Translating")):
            batch = remaining[i : i + BATCH_SIZE]

            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            translated_tokens = model.generate(
                **tokens,
                num_beams=NUM_BEAMS,
            )
            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            results.extend(decoded)

            # Periodic checkpoint save
            if (batch_num + 1) % CHECKPOINT_EVERY == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)

    # Clean up checkpoint on success
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load data — handles both JSON array and JSONL (one object per line)
    print(f"Reading: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    # Collect source texts, preserving original record order
    indices = [i for i, item in enumerate(data) if item.get("text")]
    texts   = [data[i]["text"] for i in indices]
    print(f"Found {len(texts)} records with a 'text' field.\n")

    # Translate
    translations = translate_all(texts)

    # Write translations back as a new field (original "text" is preserved)
    for idx, translation in zip(indices, translations):
        data[idx]["text_zh"] = translation

    # Save final output
    print(f"\nSaving to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Done ✓")


if __name__ == "__main__":
    main()