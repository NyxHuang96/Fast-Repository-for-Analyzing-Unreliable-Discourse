from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import os
import json
import glob

# --- Whoosh imports ---
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, BOOLEAN, KEYWORD
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import Term, And
from whoosh.highlight import HtmlFormatter
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter

# --- Jieba-based Chinese Analyzer (separate module for pickle compatibility) ---
from analyzers import ChineseAnalyzer


# =============================================================================
# App setup
# =============================================================================

app = FastAPI(title="Linguistica Corpus Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pydantic models
# =============================================================================


class SearchResult(BaseModel):
    doc_id: str
    label: str
    snippet: str
    snippet_zh: str
    is_annotated: bool


class SearchResponse(BaseModel):
    total_hits: int
    search_time_ms: int
    results: List[SearchResult]


class UploadResponse(BaseModel):
    message: str
    docs_added: int


class IndexStatsResponse(BaseModel):
    total_docs: int
    annotated_docs: int
    raw_docs: int
    labels: dict


# =============================================================================
# Index configuration
# =============================================================================

INDEX_DIR = "whoosh_index"
CORPUS_DIR = "corpus_data"  # 语料文件存放目录
# corpus_data/
#   raw/          ← 未标注语料 (is_annotated=False)
#   annotated/    ← 已标注语料 (is_annotated=True)

ix = None  # global index handle


def get_schema():
    return Schema(
        doc_id=ID(stored=True, unique=True),
        label=KEYWORD(stored=True, lowercase=True),
        text=TEXT(
            stored=True, analyzer=RegexTokenizer() | LowercaseFilter() | StopFilter()
        ),
        text_zh=TEXT(stored=True, analyzer=ChineseAnalyzer()),
        is_annotated=BOOLEAN(stored=True),
    )


def _load_json_file(filepath: str) -> list:
    """Load a .json (array) or .jsonl (one-object-per-line) file.
    Also handles pretty-printed JSON objects separated by commas (no outer []).
    """
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return docs

    # 1) Try parsing the entire file as JSON (arrays or single objects)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 2) Try wrapping with [ ] — handles files that are JSON array bodies
    #    without the outer brackets (e.g. {..},\n{..},\n{..})
    try:
        # Remove trailing commas and whitespace more aggressively
        trimmed = content.rstrip()
        while trimmed.endswith(","):
            trimmed = trimmed[:-1].rstrip()
        wrapped = "[" + trimmed + "]"
        parsed = json.loads(wrapped)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3) Iteratively extract JSON objects using raw_decode
    #    This handles any mix of whitespace, commas, and formatting
    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)
    try:
        while idx < length:
            # Skip whitespace and commas between objects
            while idx < length and content[idx] in " \t\n\r,":
                idx += 1
            if idx >= length:
                break
            obj, end_idx = decoder.raw_decode(content, idx)
            docs.append(obj)
            idx = end_idx
        if docs:
            return docs
    except json.JSONDecodeError:
        docs = []

    # 4) Last resort: treat as JSONL (one JSON object per line)
    for line in content.splitlines():
        line = line.strip().rstrip(",")
        if line:
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return docs


def _add_docs_to_writer(writer, docs: list, is_annotated: bool, start_id: int = 0):
    """Add a list of docs to the Whoosh index writer."""
    count = 0
    for i, doc in enumerate(docs):
        doc_id = doc.get("doc_id", f"doc_{start_id + i:06d}")
        label = doc.get("label", "Unknown")
        text = doc.get("text", "")
        text_zh = doc.get("text_zh", "")

        # Change update_document to add document for faster addition 
        writer.add_document(
            doc_id=doc_id,
            label=label,
            text=text,
            text_zh=text_zh,
            is_annotated=is_annotated,
        )
        count += 1
    return count


def init_index(force_rebuild: bool = False):
    """Initialize or rebuild the Whoosh index from corpus files."""
    global ix

    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)

    # Ensure corpus directories exist
    os.makedirs(os.path.join(CORPUS_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(CORPUS_DIR, "annotated"), exist_ok=True)

    schema = get_schema()

    if force_rebuild or not exists_in(INDEX_DIR):
        ix = create_in(INDEX_DIR, schema)
        writer = ix.writer()
        total = 0

        # Load raw (unannotated) corpus files
        raw_pattern = os.path.join(CORPUS_DIR, "raw", "*.json*")
        for fpath in sorted(glob.glob(raw_pattern)):
            docs = _load_json_file(fpath)
            n = _add_docs_to_writer(writer, docs, is_annotated=False, start_id=total)
            total += n
            print(f"  [raw] Loaded {n} docs from {os.path.basename(fpath)}")

        # Load annotated corpus files
        ann_pattern = os.path.join(CORPUS_DIR, "annotated", "*.json*")
        for fpath in sorted(glob.glob(ann_pattern)):
            docs = _load_json_file(fpath)
            n = _add_docs_to_writer(writer, docs, is_annotated=True, start_id=total)
            total += n
            print(f"  [annotated] Loaded {n} docs from {os.path.basename(fpath)}")

        writer.commit()
        print(f"Index built: {total} documents total.")
    else:
        ix = open_dir(INDEX_DIR)
        print(f"Existing index opened ({ix.doc_count()} documents).")


# Build index on startup
init_index()


# =============================================================================
# API endpoints
# =============================================================================


@app.get("/search", response_model=SearchResponse)
async def search_corpus(
    q: str = Query(..., description="The search query"),
    annotated_only: bool = Query(
        False, description="Filter to only annotated documents"
    ),
    label: Optional[str] = Query(
        None, description="Filter by label (Ham, Spam, Phish)"
    ),
):
    """
    Search the corpus. Automatically searches both English (text) and Chinese
    (text_zh) fields. Supports filtering by annotation status and label.
    """
    start_time = time.time()
    filtered_results = []

    with ix.searcher() as searcher:
        # Multi-field parser: searches both text and text_zh
        parser = MultifieldParser(["text", "text_zh"], ix.schema, group=OrGroup)
        try:
            query = parser.parse(q)
        except Exception:
            return SearchResponse(total_hits=0, search_time_ms=0, results=[])

        # Build combined query with filters using And()
        # NOTE: Whoosh BOOLEAN fields index as 't' / 'f' strings
        parts = [query]
        if annotated_only:
            parts.append(Term("is_annotated", "t"))
        if label:
            parts.append(Term("label", label.lower()))

        final_query = And(parts) if len(parts) > 1 else query

        results = searcher.search(final_query, limit=50)

        # HTML highlight formatter
        results.formatter = HtmlFormatter(
            tagname="span", classname="highlight", termclass="highlight"
        )

        for r in results:
            # Highlight English text
            snippet = r.highlights("text")
            if not snippet:
                raw_text = r.get("text", "")
                snippet = (raw_text[:150] + "...") if raw_text else ""

            # Highlight Chinese text
            snippet_zh = r.highlights("text_zh")
            if not snippet_zh:
                raw_zh = r.get("text_zh", "")
                snippet_zh = (raw_zh[:150] + "...") if raw_zh else ""

            filtered_results.append(
                SearchResult(
                    doc_id=r["doc_id"],
                    label=r.get("label", "Unknown"),
                    snippet=snippet,
                    snippet_zh=snippet_zh,
                    is_annotated=r["is_annotated"],
                )
            )

    search_time_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        total_hits=len(filtered_results),
        search_time_ms=search_time_ms,
        results=filtered_results,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_corpus(
    file: UploadFile = File(..., description="JSON or JSONL corpus file"),
    is_annotated: bool = Form(
        False, description="Whether these documents are annotated"
    ),
):
    """
    Upload a JSON/JSONL corpus file to dynamically add documents to the index.
    The file will also be saved to the appropriate corpus_data/ subdirectory.
    """
    if not file.filename.endswith((".json", ".jsonl")):
        raise HTTPException(
            status_code=400, detail="Only .json or .jsonl files are accepted."
        )

    content = await file.read()
    text = content.decode("utf-8").strip()

    # Parse documents
    try:
        if text.startswith("["):
            docs = json.loads(text)
        else:
            docs = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if not docs:
        raise HTTPException(status_code=400, detail="File contains no documents.")

    # Save file to corpus directory for persistence
    subdir = "annotated" if is_annotated else "raw"
    save_path = os.path.join(CORPUS_DIR, subdir, file.filename)
    with open(save_path, "wb") as f:
        f.write(content)

    # Add to index
    writer = ix.writer()
    current_count = ix.doc_count()
    n = _add_docs_to_writer(
        writer, docs, is_annotated=is_annotated, start_id=current_count
    )
    writer.commit()

    return UploadResponse(message=f"Successfully added {n} documents.", docs_added=n)


@app.post("/reindex")
async def reindex():
    """Force a full re-index from all files in corpus_data/."""
    init_index(force_rebuild=True)
    return {"message": f"Re-index complete. Total documents: {ix.doc_count()}"}


@app.get("/stats", response_model=IndexStatsResponse)
async def get_stats(
    corpus: str = Query(
        "all",
        description="Filter: all, english, chinese, english_annotated, chinese_annotated",
    ),
):
    """Return index statistics, optionally filtered by corpus type."""
    label_counts = {}
    annotated = 0
    raw = 0

    with ix.searcher() as searcher:
        for doc in searcher.all_stored_fields():
            # --- Determine language ---
            has_en = bool(doc.get("text", "").strip())
            has_zh = bool(doc.get("text_zh", "").strip())
            is_ann = doc.get("is_annotated", False)

            # Apply corpus filter
            if corpus == "english" and not has_en:
                continue
            elif corpus == "chinese" and not has_zh:
                continue
            elif corpus == "english_annotated" and not (has_en and is_ann):
                continue
            elif corpus == "chinese_annotated" and not (has_zh and is_ann):
                continue
            # "all" — no filter

            lbl = doc.get("label", "Unknown")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            if is_ann:
                annotated += 1
            else:
                raw += 1

    total = annotated + raw
    return IndexStatsResponse(
        total_docs=total,
        annotated_docs=annotated,
        raw_docs=raw,
        labels=label_counts,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
