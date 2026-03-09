from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="Linguistica Corpus Search API")

# Enable CORS since the frontend will be served from a different origin
# (e.g., opened locally via file:// or a simple HTTP server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class SearchResult(BaseModel):
    doc_id: str
    snippet: str
    is_annotated: bool


class SearchResponse(BaseModel):
    total_hits: int
    search_time_ms: int
    results: List[SearchResult]


# ... Add Whoosh imports ...
import os
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, BOOLEAN
from whoosh.qparser import QueryParser
from whoosh.query import Term
from whoosh.highlight import HtmlFormatter

# Mock database of corpus documents
MOCK_CORPUS = [
    {
        "doc_id": "doc_001",
        "text": "The quick brown fox jumps over the lazy dog.",
        "is_annotated": True,
    },
    {
        "doc_id": "doc_002",
        "text": "Machine learning and natural language processing are fascinating fields of study.",
        "is_annotated": False,
    },
    {
        "doc_id": "doc_003",
        "text": "A comprehensive study on the linguistic patterns and linguistics of early internet forums.",
        "is_annotated": True,
    },
    {
        "doc_id": "doc_004",
        "text": "Another document about foxes and dogs interacting.",
        "is_annotated": False,
    },
    {
        "doc_id": "doc_005",
        "text": "The linguistics and linguistic analysis of text corpora involves significant data processing.",
        "is_annotated": True,
    },
]


INDEX_DIR = "whoosh_index"


def init_index():
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)

    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        text=TEXT(stored=True),
        is_annotated=BOOLEAN(stored=True),
    )

    if not exists_in(INDEX_DIR):
        ix = create_in(INDEX_DIR, schema)
        writer = ix.writer()
        for doc in MOCK_CORPUS:
            writer.add_document(
                doc_id=doc["doc_id"], text=doc["text"], is_annotated=doc["is_annotated"]
            )
        writer.commit()
        return ix
    return open_dir(INDEX_DIR)


ix = init_index()


@app.get("/search", response_model=SearchResponse)
async def search_corpus(
    q: str = Query(..., description="The search query"),
    annotated_only: bool = Query(
        False, description="Filter to only return annotated documents"
    ),
):
    start_time = time.time()
    filtered_results = []

    with ix.searcher() as searcher:
        # Standard query parser against the 'text' field
        parser = QueryParser("text", ix.schema)
        try:
            query = parser.parse(q)
        except Exception:
            # Handle invalid query syntax gracefully
            return SearchResponse(total_hits=0, search_time_ms=0, results=[])

        # Optional filter for 'annotated_only'
        filter_q = None
        if annotated_only:
            filter_q = Term("is_annotated", True)

        # Search the index
        results = searcher.search(query, filter=filter_q, limit=50)

        # Configure Whoosh formatter to use the exact same span class as our CSS
        results.formatter = HtmlFormatter(
            tagname="span", classname="highlight", termclass="highlight"
        )

        for r in results:
            # Produce highlighted snippets
            snippet = r.highlights("text")
            if not snippet:
                # Fallback if text matched but wasn't highlighted properly
                snippet = r["text"][:150] + "..."

            filtered_results.append(
                SearchResult(
                    doc_id=r["doc_id"],
                    snippet=snippet,
                    is_annotated=r["is_annotated"],
                )
            )

    search_time_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        total_hits=len(filtered_results),
        search_time_ms=search_time_ms,
        results=filtered_results,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
