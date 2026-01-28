import os
import re
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
DOCS_DIR = "./docs"
INDEX_DIR = "./index_store"
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "docs.index")
METADATA_PATH = os.path.join(INDEX_DIR, "docs_meta.json")

# A solid default embedding model (small + good)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking config (character-based for simplicity)
CHUNK_SIZE = 900       # chars
CHUNK_OVERLAP = 150    # chars

app = FastAPI(title="RAG Layer-1 Demo (FAISS + Embeddings)")

embedder = SentenceTransformer(EMBED_MODEL_NAME)

faiss_index: Optional[faiss.Index] = None
chunks_meta: List[Dict] = []  # parallel array with faiss vectors


# ----------------------------
# Utils
# ----------------------------
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def list_docs(docs_dir: str) -> List[str]:
    exts = {".txt", ".md"}
    paths = []
    for root, _, files in os.walk(docs_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                paths.append(os.path.join(root, name))
    return sorted(paths)

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    # Returns float32 vectors
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def save_store(index: faiss.Index, meta: List[Dict]) -> None:
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_store() -> Tuple[Optional[faiss.Index], List[Dict]]:
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        return None, []
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def ensure_loaded():
    global faiss_index, chunks_meta
    if faiss_index is None:
        faiss_index, chunks_meta = load_store()

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    # Using cosine similarity via normalized vectors + inner product
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# ----------------------------
# Schemas
# ----------------------------
class IngestResponse(BaseModel):
    docs_count: int
    chunks_count: int
    index_path: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchHit(BaseModel):
    score: float
    source: str
    chunk_id: int
    text: str

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHit]


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def health():
    return {"status": "ok", "model": EMBED_MODEL_NAME}

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    global faiss_index, chunks_meta

    doc_paths = list_docs(DOCS_DIR)
    if not doc_paths:
        raise HTTPException(status_code=400, detail=f"No .txt/.md files found in {DOCS_DIR}")

    all_chunks: List[str] = []
    meta: List[Dict] = []

    for p in doc_paths:
        content = read_text_file(p)
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            meta.append({
                "source": os.path.relpath(p, DOCS_DIR),
                "chunk_in_doc": i,
                "text": c
            })

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks produced. Check your docs.")

    vectors = embed_texts(all_chunks)
    index = build_faiss_index(vectors)

    save_store(index, meta)

    faiss_index = index
    chunks_meta = meta

    return IngestResponse(
        docs_count=len(doc_paths),
        chunks_count=len(all_chunks),
        index_path=os.path.abspath(FAISS_INDEX_PATH)
    )

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    ensure_loaded()
    if faiss_index is None or not chunks_meta:
        raise HTTPException(status_code=400, detail="Index not found. Run POST /ingest first.")

    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty.")
    top_k = max(1, min(req.top_k, 20))

    q_vec = embed_texts([q])  # shape (1, dim)
    scores, ids = faiss_index.search(q_vec, top_k)

    hits: List[SearchHit] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        m = chunks_meta[idx]
        hits.append(SearchHit(
            score=float(score),
            source=m["source"],
            chunk_id=int(idx),
            text=m["text"]
        ))

    return SearchResponse(query=q, top_k=top_k, hits=hits)
