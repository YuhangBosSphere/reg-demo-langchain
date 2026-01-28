import os
import re
import json
import ast
import operator as op
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

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# 如果你未来想中英混合更稳，改成：
# EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Agent settings
MAX_SUBQUESTIONS = 5
MIN_TOP1_SCORE = 0.15  # 证据太弱就拒答（你可根据实际调）
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

app = FastAPI(title="Agentic RAG (Layer-3) - Handwritten")

embedder = SentenceTransformer(EMBED_MODEL_NAME)

faiss_index: Optional[faiss.Index] = None
chunks_meta: List[Dict] = []  # parallel with vectors


# ----------------------------
# Utils: files & chunking
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


# ----------------------------
# Utils: embeddings & FAISS
# ----------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine-ish if embeddings normalized
    index.add(vectors)
    return index

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


# ----------------------------
# Schemas
# ----------------------------
class IngestResponse(BaseModel):
    docs_count: int
    chunks_count: int
    index_path: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K

class SearchHit(BaseModel):
    score: float
    source: str
    chunk_id: int
    chunk_in_doc: int
    text: str

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHit]

class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K

class AgentStep(BaseModel):
    subquestion: str
    top_hit_score: float
    used_tool: str
    notes: str

class AskResponse(BaseModel):
    question: str
    plan: List[str]
    steps: List[AgentStep]
    top_k: int
    answer: str
    sources: List[SearchHit]
    refused: bool
    refusal_reason: Optional[str] = None


# ----------------------------
# Tools (Tool calling)
# ----------------------------
def retrieval_tool(query: str, top_k: int) -> List[SearchHit]:
    """Tool: semantic retrieval."""
    ensure_loaded()
    if faiss_index is None or not chunks_meta:
        raise HTTPException(status_code=400, detail="Index not found. Run POST /ingest first.")

    q = query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty.")

    k = max(1, min(top_k, MAX_TOP_K))
    q_vec = embed_texts([q])  # (1, dim)
    scores, ids = faiss_index.search(q_vec, k)

    hits: List[SearchHit] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        m = chunks_meta[idx]
        hits.append(SearchHit(
            score=float(score),
            source=m.get("source", ""),
            chunk_id=int(idx),
            chunk_in_doc=int(m.get("chunk_in_doc", -1)),
            text=m.get("text", "")
        ))
    return hits

# Safe calculator for simple expressions: + - * / ( )
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def _eval_expr(node):
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("Unsupported expression")

def calculator_tool(expression: str) -> float:
    """Tool: safe arithmetic."""
    expr = expression.strip()
    if not expr:
        raise ValueError("Empty expression")
    tree = ast.parse(expr, mode="eval")
    return float(_eval_expr(tree.body))


# ----------------------------
# Agent logic: Plan -> Retrieve -> Decide -> Synthesize
# ----------------------------
def decompose_question(question: str) -> List[str]:
    """
    Very lightweight query decomposition (rule-based).
    Later you can swap this with an LLM planner or LangChain/LlamaIndex.
    """
    q = question.strip()
    if not q:
        return []

    # Split by common conjunctions / punctuation
    # Works for English and Chinese-ish: "and", "or", "?", "？", "；", "。" etc.
    parts = re.split(r"(?:\?|？|;|；|\.\s+|。|\n)+", q)
    parts = [p.strip() for p in parts if p.strip()]

    # If still one big sentence, try splitting by connectors
    if len(parts) == 1:
        parts = re.split(r"(?:\band\b|\bthen\b|\balso\b|\balong with\b|以及|并且|同时|还有|和|而且)", q)
        parts = [p.strip(" ,，。；;") for p in parts if p.strip(" ,，。；;")]

    # Heuristic: if question contains late + illness, split into 2 explicit subquestions
    low = q.lower()
    has_late = ("late" in low) or ("迟交" in q) or ("late submission" in low)
    has_ill = ("ill" in low) or ("medical" in low) or ("生病" in q) or ("emergency" in low)

    if has_late and has_ill:
        plan = [
            "What is the late submission penalty (timeline and percentage)?",
            "Are there exceptions or waivers for illness/emergency? What documentation is required?"
        ]
        return plan[:MAX_SUBQUESTIONS]

    # Default cap
    return parts[:MAX_SUBQUESTIONS] if parts else [q]


def decide_need_more_retrieval(hits: List[SearchHit], min_score: float) -> bool:
    """Stop rule: if top hit too weak, mark as missing evidence."""
    if not hits:
        return True
    return hits[0].score < min_score


def dedupe_hits(hits: List[SearchHit]) -> List[SearchHit]:
    """Deduplicate by chunk_id, keep highest score."""
    best: Dict[int, SearchHit] = {}
    for h in hits:
        if (h.chunk_id not in best) or (h.score > best[h.chunk_id].score):
            best[h.chunk_id] = h
    # sort by score descending
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def synthesize_answer(question: str, evidence: List[SearchHit]) -> Tuple[str, bool, Optional[str]]:
    """
    Non-LLM synthesis (deterministic).
    - Extracts key policy lines if possible.
    - If evidence is too weak, refuses.
    """
    if not evidence:
        return ("I don't know based on the provided context.", True, "No evidence retrieved.")

    # Refusal if even top evidence is weak
    if evidence[0].score < MIN_TOP1_SCORE:
        return ("I don't know based on the provided context.", True, f"Low confidence evidence (top score={evidence[0].score:.3f}).")

    # Simple extraction heuristics
    joined = "\n\n".join([h.text for h in evidence[:5]])
    low = joined.lower()

    penalty_info = []
    if "72" in low or "three" in question.lower() or "3" in question:
        # try to find lines mentioning 48-72 or 72 hours or 3 days
        for line in joined.splitlines():
            l = line.lower()
            if "48" in l and "72" in l:
                penalty_info.append(line.strip())
            if "more than 72" in l or "over 72" in l:
                penalty_info.append(line.strip())
            if "72 hours" in l:
                penalty_info.append(line.strip())
            if "30%" in l and ("72" in l or "48" in l):
                penalty_info.append(line.strip())

    illness_info = []
    for line in joined.splitlines():
        l = line.lower()
        if "illness" in l or "medical" in l or "emergency" in l or "doctor" in l or "documentation" in l:
            illness_info.append(line.strip())

    # Build a concise answer with citations hint
    answer_lines = []
    answer_lines.append("Based on the course policy documents:")

    if penalty_info:
        answer_lines.append("- Late penalty relevant to 'three days / 48–72 hours':")
        for x in penalty_info[:3]:
            answer_lines.append(f"  • {x}")
    else:
        answer_lines.append("- Late penalty: see cited policy chunks in sources (no exact 'three-day' line extracted).")

    if illness_info:
        answer_lines.append("- Illness/emergency exceptions:")
        for x in illness_info[:4]:
            answer_lines.append(f"  • {x}")
    else:
        answer_lines.append("- Illness/emergency exceptions: not found explicitly in retrieved context.")

    answer_lines.append("")
    answer_lines.append("Sources are provided with chunk_id and file name for verification.")

    return ("\n".join(answer_lines), False, None)


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": EMBED_MODEL_NAME,
        "cwd": os.getcwd(),
        "docs_abs": os.path.abspath(DOCS_DIR),
        "index_abs": os.path.abspath(INDEX_DIR),
    }


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
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty.")
    top_k = max(1, min(req.top_k, MAX_TOP_K))

    hits = retrieval_tool(q, top_k)

    return SearchResponse(query=q, top_k=top_k, hits=hits)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Agentic RAG:
      1) Plan (decompose question)
      2) For each subquestion: retrieval_tool
      3) Stop rule: if evidence weak, mark missing
      4) Merge evidence and synthesize answer
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    top_k = max(1, min(req.top_k, MAX_TOP_K))

    plan = decompose_question(question)
    if not plan:
        raise HTTPException(status_code=400, detail="Failed to generate a plan.")

    all_hits: List[SearchHit] = []
    steps: List[AgentStep] = []

    for subq in plan:
        hits = retrieval_tool(subq, top_k=top_k)
        need_more = decide_need_more_retrieval(hits, MIN_TOP1_SCORE)

        top_score = hits[0].score if hits else -1.0
        notes = "ok"
        if need_more:
            notes = "low_evidence_or_empty"

        steps.append(AgentStep(
            subquestion=subq,
            top_hit_score=float(top_score),
            used_tool="retrieval_tool",
            notes=notes
        ))

        # Always keep some evidence; the synthesis stage will decide refusal if weak
        all_hits.extend(hits)

    merged = dedupe_hits(all_hits)
    # keep top evidence only
    merged = merged[:max(top_k, 8)]

    answer, refused, reason = synthesize_answer(question, merged)

    return AskResponse(
        question=question,
        plan=plan,
        steps=steps,
        top_k=top_k,
        answer=answer,
        sources=merged,
        refused=refused,
        refusal_reason=reason
    )
