import os
import json
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import traceback

# LangChain v1 / LangGraph
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate


from langgraph.checkpoint.memory import InMemorySaver

# Vectorstore + embeddings + splitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#local llm
from langchain_ollama import ChatOllama

#section chunk 
import hashlib
from collections import Counter

#update_2 
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from fastapi.responses import HTMLResponse

#graphRAG
import networkx as nx
import math

# ----------------------------
# Config
# ----------------------------
DOCS_DIR = "./docs"
INDEX_DIR = "./index_store_lc" 
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_DIR = os.path.join(INDEX_DIR, "faiss_store")  # folder for FAISS.save_local()
META_PATH = os.path.join(INDEX_DIR, "meta.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

LOGS_DIR = os.path.join(INDEX_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
DAY1_LOG_PATH = os.path.join(LOGS_DIR, "day1_run.jsonl")

#GraphRAG 
GRAPH_PATH = os.path.join(INDEX_DIR, "graphrag_graph.json")
E2C_PATH = os.path.join(INDEX_DIR, "graphrag_entity2chunks.json")
C2E_PATH = os.path.join(INDEX_DIR, "graphrag_chunk2entities.json")


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="LangChain v1 Agent + RAG (FAISS)")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},  # cosine via normalized vectors
)

# global store (lazy load)
vectorstore: Optional[FAISS] = None

# LangGraph checkpointer = short-term memory
checkpointer = InMemorySaver()

SYSTEM_PROMPT = """You are a clinic operations assistant.
Answer ONLY using the retrieved policy excerpts returned by the tool.
If the tool returns no relevant excerpts, say you don't know and ask the user to check the clinic policy document.
Do not provide medical advice. This is operational/policy Q&A only.
Always be concise and cite excerpts by quoting short snippets (not more than 1-2 sentences each)."""


# ----------------------------
# Utils
# ----------------------------
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def list_docs(docs_dir: str) -> List[str]:
    exts = {".txt", ".md"}
    paths: List[str] = []
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

def ensure_vectorstore_loaded() -> FAISS:
    global vectorstore
    if vectorstore is not None:
        return vectorstore
    if not os.path.exists(FAISS_DIR):
        raise HTTPException(status_code=400, detail="Vectorstore not found. Run POST /ingest first.")
    # FAISS.load_local uses pickle for docstore metadata → set allow_dangerous_deserialization=True
    # Only do this if you trust the local files (your own machine / your own generated store).
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore

# ----------------------------
# logger (jsonl)
# ----------------------------
def log_event(event: str, payload: Dict[str, Any]) -> None:
    """
    Append one JSON line to logs/day1_run.jsonl for debugging + eval later.
    """
    try:
        record = {
            "event": event,
            "payload": payload,
        }
        with open(DAY1_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # logging should never crash the app
        pass


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]

def split_lines(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    return [ln.strip() for ln in text.split("\n")]

def denoise_boilerplate(text: str, freq_ratio: float = 0.15, min_len: int = 6) -> str:
    """
    Remove repeated boilerplate lines (headers/footers/disclaimers).
    freq_ratio: if a line appears in >= freq_ratio * total_lines, treat it as boilerplate.
    """
    lines = split_lines(text)
    # Keep non-empty lines for frequency statistics
    nonempty = [ln for ln in lines if ln]
    if len(nonempty) < 50:
        return text  # too short; skip aggressive denoise

    counts = Counter(nonempty)
    threshold = max(2, int(len(nonempty) * freq_ratio))

    cleaned = []
    for ln in lines:
        if not ln:
            cleaned.append("")  # keep blank to preserve paragraph breaks
            continue
        if len(ln) >= min_len and counts.get(ln, 0) >= threshold:
            continue  # drop boilerplate
        # drop obvious page number lines
        if re.fullmatch(r"Page\s*\d+(\s*of\s*\d+)?", ln, flags=re.I):
            continue
        cleaned.append(ln)

    # Rebuild text with paragraph breaks
    rebuilt = "\n".join(cleaned)
    return normalize_whitespace(rebuilt)

_heading_md = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_heading_num = re.compile(r"^(\d+(\.\d+)*)\s+(.+?)\s*$")
_heading_colon = re.compile(r"^(.+?):\s*$")

def sectionize(text: str) -> List[Dict]:
    """
    Split text into sections based on headings.
    Returns list of {path: [..], title: str, body: str}
    """
    lines = split_lines(text)
    sections = []
    path: List[str] = []
    cur_title = "ROOT"
    cur_body: List[str] = []

    def push_section(title: str, body_lines: List[str], path_snapshot: List[str]):
        body = normalize_whitespace("\n".join(body_lines))
        if body:
            sections.append({"path": path_snapshot[:], "title": title, "body": body})

    for ln in lines:
        m1 = _heading_md.match(ln)
        m2 = _heading_num.match(ln)
        m3 = _heading_colon.match(ln)

        if m1:
            # markdown heading level
            level = len(m1.group(1))
            title = m1.group(2).strip()
            push_section(cur_title, cur_body, path)
            cur_body = []
            # adjust path depth
            path = path[: level - 1]
            path.append(title)
            cur_title = title
            continue

        if m2:
            # numeric heading: treat depth by dots count (1.2.3 -> depth 3)
            num = m2.group(1)
            title = m2.group(3).strip()
            depth = num.count(".") + 1
            push_section(cur_title, cur_body, path)
            cur_body = []
            path = path[: depth - 1]
            path.append(title)
            cur_title = title
            continue

        if m3 and len(ln) <= 80:
            # "Late Arrival Policy:" style heading
            title = m3.group(1).strip()
            push_section(cur_title, cur_body, path)
            cur_body = []
            # treat as same depth (append)
            if path and path[-1] == cur_title:
                path[-1] = title
            else:
                path.append(title)
            cur_title = title
            continue

        cur_body.append(ln)

    push_section(cur_title, cur_body, path)
    return sections

_bullet = re.compile(r"^\s*([-*•]|\d+\.)\s+")

def split_by_bullets(section_text: str) -> List[str]:
    """
    Split a section body into bullet items if it seems list-like.
    """
    lines = section_text.split("\n")
    # detect bullet density
    bullet_lines = sum(1 for ln in lines if _bullet.match(ln))
    if bullet_lines < 3:
        return [section_text]  # fallack: not list-like enough 

    items = []
    cur = []
    for ln in lines:
        if _bullet.match(ln) and cur:
            items.append(normalize_whitespace("\n".join(cur))) #push cur
            cur = [ln] # bullet line
        else:
            cur.append(ln)  
    if cur:
        items.append(normalize_whitespace("\n".join(cur))) #push cur last time

    # filter tiny items
    items = [it for it in items if len(it) >= 80]
    return items if items else [section_text]

def chunk_section_text(text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """
    List-first chunking: split by bullets, then apply splitter to each item if needed.
    """
    parts = split_by_bullets(text)
    chunks: List[str] = []
    for part in parts:
        if len(part) <= CHUNK_SIZE * 1.2:
            chunks.append(part)
        else:
            chunks.extend(splitter.split_text(part)) #push element 
    return [c.strip() for c in chunks if c.strip()]

#update_2 
def get_all_docs_from_vs(vs: FAISS) -> List[Document]:
    # langchain FAISS 的 docstore 通常是 InMemoryDocstore，内部 dict 存 Document
    try:
        store = getattr(vs, "docstore", None)
        d = getattr(store, "_dict", None)
        if isinstance(d, dict):
            return list(d.values())
    except Exception:
        pass
    return []

def hybrid_retrieve(
    vs: FAISS,
    query: str,
    k_final: int = 5,
    k_vec: int = 8,
    k_bm25: int = 8,
) -> List[tuple[Document, float]]:
    # 1) vector candidates
    vec_results = vs.similarity_search_with_score(query, k=max(1, k_vec))  # [(doc, score)]
    
    # 2) bm25 candidates (built on the fly)
    all_docs = get_all_docs_from_vs(vs)
    bm25_results: List[Document] = []
    if all_docs:
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = max(1, k_bm25)
        bm25_results = bm25.invoke(query)

    # 3) merge by chunk_id (or fallback to hash of text)
    merged: Dict[str, Dict[str, Any]] = {}

    # add vector results
    for doc, score in vec_results:
        cid = str(doc.metadata.get("chunk_id", "")) or sha1_short(doc.page_content)
        merged.setdefault(cid, {"doc": doc, "vec_score": float(score), "bm25_hit": 0.0})
        merged[cid]["vec_score"] = float(score)

    # add bm25 results (no score, just mark hit / count)
    for doc in bm25_results:
        cid = str(doc.metadata.get("chunk_id", "")) or sha1_short(doc.page_content)
        merged.setdefault(cid, {"doc": doc, "vec_score": None, "bm25_hit": 0.0})
        merged[cid]["bm25_hit"] += 1.0

    # 4) compute a simple combined score for pre-ranking
    # 注意：FAISS 的 score 可能是 distance（越小越好）也可能是 similarity（越大越好）
    # (element,float) and [str,{element,float,float}]
    items: List[tuple[Document, float]] = []
    for cid, item in merged.items():
        base = 0.0
        if item["bm25_hit"] > 0:
            base += 1.0
        if item["vec_score"] is not None:
            # 给一点权重让 vector 结果也上来
            base += 0.5
        items.append((item["doc"], float(base)))

    # 5) take more candidates than final, let reranker refine
    items.sort(key=lambda x: x[1], reverse=True)
    return items[: max(k_final, 10)]

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker: Optional[CrossEncoder] = None

def ensure_reranker_loaded() -> CrossEncoder:
    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANK_MODEL_NAME)
    return reranker

def sigmoid(x: float) -> float:
    # 防溢出保护
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def rerank_docs(query: str, docs: List[Document], top_n: int = 5) -> List[tuple[Document, float]]:
    if not docs:
        return []
    ce = ensure_reranker_loaded()
    pairs = [(query, d.page_content) for d in docs]
    raw_scores = ce.predict(pairs)  # can be negative logits
    scores = [sigmoid(float(s)) for s in raw_scores]  # now 0..1
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def dynamic_top_k_by_dropoff(
    reranked: List[tuple[Document, float]],
    k_min: int = 2,
    k_max: int = 8,
    drop_ratio: float = 0.72,
) -> int:
    """abs_min: float = 0.18,
    NOTE: abs_min removed because cross-encoder returns logits (can be negative)."""

    if not reranked:
        return 0
    scores = [float(s) for _, s in reranked]
    s0 = scores[0]

    k = 0
    for s in scores[:k_max]:
        if k >= k_min and s < s0 * drop_ratio:
            break
        k += 1

    return max(k_min, k)


_re_en_phrase = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_re_acronym = re.compile(r"\b([A-Z]{2,6}\d{0,3})\b")
_re_cn = re.compile(r"[\u4e00-\u9fff]{2,}")

def extract_entities_heuristic(text: str, max_entities: int = 30) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    ents = set()

    for m in _re_en_phrase.findall(t):
        s = m.strip()
        if 2 <= len(s) <= 40:
            ents.add(s)

    for m in _re_acronym.findall(t):
        s = m.strip()
        if 2 <= len(s) <= 12:
            ents.add(s)

    for m in _re_cn.findall(t):
        s = m.strip()
        if 2 <= len(s) <= 12:
            ents.add(s)

    # 去掉太泛
    stop = {"Policy", "Clinic", "Patient", "Procedure", "Document", "规定", "政策", "流程"}
    ents = [e for e in ents if e not in stop]

    # 稍微按长度/形态排序，让更“像实体”的优先
    ents.sort(key=lambda x: (-min(len(x), 20), x))
    return ents[:max_entities]

def build_graphrag_indexes(docs: List[Document]) -> Dict[str, Any]:
    """
    Return:
      - chunk2entities: {chunk_id: [ent,...]}
      - entity2chunks: {ent: [chunk_id,...]}
      - graph: adjacency list {chunk_id: {neighbor_chunk_id: weight}}
    """
    chunk2entities: Dict[str, List[str]] = {}
    entity2chunks: Dict[str, List[str]] = {}

    # 1) chunk -> entities
    for d in docs:
        cid = str(d.metadata.get("chunk_id", "")).strip()
        if not cid:
            continue
        ents = extract_entities_heuristic(d.page_content)
        if not ents:
            continue
        chunk2entities[cid] = ents
        for e in ents:
            entity2chunks.setdefault(e, []).append(cid)

    # 2) chunk graph
    G = nx.Graph()
    for cid in chunk2entities.keys():
        G.add_node(cid)

    # 对每个实体，把它出现过的 chunks 两两连边
    for e, cids in entity2chunks.items():
        # 去重
        uniq = list(dict.fromkeys(cids))
        if len(uniq) < 2:
            continue
        # 两两连边（若你担心 O(n^2)，可设置上限）
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                if a == b:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    # 3) adjacency list (JSON 友好)
    adj: Dict[str, Dict[str, int]] = {}
    for a, b, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w

    return {"chunk2entities": chunk2entities, "entity2chunks": entity2chunks, "adj": adj}

# ----------------------------
# GraphRAG globals (lazy load)
# ----------------------------
chunk2entities: Optional[Dict[str, List[str]]] = None
entity2chunks: Optional[Dict[str, List[str]]] = None
chunk_adj: Optional[Dict[str, Dict[str, int]]] = None


def ensure_graphrag_loaded() -> None:
    global chunk2entities, entity2chunks, chunk_adj
    if chunk2entities is not None and entity2chunks is not None and chunk_adj is not None:
        return
    if not (os.path.exists(C2E_PATH) and os.path.exists(E2C_PATH) and os.path.exists(GRAPH_PATH)):
        # 没构建过就当不存在（Ask 时会 fallback）
        chunk2entities, entity2chunks, chunk_adj = {}, {}, {}
        return
    with open(C2E_PATH, "r", encoding="utf-8") as f:
        chunk2entities = json.load(f)
    with open(E2C_PATH, "r", encoding="utf-8") as f:
        entity2chunks = json.load(f)
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        chunk_adj = json.load(f)

def graphrag_candidate_chunk_ids(query: str, hops: int = 2, max_chunks: int = 60) -> List[str]:
    ensure_graphrag_loaded()
    if not entity2chunks or not chunk_adj:
        return []

    q_ents = extract_entities_heuristic(query, max_entities=20)
    if not q_ents:
        return []

    # 1) seed chunks：包含 query 实体的 chunks
    seeds: List[str] = []
    for e in q_ents:
        for cid in entity2chunks.get(e, [])[:50]:
            seeds.append(cid)
    # 去重
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        return []

    # 2) BFS 扩展 chunk 图
    expanded = set(seeds)
    frontier = list(seeds)

    for _ in range(max(0, int(hops))):
        nxt = []
        for cid in frontier:
            for nb, w in (chunk_adj.get(cid, {}) or {}).items():
                if nb not in expanded:
                    expanded.add(nb)
                    nxt.append(nb)
                if len(expanded) >= max_chunks:
                    break
            if len(expanded) >= max_chunks:
                break
        frontier = nxt
        if not frontier:
            break

    return list(expanded)

def filter_docs_by_chunk_ids(all_docs: List[Document], chunk_ids: List[str], limit: int = 200) -> List[Document]:
    wanted = set(chunk_ids)
    out = []
    for d in all_docs:
        cid = str(d.metadata.get("chunk_id", ""))
        if cid in wanted:
            out.append(d)
            if len(out) >= limit:
                break
    return out

def graphrag_retrieve_and_rerank(
    vs: FAISS,
    query: str,
    hops: int = 2,
    bm25_k: int = 30,
    rerank_top_n: int = 8,
) -> List[tuple[Document, float]]:
    """
    1) GraphRAG: candidate chunk ids
    2) BM25 on candidate docs
    3) Cross-encoder rerank
    """
    all_docs = get_all_docs_from_vs(vs)
    if not all_docs:
        return []

    cand_ids = graphrag_candidate_chunk_ids(query, hops=hops, max_chunks=80)
    if not cand_ids:
        return []

    cand_docs = filter_docs_by_chunk_ids(all_docs, cand_ids, limit=300)
    if not cand_docs:
        return []

    bm25 = BM25Retriever.from_documents(cand_docs)
    bm25.k = max(5, int(bm25_k))
    bm25_hits = bm25.invoke(query)  # List[Document]

    # bm25 可能返回重复 doc（看实现），去重
    uniq = []
    seen = set()
    for d in bm25_hits:
        cid = str(d.metadata.get("chunk_id", "")) or sha1_short(d.page_content)
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(d)

    # rerank
    reranked = rerank_docs(query, uniq, top_n=max(5, int(rerank_top_n)))
    return reranked

# ----------------------------
# Schemas
# ----------------------------
class IngestResponse(BaseModel):
    docs_count: int
    chunks_count: int
    store_dir: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchHit(BaseModel):
    score: float
    source: str
    chunk_id: str
    text: str

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHit]

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"  # thread_id for short-term memory
    top_k: int = 5
    
# ----------------------------
# unified schemas (Agent-ready)
# ----------------------------
class Chunk(BaseModel):
    text: str
    source: str
    chunk_id: str
    score: float
    metadata: Dict[str, Any] = {}

class SearchResult(BaseModel):
    query: str
    top_k: int
    chunks: List[Chunk]

class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: List[str]          # e.g. ["policy.md#file.txt::abcd::003"]
    used_chunks: List[Chunk]


# ----------------------------
# internal search wrapper (structured)
# ----------------------------
def search_policy_chunks(query: str, top_k: int = 5) -> SearchResult:
    """
    Use your existing retrieval logic to return structured chunks.
    Day 1 MVP: use the SAME vs.similarity_search_with_score you already have in tool/search route.
    (Later Day 2 you can swap to hybrid_retrieve + rerank here.)
    """
    vs = ensure_vectorstore_loaded()
    q = query.strip()
    k = max(1, min(int(top_k), 10))

    results = vs.similarity_search_with_score(q, k=k)
    chunks: List[Chunk] = []

    for doc, score in results:
        chunks.append(
            Chunk(
                text=str(doc.page_content).strip(),
                source=str(doc.metadata.get("source", "unknown")),
                chunk_id=str(doc.metadata.get("chunk_id", "unknown")),
                score=float(score),
                metadata=dict(doc.metadata or {}),
            )
        )

    sr = SearchResult(query=q, top_k=k, chunks=chunks)
    log_event("day1.search_policy_chunks", sr.model_dump())
    return sr
# ----------------------------
# New Tool (structured JSON) for Agent
# ----------------------------
@tool
def retrieve_policy_chunks_json(query: str, top_k: int = 5) -> str:
    """
    Search local clinic policy documents and return JSON {query, top_k, chunks:[...]}.
    This is Agent-friendly because the output is machine-readable.
    """
    sr = search_policy_chunks(query=query, top_k=top_k)
    if not sr.chunks:
        return json.dumps({"query": query, "top_k": top_k, "chunks": []}, ensure_ascii=False)
    return sr.model_dump_json()


# ----------------------------
# Tool: retrieve chunks
# ----------------------------
@tool
def retrieve_policy_excerpts(query: str, top_k: int = 5) -> str:
    """Search the local clinic policy documents and return the most relevant excerpts with sources."""
    vs = ensure_vectorstore_loaded()
    k = max(1, min(int(top_k), 10))

    # similarity_search_with_score returns List[(Document, score)]
    results = vs.similarity_search_with_score(query, k=k)

    if not results:
        return "NO_HITS"

    # Build a compact, model-readable “evidence pack”
    lines: List[str] = []
    for i, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "unknown")
        cid = doc.metadata.get("chunk_id", "unknown")
        text = doc.page_content.strip()
        if len(text) > 700:
            text = text[:700].rstrip() + "..."
        lines.append(f"[{i}] score={score:.4f} source={src} chunk_id={cid}\n{text}")
    return "\n\n".join(lines)


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url="http://127.0.0.1:11434",
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a clinic operations assistant. "
     "Answer ONLY using the provided policy excerpts. "
     "If the excerpts do not contain the answer, say you don't know."),
    ("human",
     "Question:\n{question}\n\n"
     "Policy excerpts:\n{context}")
])

# ----------------------------
#  answer with context (structured)
# ----------------------------
_cite_pat = re.compile(r"\[(\d+)\]")

def answer_with_context(question: str, chunks: List[Chunk]) -> AnswerResult:
    """
    Build context from structured chunks and call your existing PROMPT | llm chain.
    Extract citations by detecting [1], [2]... in the answer.
    """
    # Keep context compact to avoid token blow-ups
    evidence = []
    for i, c in enumerate(chunks[:10], start=1):
        snippet = c.text
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."
        evidence.append(f"[{i}] ({c.source} | {c.chunk_id}) {snippet}")

    context = "\n\n".join(evidence)

    chain = PROMPT | llm
    msg = chain.invoke({"question": question, "context": context})
    answer_text = getattr(msg, "content", str(msg))

    # citations: map [i] -> source#chunk_id
    citations: List[str] = []
    seen = set()
    for m in _cite_pat.findall(answer_text or ""):
        try:
            idx = int(m)
        except Exception:
            continue
        if 1 <= idx <= len(chunks):
            key = f"{chunks[idx-1].source}#{chunks[idx-1].chunk_id}"
            if key not in seen:
                seen.add(key)
                citations.append(key)

    ar = AnswerResult(
        question=question,
        answer=answer_text,
        citations=citations,
        used_chunks=chunks[:10],
    )
    log_event("day1.answer_with_context", ar.model_dump())
    return ar


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def health():
    return {"status": "ok", "embeddings": EMBED_MODEL_NAME, "store_dir": os.path.abspath(FAISS_DIR)}

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    global vectorstore

    doc_paths = list_docs(DOCS_DIR)
    if not doc_paths:
        raise HTTPException(status_code=400, detail=f"No .txt/.md files found in {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents: List[Document] = []
    chunks_count = 0

    for p in doc_paths:
        raw = read_text_file(p)
        raw = normalize_whitespace(raw)
        raw = denoise_boilerplate(raw)

        if not raw:
            continue

        # 1) sectionize
        secs = sectionize(raw)
        if not secs:
            secs = [{"path": ["ROOT"], "title": "ROOT", "body": raw}]

        # 2) chunk within sections (list-first)
        rel_source = os.path.relpath(p, DOCS_DIR)

        for s_idx, sec in enumerate(secs):
            section_path = sec["path"] if sec["path"] else [sec["title"]]
            section_title = sec["title"]
            body = sec["body"]

            sec_chunks = chunk_section_text(body, splitter)

            for c_idx, c in enumerate(sec_chunks):
                chunks_count += 1
                # stable, readable chunk id
                sec_key = " > ".join(section_path) if section_path else section_title
                cid = f"{os.path.basename(p)}::{sha1_short(sec_key)}::{c_idx:03d}"

                documents.append(
                    Document(
                        page_content=c,
                        metadata={
                            "source": rel_source,
                            "section_title": section_title,
                            "section_path": section_path,
                            "section_key": sec_key,
                            "chunk_in_section": c_idx,
                            "chunk_id": cid,
                        },
                    )
                )

    if not documents:
        raise HTTPException(status_code=400, detail="No chunks produced. Check your docs content.")

    vs = FAISS.from_documents(documents, embeddings)
    vs.save_local(FAISS_DIR)

    # ---- GraphRAG offline build ----
    graphrag = build_graphrag_indexes(documents)

    with open(C2E_PATH, "w", encoding="utf-8") as f:
        json.dump(graphrag["chunk2entities"], f, ensure_ascii=False)

    with open(E2C_PATH, "w", encoding="utf-8") as f:
        json.dump(graphrag["entity2chunks"], f, ensure_ascii=False)

    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graphrag["adj"], f, ensure_ascii=False)


    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"docs": [os.path.relpath(p, DOCS_DIR) for p in doc_paths], "chunks": chunks_count},
            f,
            ensure_ascii=False,
            indent=2,
        )

    vectorstore = vs

    return IngestResponse(
        docs_count=len(doc_paths),
        chunks_count=chunks_count,
        store_dir=os.path.abspath(FAISS_DIR),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    vs = ensure_vectorstore_loaded() #FAISS 
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty.")
    k = max(1, min(int(req.top_k), 20))

    results = vs.similarity_search_with_score(q, k=k)

    hits: List[SearchHit] = []
    for doc, score in results:
        hits.append(
            SearchHit(
                score=float(score),
                source=str(doc.metadata.get("source", "unknown")),
                chunk_id=str(doc.metadata.get("chunk_id", "unknown")),
                text=doc.page_content,
            )
        )
    return SearchResponse(query=q, top_k=k, hits=hits)

ASK_TOPK_MAX = 10
SCORE_REJECT_THRESHOLD = 0.75  # 你可以调：越小越“更愿意回答”（不同 embedding/FAISS分数体系会不同）

@app.post("/ask")
def ask(req: AskRequest):
    vs = ensure_vectorstore_loaded()
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    # ---- 1) GraphRAG first ----
    graphrag_reranked = graphrag_retrieve_and_rerank(
        vs, question, hops=2, bm25_k=30, rerank_top_n=20
    )

    if graphrag_reranked:
        k_ctx = dynamic_top_k_by_dropoff(graphrag_reranked, k_min=2, k_max=8)
        selected = graphrag_reranked[:k_ctx] if k_ctx > 0 else graphrag_reranked[:2]
    else:
        selected = []

    # ---- 2) fallback to your original hybrid pipeline ----
    if not selected:
        k = max(1, min(int(req.top_k), ASK_TOPK_MAX))
        candidates = hybrid_retrieve(vs, question, k_final=k, k_vec=10, k_bm25=10)
        if not candidates:
            return {"answer": "I don't know based on the provided policy excerpts.", "evidence": []}

        cand_docs = [d for d, _ in candidates]
        reranked = rerank_docs(question, cand_docs, top_n=20)
        k_ctx = dynamic_top_k_by_dropoff(reranked, k_min=2, k_max=8)
        if k_ctx == 0:
            return {"answer": "I don't know based on the provided policy excerpts.", "evidence": []}
        selected = reranked[:k_ctx]

        # 你的 reject 逻辑也可以保留
        top_rerank = reranked[0][1]
        if top_rerank < 0.2:
            return {"answer": "I don't know based on the provided policy excerpts.", "evidence": []}

    # ---- 3) build evidence & answer (use selected docs) ----
    evidence = []
    for doc, rscore in selected[:5]:
        snippet = doc.page_content.strip()
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."
        evidence.append({
            "score": float(rscore),
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "section": doc.metadata.get("section_title", ""),
            "snippet": snippet,
        })

    context = "\n\n".join(
        f"[{i+1}] ({e['source']} | {e['chunk_id']}) {e['snippet']}"
        for i, e in enumerate(evidence[:5])
    )

    chain = PROMPT | llm
    msg = chain.invoke({"question": question, "context": context})
    answer_text = getattr(msg, "content", str(msg))

    return {"question": question, "answer": answer_text, "evidence": evidence[:5]}


# ----------------------------
# Agent-ready endpoint (structured)
# (Does NOT change the existing /ask behavior.)
# ----------------------------
@app.post("/day1/agent_ask", response_model=AnswerResult)
def day1_agent_ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    # Day1: structured search (vector only)
    sr = search_policy_chunks(query=question, top_k=req.top_k)

    if not sr.chunks:
        ar = AnswerResult(
            question=question,
            answer="I don't know based on the provided policy excerpts.",
            citations=[],
            used_chunks=[],
        )
        log_event("day1.agent_ask.no_hits", ar.model_dump())
        return ar

    ar = answer_with_context(question=question, chunks=sr.chunks)
    log_event("day1.agent_ask.done", ar.model_dump())
    return ar

@app.post("/debug/trace_ask")
def debug_trace_ask(req: AskRequest):
    vs = ensure_vectorstore_loaded()
    ensure_graphrag_loaded()

    q = (req.question or "").strip()
    out = {"question": q}

    # A) GraphRAG candidate ids
    q_ents = extract_entities_heuristic(q, max_entities=20)
    cand_ids = graphrag_candidate_chunk_ids(q, hops=2, max_chunks=80)

    out["A_query_entities"] = q_ents
    out["A_cand_ids_count"] = len(cand_ids)
    out["A_cand_ids_sample"] = cand_ids[:10]

    # B) Filter docs by cand ids (does chunk_id match?)
    all_docs = get_all_docs_from_vs(vs)
    out["B_all_docs_count"] = len(all_docs)

    cand_docs = filter_docs_by_chunk_ids(all_docs, cand_ids, limit=300)
    out["B_cand_docs_count"] = len(cand_docs)
    out["B_cand_docs_sample_chunk_ids"] = [d.metadata.get("chunk_id") for d in cand_docs[:5]]

    # C) BM25 on candidate docs
    bm25_hits = []
    if cand_docs:
        bm25 = BM25Retriever.from_documents(cand_docs)
        bm25.k = 30
        bm25_hits = bm25.invoke(q)

    out["C_bm25_hits_count"] = len(bm25_hits)
    out["C_bm25_sample_chunk_ids"] = [d.metadata.get("chunk_id") for d in bm25_hits[:5]]

    # D) Rerank on BM25 hits
    reranked = rerank_docs(q, bm25_hits, top_n=20)
    out["D_reranked_count"] = len(reranked)
    out["D_top_rerank_score"] = float(reranked[0][1]) if reranked else None
    out["D_top_chunk_id"] = reranked[0][0].metadata.get("chunk_id") if reranked else None

    # E) Dynamic top-k decision
    k_ctx = dynamic_top_k_by_dropoff(reranked, k_min=2, k_max=8)
    out["E_k_ctx"] = int(k_ctx)

    # F) Also show fallback vector-only (sanity)
    vec_results = vs.similarity_search_with_score(q, k=5)
    out["F_vec_hits_count"] = len(vec_results)
    out["F_vec_sample_chunk_ids"] = [doc.metadata.get("chunk_id") for doc, _ in vec_results[:5]]

    return out


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Clinic Policy RAG – Explainable UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    textarea { width: 100%; height: 80px; }
    input { width: 120px; }
    button { padding: 8px 12px; margin-top: 8px; }
    .box { border: 1px solid #ddd; padding: 12px; border-radius: 8px; margin-top: 16px; }
    .evi { border-top: 1px dashed #ddd; padding-top: 10px; margin-top: 10px; }
    .muted { color: #666; font-size: 12px; }
    .score { font-weight: bold; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h2>Clinic Policy RAG – Explainable UI</h2>

  <div class="box">
    <div class="muted">Ask a question. The UI will show the answer and the evidence chunks used.</div>
    <textarea id="q" placeholder="e.g., What is the late arrival policy?"></textarea>
    <div style="margin-top:8px;">
      Top K: <input id="topk" type="number" value="5" min="1" max="10" />
      <button onclick="ask()">Ask</button>
    </div>
  </div>

  <div id="status" class="muted"></div>

  <div id="answerBox" class="box" style="display:none;">
    <h3>Answer</h3>
    <pre id="answer"></pre>
  </div>

  <div id="evidenceBox" class="box" style="display:none;">
    <h3>Evidence</h3>
    <div id="evidence"></div>
  </div>

<script>
async function ask(){
  const q = document.getElementById('q').value.trim();
  const topk = parseInt(document.getElementById('topk').value || "5", 10);
  const status = document.getElementById('status');
  status.textContent = "Running...";

  document.getElementById('answerBox').style.display = "none";
  document.getElementById('evidenceBox').style.display = "none";

  try{
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: topk })
    });

    const data = await resp.json();

    if(!resp.ok){
      status.textContent = "Error: " + (data.detail || resp.status);
      return;
    }

    status.textContent = "Done.";

    // Answer
    document.getElementById('answer').textContent = data.answer || "";
    document.getElementById('answerBox').style.display = "block";

    // Evidence
    const eviDiv = document.getElementById('evidence');
    eviDiv.innerHTML = "";
    const evidence = data.evidence || [];
    if(evidence.length === 0){
      eviDiv.innerHTML = "<div class='muted'>(no evidence returned)</div>";
    } else {
      evidence.forEach((e, idx) => {
        const el = document.createElement('div');
        el.className = "evi";
        el.innerHTML = `
          <div><span class="score">#${idx+1} score=${Number(e.score).toFixed(4)}</span>
            <span class="muted"> | source=${e.source} | chunk_id=${e.chunk_id} | section=${e.section || ""}</span>
          </div>
          <pre>${(e.snippet || "").replace(/</g,"&lt;")}</pre>
        `;
        eviDiv.appendChild(el);
      });
    }
    document.getElementById('evidenceBox').style.display = "block";

  }catch(err){
    status.textContent = "Error: " + err;
  }
}
</script>
</body>
</html>
""" 


