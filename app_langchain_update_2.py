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

def rerank_docs(query: str, docs: List[Document], top_n: int = 5) -> List[tuple[Document, float]]:
    if not docs:
        return []
    ce = ensure_reranker_loaded()
    pairs = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs)  # ndarray-like, higher is better
    ranked = sorted(zip(docs, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def dynamic_top_k_by_dropoff(
    reranked: List[tuple[Document, float]],
    k_min: int = 2,
    k_max: int = 8,
    drop_ratio: float = 0.72,
    abs_min: float = 0.18,
) -> int:
    if not reranked:
        return 0
    scores = [float(s) for _, s in reranked]
    s0 = scores[0]

    k = 0
    for s in scores[:k_max]:
        if s < abs_min:
            break
        if k >= k_min and s < s0 * drop_ratio:
            break
        k += 1

    return max(k_min, k) if scores[0] >= abs_min else 0


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

    k = max(1, min(int(req.top_k), ASK_TOPK_MAX))
    
        # 1) Hybrid candidates
    candidates = hybrid_retrieve(vs, question, k_final=k, k_vec=10, k_bm25=10)
    if not candidates:
        return {"answer": "I don't know based on the provided documents.", "evidence": []}

    cand_docs = [d for d, _ in candidates]

    # 2) Rerank (cross-encoder)
    reranked = rerank_docs(question, cand_docs, top_n=20)  # 先取多一点候选
    k_ctx = dynamic_top_k_by_dropoff(reranked, k_min=2, k_max=8)

    if k_ctx == 0:
        return {"answer": "I don't know based on the provided policy excerpts.", "evidence": []}

    selected = reranked[:k_ctx]


    # 3) Optional reject based on rerank score (更稳定：因为越大越相关)
    top_rerank = reranked[0][1]
    RERANK_REJECT_THRESHOLD = 0.2  # 你可以调：越大越严格（模型不同会不同）
    if top_rerank < RERANK_REJECT_THRESHOLD:
        return {"answer": "I don't know based on the provided policy excerpts.", "evidence": []}

    # 4) Build evidence from reranked docs
    evidence = []
    for doc, rscore in reranked[:5]:
        snippet = doc.page_content.strip()
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."
        evidence.append({
            "score": float(rscore),  # 这里用 rerank score
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

    return {
        "question": question,
        "answer": answer_text,
        "evidence": evidence[:5],
    }


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


