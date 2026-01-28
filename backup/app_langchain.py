import os
import re
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage

# LangChain community
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Memory
from langchain_classic.memory import ConversationBufferMemory

# Agents
from langchain.agents import initialize_agent, AgentType

# LLMs
# Option 1: Ollama (local)
from langchain_community.chat_models import ChatOllama
# Option 2: OpenAI
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


# ----------------------------
# Config
# ----------------------------
DOCS_DIR = "./docs"
INDEX_DIR = "./index_store_lc"
os.makedirs(INDEX_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# A simple in-memory session store (for demo)
# Real production: Redis / DB keyed by user_id/session_id
MEMORY_STORE: Dict[str, ConversationBufferMemory] = {}

app = FastAPI(title="LangChain Agentic RAG (FAISS local)")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

vectorstore: Optional[FAISS] = None


# ----------------------------
# Schemas
# ----------------------------
class IngestResponse(BaseModel):
    docs_count: int
    chunks_count: int
    index_dir: str

class AskRequest(BaseModel):
    session_id: str = "default"
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    chat_history: List[Dict[str, str]]


# ----------------------------
# Utilities
# ----------------------------
def list_docs(docs_dir: str) -> List[str]:
    exts = {".txt", ".md"}
    paths = []
    for root, _, files in os.walk(docs_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                paths.append(os.path.join(root, name))
    return sorted(paths)

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_vectorstore() -> FAISS:
    global vectorstore
    if vectorstore is not None:
        return vectorstore
    # FAISS.load_local needs allow_dangerous_deserialization=True for some setups
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        vectorstore = None
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="Vectorstore not found. Run POST /ingest first.")
    return vectorstore

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in MEMORY_STORE:
        MEMORY_STORE[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return MEMORY_STORE[session_id]

def get_llm():
    """
    Choose LLM by environment:
      - If OLLAMA_MODEL is set -> use ChatOllama
      - Else if OPENAI_API_KEY is set -> use ChatOpenAI
    """
    ollama_model = os.getenv("OLLAMA_MODEL")
    openai_key = os.getenv("OPENAI_API_KEY")

    if ollama_model:
        return ChatOllama(model=ollama_model, temperature=0)
    if openai_key:
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed. Run: pip install langchain-openai")
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    raise RuntimeError(
        "No LLM configured. Set OLLAMA_MODEL (recommended) or OPENAI_API_KEY."
    )

def build_sources(docs) -> List[Dict[str, Any]]:
    """
    Convert LangChain Documents to a simple list for response.
    """
    out = []
    for d in docs:
        out.append({
            "source": d.metadata.get("source", ""),
            "chunk_id": d.metadata.get("chunk_id", None),
            "chunk_in_doc": d.metadata.get("chunk_in_doc", None),
            "text": d.page_content
        })
    return out


# ----------------------------
# Tools
# ----------------------------
def retrieval_tool_fn(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Tool: retrieve relevant chunks from local FAISS.
    Returns: {"hits": [...]} where each hit includes source + text.
    """
    vs = load_vectorstore()
    k = max(1, min(int(top_k), 20))
    docs = vs.similarity_search(query, k=k)  # returns List[Document]
    return {"hits": build_sources(docs)}

# A safe calculator (very minimal)
def calculator_tool_fn(expression: str) -> Dict[str, Any]:
    """
    Tool: calculate simple arithmetic like "3*10" or "30/100".
    NOTE: Keep it strict; do NOT eval arbitrary code.
    """
    expr = expression.strip()
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return {"error": "Only digits and + - * / ( ) are allowed."}
    try:
        value = eval(expr, {"__builtins__": {}}, {})
        return {"value": float(value)}
    except Exception as e:
        return {"error": f"Bad expression: {e}"}

retrieval_tool = StructuredTool.from_function(
    name="retrieve_chunks",
    description="Search the local policy/docs knowledge base and return relevant text chunks with citations.",
    func=retrieval_tool_fn,
)

calculator_tool = StructuredTool.from_function(
    name="calculator",
    description="Do simple arithmetic calculations (e.g., 3*10, (30/100)*80).",
    func=calculator_tool_fn,
)


# ----------------------------
# Agent builder
# ----------------------------
def build_agent_executor(session_id: str) -> AgentExecutor:
    """
    Build a tool-calling agent (function-calling style) with memory.
    """
    llm = get_llm()

    # System+user prompt: enforce "retrieve before answer"
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful assistant. "
         "When answering policy questions, you MUST first call retrieve_chunks to gather evidence. "
         "Use the retrieved chunks as the ONLY source of truth. "
         "If the evidence is insufficient, say you don't know based on the provided context. "
         "When calculations are needed, call calculator."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [retrieval_tool, calculator_tool]


    memory = get_memory(session_id)

    executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 通用、对非OpenAI也更稳
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "embeddings": EMBED_MODEL_NAME,
        "docs_abs": os.path.abspath(DOCS_DIR),
        "index_abs": os.path.abspath(INDEX_DIR),
        "llm_mode": "ollama" if os.getenv("OLLAMA_MODEL") else ("openai" if os.getenv("OPENAI_API_KEY") else "none")
    }

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    global vectorstore

    doc_paths = list_docs(DOCS_DIR)
    if not doc_paths:
        raise HTTPException(status_code=400, detail=f"No .txt/.md files found in {DOCS_DIR}")

    texts = []
    metadatas = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    for p in doc_paths:
        raw = read_text_file(p)
        raw = raw.strip()
        if not raw:
            continue

        chunks = splitter.split_text(raw)
        rel = os.path.relpath(p, DOCS_DIR)

        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({
                "source": rel,
                "chunk_in_doc": i,
                "chunk_id": len(texts) - 1,
            })

    if not texts:
        raise HTTPException(status_code=400, detail="No chunks produced. Check docs content.")

    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vectorstore.save_local(INDEX_DIR)

    return IngestResponse(
        docs_count=len(doc_paths),
        chunks_count=len(texts),
        index_dir=os.path.abspath(INDEX_DIR)
    )

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    # Build agent for this session
    executor = build_agent_executor(req.session_id)

    # Run agent
    result = executor.invoke({"input": question})

    answer = result.get("output", "")
    intermediate = result.get("intermediate_steps", [])

    # Extract retrieval outputs (sources)
    sources: List[Dict[str, Any]] = []
    for step in intermediate:
        # step is typically (AgentAction, observation)
        try:
            action, observation = step
            if getattr(action, "tool", "") == "retrieve_chunks":
                # observation is dict-like: {"hits": [...]}
                if isinstance(observation, dict) and "hits" in observation:
                    sources.extend(observation["hits"])
        except Exception:
            continue

    # Render chat history
    mem = get_memory(req.session_id)
    history = []
    for m in mem.chat_memory.messages:
        role = "system" if isinstance(m, SystemMessage) else getattr(m, "type", "unknown")
        history.append({"role": role, "content": m.content})

    return AskResponse(
        session_id=req.session_id,
        question=question,
        answer=answer,
        sources=sources[:50],
        chat_history=history
    )
