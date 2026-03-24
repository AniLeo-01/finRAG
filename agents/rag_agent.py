"""
RAG Sub-Agent – answers questions by retrieving context from financial documents.

Uses Google ADK's FunctionTool to wrap the RAG retrieval + generation pipeline
so the orchestrator can delegate document-based questions here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GEMINI_MODEL, DATA_DIR
from rag.ingest import ingest_documents
from rag.retriever import retrieve_chunks


# ── Tool functions exposed to the LLM ────────────────────────────────────

def search_financial_documents(query: str, top_k: int = 5) -> dict:
    """
    Search the ingested financial documents (10-K, Form 4, etc.) and return
    the most relevant text passages.

    Args:
        query: The user's natural-language question about the financial documents.
        top_k: Number of top chunks to return (default 5).

    Returns:
        A dict containing the retrieved chunks with source metadata.
    """
    try:
        chunks = retrieve_chunks(query=query, top_k=top_k)
        return {
            "status": "success",
            "num_results": len(chunks),
            "results": chunks,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def ingest_new_documents(force_reindex: bool = False) -> dict:
    """
    Ingest (or re-ingest) PDF documents from the data/ folder into the
    vector store. Call this if documents have been added or updated.

    Args:
        force_reindex: If True, drop existing index and rebuild from scratch.

    Returns:
        Status message with the number of chunks indexed.
    """
    try:
        count = ingest_documents(data_dir=DATA_DIR, force=force_reindex)
        return {"status": "success", "chunks_indexed": count}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def list_available_documents() -> dict:
    """
    List all PDF documents currently available in the data/ directory.

    Returns:
        A dict with file names and sizes.
    """
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    files = []
    for p in pdfs:
        files.append({"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)})
    return {"documents": files, "count": len(files)}


# ── Build the ADK Agent ──────────────────────────────────────────────────

rag_agent = Agent(
    name="rag_agent",
    model=GEMINI_MODEL,
    description=(
        "Specializes in answering questions about financial documents "
        "(SEC filings like 10-K annual reports, Form 4 insider-trading "
        "disclosures, etc.) using retrieval-augmented generation over "
        "a ChromaDB vector store."
    ),
    instruction=(
        "You are a financial document analyst. When the user asks a question "
        "about financial filings, company performance, insider transactions, "
        "or any information that may be in the ingested documents:\n\n"
        "1. Use `search_financial_documents` to retrieve relevant passages.\n"
        "2. Synthesize a clear, well-sourced answer citing the document name "
        "   and page number for every claim.\n"
        "3. If the documents haven't been ingested yet, call `ingest_new_documents` first.\n"
        "4. If you can't find the answer in the documents, say so clearly.\n"
        "5. Use `list_available_documents` to tell the user which filings are available.\n\n"
        "Always ground your answers in the retrieved text — do not fabricate information."
    ),
    tools=[
        FunctionTool(search_financial_documents),
        FunctionTool(ingest_new_documents),
        FunctionTool(list_available_documents),
    ],
)
