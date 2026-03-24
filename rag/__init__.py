"""RAG module – PDF ingestion, chunking, embedding, and retrieval."""

from rag.ingest import ingest_documents
from rag.retriever import retrieve_chunks

__all__ = ["ingest_documents", "retrieve_chunks"]
