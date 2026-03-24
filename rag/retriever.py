"""
Retriever – queries ChromaDB and returns the most relevant document chunks.
"""

from __future__ import annotations

import chromadb
from google import genai

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    TOP_K_RESULTS,
)


def _embed_query(query: str) -> list[float]:
    """Embed a single query string using Gemini."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
    )
    return response.embeddings[0].values


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K_RESULTS,
    filter_source: str | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Returns a list of dicts with keys: text, source, page, score.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

    query_embedding = _embed_query(query)

    where_filter = {"source": filter_source} if filter_source else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", -1),
                "score": round(1 - dist, 4),  # cosine similarity
            }
        )

    return chunks
