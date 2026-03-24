"""
Document ingestion pipeline.

Reads PDFs from the data/ folder, splits them into overlapping chunks,
generates Gemini embeddings, and stores everything in ChromaDB.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import chromadb
from google import genai

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)


# ── PDF text extraction ─────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF, returning a list of {page, text} dicts."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text, "source": pdf_path.name})
    doc.close()
    return pages


# ── Text chunking ────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Embedding helper ─────────────────────────────────────────────────────

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via Google Gemini Embedding API with rate-limit handling."""
    import time

    client = genai.Client(api_key=GOOGLE_API_KEY)
    results: list[list[float]] = []
    # Use small batches to stay within free-tier rate limits
    batch_size = 20
    max_retries = 8

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"    Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks) …")

        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                results.extend([e.values for e in response.embeddings])
                break
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = min(8 * (attempt + 1), 60)
                    print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries}) …")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed to embed batch {batch_num} after {max_retries} retries")

        # Throttle between batches to respect rate limits
        if i + batch_size < len(texts):
            time.sleep(5)

    return results


# ── Main ingestion entrypoint ────────────────────────────────────────────

def ingest_documents(data_dir: Optional[Path] = None, force: bool = False) -> int:
    """
    Ingest all PDFs from `data_dir` into ChromaDB.

    Returns the number of chunks stored.
    """
    data_dir = data_dir or DATA_DIR
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    # Initialise ChromaDB (persistent)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if force:
        # Drop existing collection to re-index
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip if already populated and not forced
    if collection.count() > 0 and not force:
        print(f"Collection already has {collection.count()} chunks. Use force=True to re-index.")
        return collection.count()

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)
        for page_info in pages:
            chunks = chunk_text(page_info["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{page_info['source']}:p{page_info['page']}:c{idx}".encode()
                ).hexdigest()
                all_chunks.append(chunk)
                all_metadatas.append(
                    {
                        "source": page_info["source"],
                        "page": page_info["page"],
                        "chunk_index": idx,
                    }
                )
                all_ids.append(chunk_id)

    print(f"  Generating embeddings for {len(all_chunks)} chunks …")
    embeddings = _get_embeddings(all_chunks)

    print("  Storing in ChromaDB …")
    # Upsert in batches (ChromaDB limit is ~5461 per call)
    batch = 5000
    for i in range(0, len(all_chunks), batch):
        collection.upsert(
            ids=all_ids[i : i + batch],
            documents=all_chunks[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=all_metadatas[i : i + batch],
        )

    total = collection.count()
    print(f"  ✓ Ingested {total} chunks into '{CHROMA_COLLECTION_NAME}'")
    return total
