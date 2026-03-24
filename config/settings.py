"""
Central configuration for the FinRAG system.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
_default_chroma_dir = str(PROJECT_ROOT / "chroma_db")
# ChromaDB's SQLite backend requires a filesystem that supports file locking.
# If PROJECT_ROOT is on a network/FUSE mount, fall back to a local temp directory.
try:
    _test_dir = Path(_default_chroma_dir)
    _test_dir.mkdir(parents=True, exist_ok=True)
    _test_file = _test_dir / ".lock_test"
    _test_file.write_text("test")
    _test_file.unlink()
    _chroma_fallback = _default_chroma_dir
except OSError:
    import tempfile
    _chroma_fallback = os.path.join(tempfile.gettempdir(), "finrag_chroma_db")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", _chroma_fallback)

# ── Google Gemini ────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# ── ChromaDB ─────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "financial_documents"

# ── RAG parameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between chunks
TOP_K_RESULTS = 5          # number of chunks to retrieve

# ── MCP Server ───────────────────────────────────────────────────────────
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
