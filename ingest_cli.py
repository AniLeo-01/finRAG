"""
Standalone ingestion script – run this to populate the vector store.

Usage:
    python ingest_cli.py              # ingest (skip if already done)
    python ingest_cli.py --force      # force re-index from scratch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.ingest import ingest_documents


def main():
    parser = argparse.ArgumentParser(description="Ingest financial PDFs into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force re-index")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    print("Starting document ingestion …")
    count = ingest_documents(data_dir=data_dir, force=args.force)
    print(f"\nDone — {count} total chunks in vector store.")


if __name__ == "__main__":
    main()
