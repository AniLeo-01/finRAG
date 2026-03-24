# FinRAG — Financial Document RAG + MCP Agent

A multi-agent financial document analysis system built on **Google ADK**, **FastMCP**, **ChromaDB**, and **Google Gemini**. It combines retrieval-augmented generation over SEC filings with real-time financial tools, orchestrated by a top-level agent that routes queries to the right specialist.

---

## Architecture

```
                         ┌─────────────────────────┐
                         │   Orchestrator Agent     │
                         │  (finrag_orchestrator)   │
                         └────────┬────────┬────────┘
                  Document Q&A    │        │   Tools / Lookups
                    ┌─────────────┘        └──────────────┐
                    ▼                                     ▼
          ┌──────────────────┐               ┌─────────────────────┐
          │   RAG Sub-Agent  │               │   MCP Sub-Agent     │
          │   (rag_agent)    │               │   (mcp_agent)       │
          └───────┬──────────┘               └──────────┬──────────┘
                  │                                     │ stdio JSON-RPC
        ┌─────────┴─────────┐               ┌──────────┴──────────┐
        │  ChromaDB + Gemini│               │  FastMCP Server     │
        │  Embeddings       │               │  (8 tools)          │
        └───────────────────┘               └─────────────────────┘
```

**RAG Agent** answers questions about financial document contents (10-K, Form 4, etc.) by searching a ChromaDB vector store populated with Gemini embeddings.

**MCP Agent** connects to a FastMCP tool server exposing financial analysis tools (stock lookups, ratio calculators, SEC filing search) and generic utilities (calculator, unit converter, date/time, JSON formatter).

The **Orchestrator** analyzes user intent and routes to the appropriate sub-agent — or chains both when a question needs document data plus live tools.

---

## Quick Start

### 1. Get a Google API Key

Get one from [Google AI Studio](https://aistudio.google.com/apikey). A paid-tier key is recommended to avoid free-tier rate limits.

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-key-here
```

### 3. Install

```bash
pip install -e .
```

### 4. Ingest Documents

```bash
python ingest_cli.py          # index PDFs from data/
python ingest_cli.py --force  # re-index from scratch
```

### 5. Run

```bash
# Interactive chat (full orchestrator)
python main.py

# Single query
python main.py --query "What was Netflix's total revenue?"

# Ingest then chat
python main.py --ingest

# ADK Web UI (RAG agent only)
adk web .
```

---

## Project Structure

```
finrag/
├── main.py                     # CLI entry point (orchestrator + both agents)
├── agent.py                    # ADK entry point for `adk web` (RAG agent only)
├── ingest_cli.py               # Standalone document ingestion script
├── pyproject.toml              # Dependencies and project metadata
├── .env.example                # Environment variable template
│
├── config/
│   └── settings.py             # Centralized config (env vars + defaults)
│
├── rag/                        # RAG pipeline
│   ├── ingest.py               # PDF → chunks → Gemini embeddings → ChromaDB
│   └── retriever.py            # Query → vector search → ranked chunks
│
├── mcp_server/
│   └── server.py               # FastMCP tool server (financial + utility tools)
│
├── agents/                     # Google ADK agent definitions
│   ├── rag_agent.py            # RAG sub-agent (FunctionTool-based)
│   ├── mcp_agent.py            # MCP sub-agent (McpToolset via stdio)
│   └── orchestrator.py         # Top-level orchestrator with routing
│
├── data/                       # Source PDFs (add your financial documents here)
│   ├── form_10-K_annual_filing.pdf
│   └── form_4.pdf
│
└── docs/
    ├── ARCHITECTURE.md
    └── FinRAG_System_Architecture.docx
```

---

## Available MCP Tools

### Financial Tools

| Tool | Description |
|------|-------------|
| `stock_price_lookup` | Get the latest stock price for a ticker symbol |
| `calculate_financial_ratios` | Compute profitability, leverage, and liquidity ratios |
| `sec_filing_search` | Search for SEC filings by company and type |
| `compare_financials` | Compare a metric between two companies |

### Utility Tools

| Tool | Description |
|------|-------------|
| `calculator` | Evaluate mathematical expressions safely |
| `current_date_time` | Get current date/time in ISO-8601 |
| `unit_converter` | Convert between units (km/miles, millions/billions, etc.) |
| `json_formatter` | Parse and pretty-print JSON strings |

> **Note:** Financial tools currently use simulated data. Each tool is designed with the same interface a production API (Alpha Vantage, Yahoo Finance, EDGAR) would use — swap the implementation body to go live.

---

## Configuration

All settings are in `config/settings.py` and read from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM for agent reasoning |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model for document embeddings |
| `CHROMA_PERSIST_DIR` | auto-detected | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks returned per query |

---

## Extending

**Add documents:** Drop PDFs into `data/` and run `python ingest_cli.py --force`.

**Add MCP tools:** Add a `@mcp.tool()` function to `mcp_server/server.py` — the MCP agent discovers it automatically at startup.

**Add sub-agents:** Create a new agent in `agents/`, add it to the orchestrator's `sub_agents` list, and update the routing instructions.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Google ADK |
| LLM | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 |
| Vector Store | ChromaDB (persistent, cosine HNSW) |
| MCP Server | FastMCP 3.x (stdio transport) |
| PDF Processing | PyMuPDF |
| Language | Python 3.11+ |
