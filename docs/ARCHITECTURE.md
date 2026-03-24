# FinRAG – System Architecture Documentation

## 1. Executive Summary

FinRAG is a multi-agent financial document analysis system built on **Google Agent Development Kit (ADK)** with a **FastMCP**-based tool server. It combines Retrieval-Augmented Generation (RAG) over SEC filings with real-time financial tools, orchestrated by a top-level agent that intelligently routes user queries to the appropriate specialist sub-agent.

The system ingests PDF financial documents (10-K annual reports, Form 4 insider-trading disclosures, etc.), chunks and embeds them into a ChromaDB vector store using Google Gemini embeddings, and makes them queryable through natural language. Simultaneously, a FastMCP server exposes financial analysis tools (ratio calculators, stock lookups, filing search) and generic utilities that the orchestrator can invoke on demand.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER / CLIENT                            │
│              (CLI REPL, ADK Web UI, or API call)                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │  natural-language query
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                              │
│                 (finrag_orchestrator)                             │
│                                                                  │
│  Google ADK Agent  ·  Model: gemini-2.0-flash                    │
│  Routes queries to the appropriate sub-agent based on intent     │
│                                                                  │
│  Routing Logic:                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────┐     │
│  │ Document content?   │───▶│ Delegate to rag_agent        │     │
│  │ Filing questions?   │    │                              │     │
│  └─────────────────────┘    └──────────────────────────────┘     │
│  ┌─────────────────────┐    ┌──────────────────────────────┐     │
│  │ Calculate / lookup? │───▶│ Delegate to mcp_agent        │     │
│  │ Compare / convert?  │    │                              │     │
│  └─────────────────────┘    └──────────────────────────────┘     │
│  ┌─────────────────────┐    ┌──────────────────────────────┐     │
│  │ Both needed?        │───▶│ Chain: rag_agent → mcp_agent │     │
│  └─────────────────────┘    └──────────────────────────────┘     │
└──────────┬──────────────────────────────┬────────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐     ┌──────────────────────────────────────┐
│    RAG SUB-AGENT    │     │         MCP SUB-AGENT                │
│    (rag_agent)      │     │         (mcp_agent)                  │
│                     │     │                                      │
│  ADK Agent with     │     │  ADK Agent connected to FastMCP      │
│  FunctionTools:     │     │  server via stdio transport:         │
│                     │     │                                      │
│  • search_financial │     │  Financial Tools:                    │
│    _documents       │     │  • stock_price_lookup                │
│  • ingest_new       │     │  • calculate_financial_ratios        │
│    _documents       │     │  • sec_filing_search                 │
│  • list_available   │     │  • compare_financials                │
│    _documents       │     │                                      │
│         │           │     │  Utility Tools:                      │
│         ▼           │     │  • calculator                        │
│  ┌─────────────┐    │     │  • current_date_time                 │
│  │  ChromaDB   │    │     │  • unit_converter                    │
│  │  Vector     │    │     │  • json_formatter                    │
│  │  Store      │    │     │         │                            │
│  └──────┬──────┘    │     └─────────┼────────────────────────────┘
│         │           │               │
│    ┌────┴─────┐     │               ▼
│    │  Gemini  │     │     ┌──────────────────────┐
│    │ Embedding│     │     │   FastMCP Server      │
│    │   API    │     │     │   (stdio transport)   │
│    └──────────┘     │     │                       │
└─────────────────────┘     │   Python process      │
                            │   spawned by ADK      │
                            │   MCPToolset          │
                            └──────────────────────┘
```

---

## 3. Project Structure

```
finrag/
├── pyproject.toml              # Project metadata and dependencies
├── .env.example                # Environment variable template
├── main.py                     # Full orchestrator runner (CLI entry point)
├── agent.py                    # ADK-compatible agent definition (for `adk run`)
├── ingest_cli.py               # Standalone document ingestion script
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized configuration (env vars + defaults)
│
├── rag/                        # RAG pipeline module
│   ├── __init__.py
│   ├── ingest.py               # PDF extraction → chunking → embedding → ChromaDB
│   └── retriever.py            # Query embedding → vector search → ranked chunks
│
├── mcp_server/                 # FastMCP tool server
│   ├── __init__.py
│   └── server.py               # Tool definitions + standalone runner
│
├── agents/                     # Google ADK agent definitions
│   ├── __init__.py
│   ├── rag_agent.py            # RAG sub-agent (FunctionTool-based)
│   ├── mcp_agent.py            # MCP sub-agent (MCPToolset-based)
│   └── orchestrator.py         # Top-level orchestrator with routing
│
├── data/                       # Source financial documents
│   ├── form_10-K_annual_filing.pdf   # Netflix 10-K (122 pages)
│   └── form_4.pdf                    # Netflix Form 4 insider filing (2 pages)
│
├── chroma_db/                  # ChromaDB persistence directory (auto-created)
│
└── docs/
    └── ARCHITECTURE.md         # This document
```

---

## 4. Component Deep-Dive

### 4.1 Configuration Layer (`config/settings.py`)

All runtime configuration is centralized in a single module that reads from environment variables (via `python-dotenv`) with sensible defaults. This ensures the system works out of the box while remaining configurable for production deployments.

Key configuration categories:

- **Paths**: `PROJECT_ROOT`, `DATA_DIR`, `CHROMA_PERSIST_DIR` — file system locations.
- **Gemini API**: `GOOGLE_API_KEY`, `GEMINI_MODEL` (default `gemini-2.0-flash`), `EMBEDDING_MODEL` (default `models/text-embedding-004`).
- **ChromaDB**: `CHROMA_COLLECTION_NAME`, persistence directory.
- **RAG parameters**: `CHUNK_SIZE` (1000 chars), `CHUNK_OVERLAP` (200 chars), `TOP_K_RESULTS` (5).
- **MCP Server**: `MCP_SERVER_HOST`, `MCP_SERVER_PORT`.

### 4.2 RAG Pipeline (`rag/`)

The RAG pipeline handles the full lifecycle from raw PDF to queryable knowledge base.

#### 4.2.1 Document Ingestion (`rag/ingest.py`)

The ingestion pipeline follows four stages:

1. **PDF Text Extraction** — Uses PyMuPDF (`fitz`) to extract text page-by-page from all PDFs in the `data/` directory. Each page yields a `{page, text, source}` record preserving provenance.

2. **Text Chunking** — Splits extracted text into overlapping chunks using a sliding-window approach. The default window is 1000 characters with 200-character overlap, ensuring context continuity across chunk boundaries. This is critical for financial documents where key figures often appear near section headings.

3. **Embedding Generation** — Each chunk is embedded via the Google Gemini Embedding API (`models/text-embedding-004`). Embeddings are generated in batches of 96 (API limit) for efficiency. The embedding model produces 768-dimensional vectors optimized for semantic similarity.

4. **Vector Storage** — Chunks, embeddings, and metadata are upserted into a ChromaDB persistent collection configured with cosine similarity (`hnsw:space: cosine`). Each chunk is identified by an MD5 hash of `source:page:chunk_index` to enable idempotent re-ingestion.

```
PDF Files ──▶ PyMuPDF ──▶ Page Texts ──▶ Chunker ──▶ Chunks
                                                        │
                                                        ▼
ChromaDB ◀── Upsert ◀── Gemini Embeddings ◀── Batch Embed API
```

**Idempotency**: The ingestion checks `collection.count()` before processing. If the collection already has data and `force=False`, it skips re-ingestion. The `force=True` flag drops and rebuilds the entire index.

#### 4.2.2 Retrieval (`rag/retriever.py`)

The retriever converts a natural-language query into a vector, searches ChromaDB, and returns ranked results:

1. **Query Embedding** — The user query is embedded using the same Gemini model to ensure vector-space alignment with the stored document embeddings.

2. **Vector Search** — ChromaDB's HNSW index performs approximate nearest-neighbor search, returning the top-k most similar chunks. An optional `filter_source` parameter restricts search to a specific document (e.g., only the 10-K).

3. **Result Formatting** — Raw ChromaDB results are transformed into a clean list of `{text, source, page, score}` dictionaries. Scores are converted from cosine distance to cosine similarity (`1 - distance`).

### 4.3 FastMCP Server (`mcp_server/server.py`)

The MCP server is built with **FastMCP** and exposes tools via the Model Context Protocol. It runs as a subprocess communicating over stdio, which the ADK's `MCPToolset` connects to automatically.

#### 4.3.1 Financial Tools

| Tool | Purpose | Inputs |
|------|---------|--------|
| `calculate_financial_ratios` | Computes profitability, leverage, and liquidity ratios | Revenue, net income, assets, liabilities, equity |
| `stock_price_lookup` | Retrieves latest stock price for a ticker | Ticker symbol (e.g., "NFLX") |
| `sec_filing_search` | Searches for SEC filings by company and type | Company name, filing type |
| `compare_financials` | Compares a metric between two companies | Two company names, metric name |

**Note**: The current implementation uses simulated data for demonstration. Each tool is designed with the same interface signature that a production API (Alpha Vantage, Yahoo Finance, EDGAR FULL-TEXT) would use, making the swap straightforward.

#### 4.3.2 Generic Utility Tools

| Tool | Purpose |
|------|---------|
| `calculator` | Safe mathematical expression evaluator |
| `current_date_time` | Returns current datetime in ISO-8601 |
| `json_formatter` | Parses and pretty-prints JSON |
| `unit_converter` | Converts between units (including financial: millions ↔ billions) |

#### 4.3.3 Transport

The server uses **stdio transport** — it reads JSON-RPC messages from stdin and writes responses to stdout. This is the standard MCP communication pattern and is how Google ADK's `MCPToolset` connects to it. The ADK spawns the server as a child process and manages the lifecycle automatically.

### 4.4 Google ADK Agents (`agents/`)

#### 4.4.1 RAG Agent (`agents/rag_agent.py`)

The RAG agent wraps the retrieval pipeline in ADK `FunctionTool` objects, making the RAG capabilities callable by the LLM. It exposes three tools:

- **`search_financial_documents(query, top_k)`** — The primary retrieval tool. Takes a natural-language question and returns the top-k most relevant chunks with source metadata.
- **`ingest_new_documents(force_reindex)`** — Triggers document ingestion. The agent calls this automatically if the vector store is empty.
- **`list_available_documents()`** — Lists all PDFs in the data directory with file sizes.

The agent's instruction prompt enforces a strict grounding policy: answers must cite document name and page number, and the agent must explicitly state when it cannot find an answer in the documents.

#### 4.4.2 MCP Agent (`agents/mcp_agent.py`)

The MCP agent uses ADK's `MCPToolset` to dynamically discover and connect to the FastMCP server's tools. The connection is established asynchronously at agent creation time:

```python
tools, exit_stack = await MCPToolset.from_server(
    connection_params=StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    ),
)
```

This pattern means the MCP agent's available tools are determined at runtime by whatever the FastMCP server exposes — adding a new tool to `server.py` automatically makes it available to the agent without any changes to the agent code.

The `exit_stack` is an `AsyncExitStack` that manages the lifecycle of the MCP server subprocess. It must be closed when the orchestrator shuts down.

#### 4.4.3 Orchestrator Agent (`agents/orchestrator.py`)

The orchestrator is the root agent that receives all user queries. It uses ADK's built-in **sub-agent delegation** mechanism — the `sub_agents` parameter tells ADK that this agent can route to child agents.

**Routing strategy** (defined in the instruction prompt):

| User Intent | Routed To | Example |
|------------|-----------|---------|
| Document content questions | `rag_agent` | "What was Netflix's revenue in the 10-K?" |
| Calculations / lookups | `mcp_agent` | "What's the current NFLX stock price?" |
| Comparisons | `mcp_agent` | "Compare Netflix and Apple revenue" |
| Hybrid questions | `rag_agent` → `mcp_agent` | "What was the 10-K revenue, and what's the stock price now?" |

The orchestrator synthesizes a unified response from the sub-agent outputs, maintaining citation integrity for document-sourced claims.

---

## 5. Data Flow Diagrams

### 5.1 Document Ingestion Flow

```
User runs: python ingest_cli.py
                │
                ▼
    ┌───────────────────────┐
    │  Scan data/ for PDFs  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  For each PDF:        │
    │  PyMuPDF extracts     │
    │  text page-by-page    │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Chunk text with      │
    │  1000-char windows    │
    │  200-char overlap     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Generate MD5 IDs     │
    │  for deduplication    │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐      ┌───────────────────┐
    │  Batch embed via      │─────▶│  Gemini Embedding  │
    │  Gemini API (96/call) │◀─────│  API               │
    └───────────┬───────────┘      └───────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Upsert into ChromaDB │
    │  (cosine HNSW index)  │
    └───────────────────────┘
```

### 5.2 Query Execution Flow

```
User: "What risk factors does Netflix mention in its 10-K?"
                │
                ▼
    ┌───────────────────────────────────────┐
    │  Orchestrator receives query          │
    │  Gemini analyzes intent               │
    │  Decision: document content → rag_agent│
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │  rag_agent calls                      │
    │  search_financial_documents(          │
    │    "Netflix risk factors 10-K",       │
    │    top_k=5                            │
    │  )                                    │
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │  Retriever:                           │
    │  1. Embed query via Gemini            │
    │  2. Search ChromaDB (cosine sim)      │
    │  3. Return top-5 chunks with metadata │
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │  rag_agent synthesizes answer         │
    │  from retrieved chunks, citing        │
    │  source document + page numbers       │
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │  Orchestrator presents unified        │
    │  response to user                     │
    └───────────────────────────────────────┘
```

### 5.3 MCP Tool Invocation Flow

```
User: "Calculate the profit margin if revenue is $39B and net income is $8.7B"
                │
                ▼
    ┌───────────────────────────────────────┐
    │  Orchestrator routes to mcp_agent     │
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │  mcp_agent selects tool:              │
    │  calculate_financial_ratios(          │
    │    revenue=39e9,                      │
    │    net_income=8.7e9, ...              │
    │  )                                    │
    └───────────────────┬───────────────────┘
                        │  JSON-RPC over stdio
                        ▼
    ┌───────────────────────────────────────┐
    │  FastMCP Server                       │
    │  Executes tool function               │
    │  Returns JSON result                  │
    └───────────────────┬───────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │  mcp_agent formats result:            │
    │  "Profit margin: 22.31%              │
    │   ROA: ... ROE: ..."                  │
    └───────────────────────────────────────┘
```

---

## 6. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Agent Framework | Google ADK | ≥ 0.5.0 | Agent orchestration, sub-agent routing, tool management |
| LLM | Google Gemini | 2.0 Flash | Agent reasoning, response generation |
| Embeddings | Gemini Embedding API | text-embedding-004 | 768-dim semantic embeddings for RAG |
| Vector Store | ChromaDB | ≥ 0.5.0 | Persistent vector storage with HNSW indexing |
| MCP Framework | FastMCP | ≥ 2.0.0 | Tool server with Model Context Protocol compliance |
| PDF Processing | PyMuPDF (fitz) | ≥ 1.24.0 | High-fidelity text extraction from financial PDFs |
| HTTP Client | httpx | ≥ 0.27.0 | Async HTTP for potential API integrations |
| Config | python-dotenv | ≥ 1.0.0 | Environment variable management |
| Language | Python | ≥ 3.11 | Core runtime |

---

## 7. Agent Communication Patterns

### 7.1 ADK Sub-Agent Delegation

Google ADK implements sub-agent routing through the orchestrator's `sub_agents` list. When the orchestrator's LLM determines that a query should be handled by a sub-agent, ADK:

1. Passes the relevant context to the sub-agent
2. The sub-agent executes its tools and generates a response
3. The response flows back to the orchestrator
4. The orchestrator may synthesize or pass through the response

This is a **hierarchical delegation** pattern — the orchestrator maintains conversational state while sub-agents are stateless specialists.

### 7.2 MCP Communication (stdio)

The MCP agent ↔ FastMCP server communication uses the standard MCP protocol over stdio:

```
ADK Process                          FastMCP Process
    │                                      │
    │──── JSON-RPC request ──────────────▶│
    │     {"method": "tools/call",        │
    │      "params": {"name": "...",      │
    │                 "arguments": {...}}} │
    │                                      │
    │◀─── JSON-RPC response ──────────────│
    │     {"result": {"content": [...]}}  │
    │                                      │
```

The `MCPToolset` handles serialization/deserialization automatically. Tool schemas are discovered via the `tools/list` MCP method at connection time.

### 7.3 FunctionTool Pattern (RAG Agent)

Unlike the MCP agent, the RAG agent uses ADK's `FunctionTool` wrapper, which converts Python functions directly into tools the LLM can call:

```python
FunctionTool(search_financial_documents)
# ADK auto-generates tool schema from function signature + docstring
```

This is simpler than MCP for tools that run in-process and don't need cross-process isolation.

---

## 8. RAG Design Decisions

### 8.1 Chunking Strategy

Financial documents present unique chunking challenges — tables, multi-column layouts, and cross-referenced sections. The current character-based sliding window (1000 chars, 200 overlap) is a pragmatic starting point. Potential improvements:

- **Semantic chunking**: Split on section headers (Item 1, Item 1A, etc.) for 10-K filings
- **Table-aware chunking**: Detect and preserve tabular data as atomic chunks
- **Recursive splitting**: Use paragraph → sentence → character hierarchy

### 8.2 Embedding Model Choice

`text-embedding-004` was selected for:
- **768 dimensions**: Good balance of expressiveness and storage efficiency
- **Cosine similarity**: Natively optimized for semantic similarity tasks
- **Batch support**: Up to 100 texts per API call, reducing latency
- **Gemini ecosystem**: Consistent with the ADK's Gemini-first design

### 8.3 ChromaDB Configuration

- **Persistence**: Enabled via `PersistentClient` to survive process restarts
- **Distance metric**: Cosine (`hnsw:space: cosine`) for semantic similarity
- **HNSW index**: Default parameters for approximate nearest-neighbor search
- **Idempotent IDs**: MD5 hashes ensure re-ingestion doesn't create duplicates

---

## 9. Security Considerations

- **API Keys**: Stored in `.env` (gitignored), never hardcoded. The `.env.example` template documents required variables without exposing secrets.
- **Calculator Tool**: Uses a restricted `eval()` with an empty `__builtins__` namespace and a whitelist of math functions. This prevents arbitrary code execution while supporting complex expressions.
- **MCP Isolation**: The FastMCP server runs as a separate process. If a tool crashes, it doesn't take down the main agent process.
- **Input Validation**: All MCP tool functions validate inputs via type hints enforced by FastMCP's schema generation.

---

## 10. Extensibility Guide

### 10.1 Adding New Financial Documents

1. Place PDF files in the `data/` directory
2. Run `python ingest_cli.py --force` to re-index
3. The RAG agent automatically has access to the new content

### 10.2 Adding New MCP Tools

Add a new `@mcp.tool()` decorated function in `mcp_server/server.py`:

```python
@mcp.tool()
def my_new_tool(param: str) -> dict:
    """Tool description (becomes the LLM-visible schema)."""
    return {"result": "..."}
```

The MCP agent discovers the tool automatically at startup — no agent code changes needed.

### 10.3 Adding New Sub-Agents

1. Create a new agent file in `agents/` following the `rag_agent.py` pattern
2. Add it to the orchestrator's `sub_agents` list in `orchestrator.py`
3. Update the orchestrator's instruction prompt with routing rules for the new agent

### 10.4 Swapping to Production APIs

Each simulated tool in `server.py` includes a `note` field indicating it's simulated. To go to production:

1. **Stock prices**: Replace `stock_price_lookup` body with Alpha Vantage / Yahoo Finance API call
2. **SEC filings**: Replace `sec_filing_search` with EDGAR FULL-TEXT Search API (`efts.sec.gov`)
3. **Company data**: Replace `compare_financials` with a financial data provider (Polygon.io, IEX Cloud)

---

## 11. Running the System

### 11.1 Setup

```bash
# 1. Clone and install
cd finrag
pip install -e .

# 2. Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 3. Ingest documents
python ingest_cli.py
```

### 11.2 Interactive Mode

```bash
python main.py
# Opens a REPL where you can ask questions interactively
```

### 11.3 Single Query Mode

```bash
python main.py --query "What are Netflix's main risk factors?"
```

### 11.4 ADK Web UI

```bash
# Uses agent.py as the entry point (RAG agent only, no MCP)
adk web .
```

### 11.5 Re-Indexing Documents

```bash
python ingest_cli.py --force    # drops and rebuilds the vector store
python main.py --ingest         # ingest then start REPL
```

---

## 12. Performance Characteristics

| Operation | Estimated Latency | Notes |
|-----------|------------------|-------|
| Document ingestion (124 pages) | 30–60 seconds | Dominated by embedding API calls |
| Single query retrieval | 200–500 ms | Embedding + HNSW search |
| MCP tool call | 50–200 ms | In-process computation (simulated) |
| Full orchestrator response | 2–5 seconds | Includes LLM reasoning + tool calls |

---

## 13. Future Roadmap

1. **Streaming responses**: Leverage ADK's async streaming for real-time token output
2. **Session persistence**: Replace `InMemorySessionService` with a database-backed session store
3. **Multi-tenant support**: Namespace ChromaDB collections per user/organization
4. **Advanced chunking**: Implement section-aware chunking for 10-K filings
5. **Live market data**: Integrate Alpha Vantage or Polygon.io for real stock prices
6. **EDGAR integration**: Direct SEC EDGAR API integration for filing search
7. **Evaluation framework**: Automated RAG accuracy benchmarks with ground-truth Q&A pairs
8. **Observability**: Add OpenTelemetry tracing for agent routing decisions and tool latencies
