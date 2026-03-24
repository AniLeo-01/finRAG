"""
Orchestrator Agent – the top-level agent that delegates to sub-agents.

This is the main entry-point agent built with Google ADK. It uses ADK's
built-in agent routing to delegate:
  - Document/filing questions  →  rag_agent
  - Financial tools / utilities →  mcp_agent
"""

from __future__ import annotations

import sys
from pathlib import Path

from google.adk.agents import Agent

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GEMINI_MODEL
from agents.rag_agent import rag_agent


async def build_orchestrator() -> tuple[Agent, object]:
    """
    Build and return the orchestrator agent with both sub-agents wired in.

    Returns:
        A tuple of (orchestrator_agent, exit_stack) where exit_stack
        should be closed when done to clean up MCP connections.
    """
    # Import and create MCP agent (async because it connects to the server)
    from agents.mcp_agent import create_mcp_agent

    mcp_agent, mcp_toolset = await create_mcp_agent()

    orchestrator = Agent(
        name="finrag_orchestrator",
        model=GEMINI_MODEL,
        description="Financial analysis orchestrator that coordinates document retrieval and financial tools.",
        instruction=(
            "You are a senior financial analyst assistant. You coordinate two "
            "specialist sub-agents to help users with financial analysis:\n\n"
            "1. **rag_agent** – Use this for ANY question about the contents of "
            "   financial documents (10-K annual reports, Form 4 insider filings, "
            "   etc.). This agent searches through ingested SEC filings and "
            "   returns sourced answers.\n\n"
            "2. **mcp_agent** – Use this for financial TOOLS and UTILITIES:\n"
            "   - Looking up stock prices\n"
            "   - Calculating financial ratios\n"
            "   - Searching for SEC filings on EDGAR\n"
            "   - Comparing companies\n"
            "   - Math calculations, unit conversions, date/time queries\n\n"
            "**Routing rules:**\n"
            "- If the user asks about WHAT IS IN a document → route to rag_agent\n"
            "- If the user asks to CALCULATE, LOOK UP, or COMPARE → route to mcp_agent\n"
            "- If a question needs BOTH (e.g., 'What was Netflix's revenue in the "
            "  10-K and what's the current stock price?'), call rag_agent first "
            "  for the document data, then mcp_agent for the live data\n"
            "- Always provide a unified, coherent final answer to the user\n"
            "- Cite sources (document name, page number) when referencing filing data"
        ),
        sub_agents=[rag_agent, mcp_agent],
    )

    return orchestrator, mcp_toolset
