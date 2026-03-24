"""
MCP Sub-Agent – connects to the FastMCP server and exposes its tools.

Uses Google ADK's McpToolset to bridge the FastMCP server's tools
into the ADK agent ecosystem.
"""

from __future__ import annotations

import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GEMINI_MODEL


# ── Path to the FastMCP server script ────────────────────────────────────

MCP_SERVER_SCRIPT = str(
    Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
)


# ── Factory function (async, because McpToolset needs to connect) ────────

async def create_mcp_agent() -> tuple[Agent, McpToolset]:
    """
    Create and return the MCP sub-agent with tools loaded from the
    FastMCP server via stdio transport.
    """
    mcp_toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,   # use the same Python interpreter
                args=[MCP_SERVER_SCRIPT],
            ),
        ),
    )

    tools = await mcp_toolset.get_tools()

    mcp_agent = Agent(
        name="mcp_agent",
        model=GEMINI_MODEL,
        description=(
            "Provides financial utility tools (stock price lookup, financial "
            "ratio calculator, SEC filing search, company comparisons) and "
            "generic helpers (calculator, date/time, unit converter, JSON "
            "formatter) via an MCP server."
        ),
        instruction=(
            "You are a financial tools assistant. You have access to the "
            "following categories of tools:\n\n"
            "**Financial Tools:**\n"
            "- `stock_price_lookup` – get the latest stock price for a ticker\n"
            "- `calculate_financial_ratios` – compute profitability, leverage, "
            "  and liquidity ratios\n"
            "- `sec_filing_search` – look up SEC filings by company and type\n"
            "- `compare_financials` – compare metrics between two companies\n\n"
            "**Utility Tools:**\n"
            "- `calculator` – evaluate math expressions\n"
            "- `current_date_time` – get the current date and time\n"
            "- `unit_converter` – convert between units (including millions/billions)\n"
            "- `json_formatter` – pretty-print JSON strings\n\n"
            "Use the appropriate tool based on the user's request. Always show "
            "your work and explain the results clearly."
        ),
        tools=tools,
    )

    return mcp_agent, mcp_toolset
