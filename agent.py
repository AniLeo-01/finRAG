"""
agent.py – Google ADK agent definition file.

This is the standard entry point that `adk run` and `adk web` look for.
It exports the root agent for the ADK runner.
"""

from agents.rag_agent import rag_agent

# For `adk web` / `adk run`, we export the rag_agent as a quick-start
# standalone agent. For the full orchestrator (with MCP), use main.py.
root_agent = rag_agent
