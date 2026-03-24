"""
main.py – Full orchestrator runner with both sub-agents.

Usage:
    python main.py                          # interactive REPL
    python main.py --ingest                 # ingest documents then start REPL
    python main.py --query "What is Netflix's revenue?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config.settings import GEMINI_MODEL


async def run_interactive(orchestrator, runner, session):
    """Run an interactive REPL loop."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   FinRAG – Financial Document Analysis Agent               ║")
    print("║   Type your question, or 'quit' to exit.                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        )

        print("\nAssistant: ", end="", flush=True)
        async for event in runner.run_async(
            session_id=session.id,
            user_id="user",
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
        print("\n")


async def run_single_query(orchestrator, runner, session, query: str):
    """Run a single query and print the response."""
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)
    print()


async def main():
    parser = argparse.ArgumentParser(description="FinRAG – Financial Document Analysis Agent")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents before starting")
    parser.add_argument("--query", "-q", type=str, help="Run a single query instead of REPL")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-index documents")
    args = parser.parse_args()

    # Optionally ingest documents first
    if args.ingest or args.force_reindex:
        print("Ingesting documents …")
        from rag.ingest import ingest_documents

        count = ingest_documents(force=args.force_reindex)
        print(f"Done — {count} chunks indexed.\n")

    # Build the orchestrator
    from agents.orchestrator import build_orchestrator

    orchestrator, mcp_toolset = await build_orchestrator()

    # Set up ADK session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="finrag",
        user_id="user",
    )

    runner = Runner(
        agent=orchestrator,
        app_name="finrag",
        session_service=session_service,
    )

    try:
        if args.query:
            await run_single_query(orchestrator, runner, session, args.query)
        else:
            await run_interactive(orchestrator, runner, session)
    finally:
        await mcp_toolset.close()


if __name__ == "__main__":
    asyncio.run(main())
