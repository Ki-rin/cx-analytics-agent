#!/usr/bin/env python3
"""CLI entry point for the Customer Service Data Analyst Agent."""

import argparse
import sys

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import build_graph
from agent.memory import get_checkpointer, load_user_profile, save_user_profile


def _print_event(event: dict) -> None:
    """Print interesting events from the graph stream.

    Args:
        event: A LangGraph stream event dict (stream_mode='values').
    """
    messages = event.get("messages", [])
    if not messages:
        return

    last_msg = messages[-1]

    # Show tool calls the agent is making
    if isinstance(last_msg, AIMessage):
        if last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                tool_name = tc.get("name", "unknown_tool")
                tool_args = tc.get("args", {})
                print(f"\n[Tool Call] {tool_name}({tool_args})")
        elif last_msg.content and not last_msg.tool_calls:
            # This is the final assistant answer — will be printed separately
            pass

    # Show tool observations
    elif isinstance(last_msg, ToolMessage):
        content = last_msg.content
        # Truncate very long tool outputs for readability
        if len(content) > 800:
            content = content[:800] + "\n... (truncated)"
        print(f"[Observation] {content}")


def main() -> None:
    """Run the interactive CLI for the Customer Service Data Analyst Agent."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent — interactive CLI"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID for persistent memory (default: 'default')",
    )
    args = parser.parse_args()

    session_id = args.session

    print("Initialising agent (this may take a moment on first run) …")

    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer)

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    # Load existing profile and inject session_id so nodes can access it
    profile = load_user_profile(session_id)
    profile["_session_id"] = session_id

    print(f"\nAgent ready (session: {session_id}). Type 'quit' to exit.\n")
    if profile:
        non_internal = {k: v for k, v in profile.items() if not k.startswith("_")}
        if non_internal:
            print(f"Loaded user profile: {non_internal}\n")

    print("Example queries you can try:")
    print("  - List all categories")
    print("  - How many records are in the BILLING category?")
    print("  - Show 5 examples of the cancel_order intent")
    print("  - What is the intent distribution in SHIPPING?")
    print("  - Suggest a query\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()  # blank line before agent output

        final_answer = ""
        try:
            for event in graph.stream(
                {
                    "messages": [("user", user_input)],
                    "iteration_count": 0,
                    "query_type": "",
                    "user_profile": profile,
                    "pending_recommendation": None,
                    "last_filters": {},
                },
                config,
                stream_mode="values",
            ):
                _print_event(event)
                # Capture the latest assistant message
                msgs = event.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if isinstance(last, AIMessage) and last.content and not last.tool_calls:
                        final_answer = last.content

            # Reload profile after the turn (it may have been updated on disk)
            profile = load_user_profile(session_id)
            profile["_session_id"] = session_id

        except Exception as exc:
            print(f"[Error] An unexpected error occurred: {exc}")
            continue

        if final_answer:
            print(f"\nAssistant: {final_answer}\n")
        else:
            print()


if __name__ == "__main__":
    main()
