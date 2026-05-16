"""Persistent memory utilities for the Customer Service Data Analyst Agent."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
SMALL_MODEL = "Meta/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

PROFILES_DIR = Path("profiles")

PROFILE_EXTRACTION_PROMPT = """You are a user-profiling assistant.  Given the recent conversation
messages below, extract any new facts about the user's interests, focus areas, or preferences
related to the customer service dataset.  Return ONLY a JSON object where each key is a short
fact label and the value is the fact.  If there is nothing new to extract, return {{}}.

Examples of facts to extract:
  - "preferred_category": "BILLING"
  - "interested_in": "cancellation intents"
  - "analysis_goal": "understand shipping complaints"

Conversation:
{conversation}

Reply with ONLY valid JSON. No explanation."""


def get_checkpointer(db_path: str = "memory.db") -> SqliteSaver:
    """Create and return a SqliteSaver for persistent LangGraph checkpointing.

    Uses a direct sqlite3 connection rather than the context-manager form of
    ``from_conn_string`` so that the saver can be used outside a ``with`` block.
    ``check_same_thread=False`` is required because LangGraph may access the
    connection from multiple threads.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A ``SqliteSaver`` instance connected to the specified database.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def load_user_profile(session_id: str) -> dict:
    """Load the user profile for the given session from disk.

    Args:
        session_id: Unique identifier for the user session.

    Returns:
        A dictionary of profile facts, or an empty dict if none exists.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{session_id}.json"
    if not profile_path.exists():
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_user_profile(session_id: str, profile: dict) -> None:
    """Persist the user profile to disk.

    Args:
        session_id: Unique identifier for the user session.
        profile: Dictionary of profile facts to save.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{session_id}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def update_profile_from_conversation(
    session_id: str,
    messages: list[Any],
    llm: ChatOpenAI,
) -> dict:
    """Extract updated profile facts from recent messages using an LLM.

    The function calls the LLM to identify new facts about the user,
    merges them with the existing profile, and persists the result.

    Args:
        session_id: Unique identifier for the user session.
        messages: Recent message objects from the agent state.
        llm: A ChatOpenAI instance to use for extraction.

    Returns:
        The updated profile dictionary.
    """
    existing_profile = load_user_profile(session_id)

    # Format recent messages for the LLM (last 10 at most)
    recent = messages[-10:]
    conversation_lines: list[str] = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            conversation_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_lines.append(f"Assistant: {msg.content}")
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            conversation_lines.append(f"{role.capitalize()}: {content}")

    if not conversation_lines:
        return existing_profile

    conversation_text = "\n".join(conversation_lines)

    try:
        extractor_llm = ChatOpenAI(
            model=SMALL_MODEL,
            openai_api_key=NEBIUS_API_KEY,
            openai_api_base=BASE_URL,
            temperature=0,
            max_tokens=256,
        )
        response = extractor_llm.invoke(
            [
                SystemMessage(
                    content=PROFILE_EXTRACTION_PROMPT.format(
                        conversation=conversation_text
                    )
                )
            ]
        )
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        new_facts: dict = json.loads(raw)
        if isinstance(new_facts, dict):
            existing_profile.update(new_facts)
    except Exception:
        # Silently ignore extraction errors — profiling is best-effort
        pass

    save_user_profile(session_id, existing_profile)
    return existing_profile
