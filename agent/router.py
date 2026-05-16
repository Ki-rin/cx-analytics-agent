"""Router node: classifies the latest user message into a query type."""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState

NEBIUS_API_KEY = os.environ.get(
    "NEBIUS_API_KEY",
    "v1.CmQKHHN0YXRpY2tleS1lMDB0YThnMzY1OXZ3YjFnN2ISIXNlcnZpY2VhY2NvdW50LWUwMHc2eWUzejI3OGtyNHI4ZDIMCMD53s4GEOXqsIEDOgwIv_z2mQcQwLSyogJAAloDZTAw.AAAAAAAAAAFJpUycOQhq_9Nymgej3qOlIyNjZRT-kkPIG3E1FLRwr0xL1xdk9oYiy8ekf7oZrvlPXCAYAijVYH9LcJLtEtoP",
)
SMALL_MODEL = "google/gemma-2-9b-it-fast"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a Customer Service Data Analyst system.

Your job is to classify the user's message into EXACTLY ONE of these three categories:

1. structured   — The user wants to query, filter, count, or explore the Bitext customer
                  service dataset (e.g., "list categories", "how many records for billing?",
                  "show examples of cancellation intent", "what intents exist?",
                  "summarize the shipping category", "recommend a query").

2. unstructured — The user wants a general analysis, explanation, comparison, or narrative
                  answer that requires reasoning over the dataset (e.g., "why do customers
                  complain about shipping?", "compare billing vs account categories",
                  "what are the most common issues?").

3. out_of_scope — The message has nothing to do with the customer service dataset
                  (e.g., "write me a poem", "what is the capital of France?",
                  "tell me a joke", "help me with my Python homework").

Reply with ONLY one word — exactly one of: structured, unstructured, out_of_scope
Do NOT include any explanation, punctuation, or extra text."""


def _get_router_llm() -> ChatOpenAI:
    """Instantiate the small routing LLM."""
    return ChatOpenAI(
        model=SMALL_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0,
        max_tokens=10,
    )


def route_query(state: AgentState) -> AgentState:
    """Classify the latest user message and set query_type in state.

    Args:
        state: The current agent state containing the message history.

    Returns:
        Updated state with ``query_type`` set to one of "structured",
        "unstructured", or "out_of_scope".
    """
    messages = state.get("messages", [])
    if not messages:
        return {**state, "query_type": "structured"}

    # Find the last human message
    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (
            isinstance(msg, tuple) and msg[0] == "user"
        ):
            last_human = msg.content if hasattr(msg, "content") else msg[1]
            break

    if not last_human:
        return {**state, "query_type": "structured"}

    try:
        llm = _get_router_llm()
        response = llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=last_human),
            ]
        )
        raw = response.content.strip().lower()

        # Parse the label — be tolerant of minor formatting issues
        if "out_of_scope" in raw or "out of scope" in raw:
            query_type = "out_of_scope"
        elif "unstructured" in raw:
            query_type = "unstructured"
        elif "structured" in raw:
            query_type = "structured"
        else:
            # Default to structured so the agent at least tries to help
            query_type = "structured"
    except Exception:
        query_type = "structured"

    return {**state, "query_type": query_type}
