"""LangGraph ReAct agent graph for the Customer Service Data Analyst Agent."""

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.router import route_query
from agent.memory import update_profile_from_conversation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
LARGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SMALL_MODEL = "Meta/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MAX_ITERATIONS = 12

STRUCTURED_SYSTEM_PROMPT = """You are a Customer Service Data Analyst assistant specialised in
the Bitext customer service dataset. You have access to tools that let you explore the dataset.

Dataset overview:
- The dataset contains customer service conversations labelled with categories and intents.
- Use list_categories to see all categories, list_intents to see intents within a category.
- Use count_records, get_examples, get_intent_distribution to query the data.
- Use recommend_next_query when the user asks for suggestions on what to explore next.

Guidelines:
- Always use the available tools to answer data questions — do not guess statistics.
- If you are unsure what categories or intents exist, call list_categories first.
- When the user says "show more", "show N more", "another batch", or similar, reuse
  the same category/intent/keyword filters from the previous get_examples call.
- Be concise and clear in your final answers.
"""

UNSTRUCTURED_SYSTEM_PROMPT = """You are a Customer Service Data Analyst assistant specialised in
the Bitext customer service dataset. The user has asked an open-ended question that requires
qualitative analysis.

Your task:
1. Use get_category_summary_data to retrieve a sample of real customer queries and agent
   responses for the relevant category or intent.
2. Synthesise a clear, narrative summary based on the retrieved data — do NOT invent facts.
3. If you need to identify which categories exist first, call list_categories.

Focus on patterns, tone, and themes rather than raw numbers.
"""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _get_agent_llm() -> ChatOpenAI:
    """Return the large reasoning LLM."""
    return ChatOpenAI(
        model=LARGE_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0,
    )


def _get_small_llm() -> ChatOpenAI:
    """Return the small LLM used for profile updates."""
    return ChatOpenAI(
        model=SMALL_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0,
        max_tokens=256,
    )


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def router_node(state: AgentState) -> AgentState:
    """Classify the incoming user query and set query_type in state."""
    return route_query(state)


def out_of_scope_node(state: AgentState) -> AgentState:
    """Return a polite refusal for out-of-scope queries."""
    refusal = AIMessage(
        content=(
            "I'm sorry, but I'm only able to help with questions about the "
            "Bitext customer service dataset — such as exploring categories, "
            "intents, and example conversations. Could you ask something "
            "related to the dataset?"
        )
    )
    return {**state, "messages": state["messages"] + [refusal]}


def _call_llm_with_tools(state: AgentState, system_prompt: str) -> AgentState:
    """Shared ReAct step: invoke LLM with tools; enforce max-iteration guard."""
    iteration_count = state.get("iteration_count", 0)

    if iteration_count >= MAX_ITERATIONS:
        fallback = AIMessage(
            content=(
                "I couldn't complete the analysis within the allowed reasoning steps. "
                "Please try rephrasing your question or breaking it into smaller parts."
            )
        )
        return {
            **state,
            "messages": state["messages"] + [fallback],
            "iteration_count": iteration_count + 1,
        }

    llm = _get_agent_llm().bind_tools(ALL_TOOLS)

    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    profile = state.get("user_profile", {})
    if profile:
        profile_lines = "\n".join(
            f"  {k}: {v}" for k, v in profile.items() if not k.startswith("_")
        )
        if profile_lines:
            profile_ctx = SystemMessage(content=f"Known user profile:\n{profile_lines}")
            messages = [messages[0], profile_ctx] + messages[1:]

    response = llm.invoke(messages)
    return {
        **state,
        "messages": state["messages"] + [response],
        "iteration_count": iteration_count + 1,
    }


def call_model(state: AgentState) -> AgentState:
    """ReAct step for structured queries: invoke LLM with dataset tools."""
    return _call_llm_with_tools(state, STRUCTURED_SYSTEM_PROMPT)


def summarize_node(state: AgentState) -> AgentState:
    """ReAct step for unstructured queries: invoke LLM with summarization focus."""
    return _call_llm_with_tools(state, UNSTRUCTURED_SYSTEM_PROMPT)


def call_tools(state: AgentState) -> AgentState:
    """Execute tool calls from the last AI message and track last_filters."""
    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    # Track filters from the last get_examples call for "show more" continuations
    messages = state["messages"]
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    if last_ai and last_ai.tool_calls:
        for tc in last_ai.tool_calls:
            if tc.get("name") == "get_examples":
                result["last_filters"] = tc.get("args", {})
                break

    return result


def update_profile_node(state: AgentState) -> AgentState:
    """Update the persisted user profile after the agent has replied."""
    session_id = state.get("user_profile", {}).get("_session_id", "default")
    try:
        small_llm = _get_small_llm()
        updated_profile = update_profile_from_conversation(
            session_id=session_id,
            messages=state["messages"],
            llm=small_llm,
        )
        updated_profile["_session_id"] = session_id
        return {**state, "user_profile": updated_profile}
    except Exception:
        return state


# ---------------------------------------------------------------------------
# Routing / edge conditions
# ---------------------------------------------------------------------------


def dispatch_query(
    state: AgentState,
) -> Literal["out_of_scope_node", "call_model", "summarize_node"]:
    """Route to the correct branch based on query_type."""
    query_type = state.get("query_type", "structured")
    if query_type == "out_of_scope":
        return "out_of_scope_node"
    if query_type == "unstructured":
        return "summarize_node"
    return "call_model"


def should_continue_structured(
    state: AgentState,
) -> Literal["call_tools", "update_profile_node"]:
    """Decide whether to keep looping (structured path)."""
    last_message = state["messages"][-1] if state["messages"] else None
    if (
        last_message is not None
        and isinstance(last_message, AIMessage)
        and last_message.tool_calls
        and state.get("iteration_count", 0) < MAX_ITERATIONS
    ):
        return "call_tools"
    return "update_profile_node"


def should_continue_unstructured(
    state: AgentState,
) -> Literal["call_tools", "update_profile_node"]:
    """Decide whether to keep looping (unstructured / summarization path)."""
    last_message = state["messages"][-1] if state["messages"] else None
    if (
        last_message is not None
        and isinstance(last_message, AIMessage)
        and last_message.tool_calls
        and state.get("iteration_count", 0) < MAX_ITERATIONS
    ):
        return "call_tools"
    return "update_profile_node"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer: Any) -> StateGraph:
    """Build and compile the full LangGraph ReAct agent graph.

    The graph has three distinct paths after the router:
    - structured   → call_model (ReAct loop) → update_profile_node
    - unstructured → summarize_node (ReAct loop, summarization focus) → update_profile_node
    - out_of_scope → out_of_scope_node (polite refusal) → END

    Args:
        checkpointer: A LangGraph checkpointer (e.g. SqliteSaver) for
            persistent memory across sessions.

    Returns:
        A compiled ``CompiledGraph`` ready to be invoked or streamed.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router_node", router_node)
    graph.add_node("out_of_scope_node", out_of_scope_node)
    graph.add_node("call_model", call_model)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("call_tools", call_tools)
    graph.add_node("update_profile_node", update_profile_node)

    graph.add_edge(START, "router_node")

    graph.add_conditional_edges(
        "router_node",
        dispatch_query,
        {
            "out_of_scope_node": "out_of_scope_node",
            "call_model": "call_model",
            "summarize_node": "summarize_node",
        },
    )

    graph.add_edge("out_of_scope_node", END)

    # Structured ReAct loop
    graph.add_conditional_edges(
        "call_model",
        should_continue_structured,
        {
            "call_tools": "call_tools",
            "update_profile_node": "update_profile_node",
        },
    )

    # Unstructured / summarization loop (same tool node, different LLM prompt)
    graph.add_conditional_edges(
        "summarize_node",
        should_continue_unstructured,
        {
            "call_tools": "call_tools",
            "update_profile_node": "update_profile_node",
        },
    )

    # Tools can return to either agent node depending on where we came from.
    # Because both share the same call_tools node we route back based on query_type.
    graph.add_conditional_edges(
        "call_tools",
        lambda s: "summarize_node" if s.get("query_type") == "unstructured" else "call_model",
        {
            "call_model": "call_model",
            "summarize_node": "summarize_node",
        },
    )

    graph.add_edge("update_profile_node", END)

    return graph.compile(checkpointer=checkpointer)
