"""LangGraph ReAct agent graph for the Customer Service Data Analyst Agent."""

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.graph import CompiledGraph

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.router import route_query
from agent.memory import update_profile_from_conversation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEBIUS_API_KEY = os.environ.get(
    "NEBIUS_API_KEY",
    "v1.CmQKHHN0YXRpY2tleS1lMDB0YThnMzY1OXZ3YjFnN2ISIXNlcnZpY2VhY2NvdW50LWUwMHc2eWUzejI3OGtyNHI4ZDIMCMD53s4GEOXqsIEDOgwIv_z2mQcQwLSyogJAAloDZTAw.AAAAAAAAAAFJpUycOQhq_9Nymgej3qOlIyNjZRT-kkPIG3E1FLRwr0xL1xdk9oYiy8ekf7oZrvlPXCAYAijVYH9LcJLtEtoP",
)
LARGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct"
SMALL_MODEL = "google/gemma-2-9b-it-fast"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MAX_ITERATIONS = 12

AGENT_SYSTEM_PROMPT = """You are a Customer Service Data Analyst assistant specialised in \
the Bitext customer service dataset. You have access to tools that let you explore the dataset.

Dataset overview:
- The dataset contains customer service conversations labelled with categories and intents.
- Use list_categories to see all categories, list_intents to see intents within a category.
- Use count_records, get_examples, get_intent_distribution, and get_category_summary_data
  to analyse the data.
- Use recommend_next_query when the user asks for suggestions on what to explore next.

Guidelines:
- Always use the available tools to answer data questions — do not guess statistics.
- If you are unsure what categories or intents exist, call list_categories first.
- Be concise and clear in your final answers.
- If the user asks "what should I query next?" or "suggest a query", call recommend_next_query.
"""


def _get_agent_llm() -> ChatOpenAI:
    """Instantiate the large reasoning LLM with tools bound."""
    return ChatOpenAI(
        model=LARGE_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=BASE_URL,
        temperature=0,
    )


def _get_small_llm() -> ChatOpenAI:
    """Instantiate the small LLM for profile updates."""
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


def call_model(state: AgentState) -> AgentState:
    """Invoke the LLM (with tools bound) on the current message history."""
    llm = _get_agent_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = state["messages"]
    # Prepend system message if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)

    iteration_count = state.get("iteration_count", 0)

    # Guard: if we have hit the iteration limit, return a fallback
    if iteration_count >= MAX_ITERATIONS:
        fallback = AIMessage(
            content=(
                "I've reached the maximum number of reasoning steps. "
                "Here is my best answer based on what I've gathered so far. "
                "Please try rephrasing your question if you need more detail."
            )
        )
        return {
            **state,
            "messages": state["messages"] + [fallback],
            "iteration_count": iteration_count + 1,
        }

    response = llm_with_tools.invoke(messages)
    return {
        **state,
        "messages": state["messages"] + [response],
        "iteration_count": iteration_count + 1,
    }


def call_tools(state: AgentState) -> AgentState:
    """Execute any tool calls present in the last AI message."""
    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)
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
        # Profile update is best-effort — never crash the main flow
        return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def dispatch_query(state: AgentState) -> Literal["out_of_scope_node", "call_model"]:
    """Route to agent or refusal based on query_type."""
    if state.get("query_type") == "out_of_scope":
        return "out_of_scope_node"
    return "call_model"


def should_continue(state: AgentState) -> Literal["call_tools", "update_profile_node"]:
    """Decide whether to call tools or finish the ReAct loop."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # If the last message has tool calls, execute them
    if (
        last_message is not None
        and isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "call_tools"

    # Otherwise the agent has finished — move to profile update
    return "update_profile_node"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer: Any) -> CompiledGraph:
    """Build and compile the full LangGraph ReAct agent graph.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. SqliteSaver) for
            persistent memory across sessions.

    Returns:
        A compiled ``CompiledGraph`` ready to be invoked or streamed.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router_node", router_node)
    graph.add_node("out_of_scope_node", out_of_scope_node)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", call_tools)
    graph.add_node("update_profile_node", update_profile_node)

    # Edges
    graph.add_edge(START, "router_node")

    graph.add_conditional_edges(
        "router_node",
        dispatch_query,
        {
            "out_of_scope_node": "out_of_scope_node",
            "call_model": "call_model",
        },
    )

    graph.add_edge("out_of_scope_node", END)

    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tools": "call_tools",
            "update_profile_node": "update_profile_node",
        },
    )

    graph.add_edge("call_tools", "call_model")
    graph.add_edge("update_profile_node", END)

    return graph.compile(checkpointer=checkpointer)
