"""Streamlit chat UI for the Customer Service Data Analyst Agent."""

import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import build_graph
from agent.memory import get_checkpointer, load_user_profile

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CX Analytics Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Singleton graph / checkpointer (cached across re-runs)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_graph() -> Any:
    """Build and cache the LangGraph agent graph."""
    checkpointer = get_checkpointer()
    return build_graph(checkpointer)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise Streamlit session_state keys on first run."""
    if "session_id" not in st.session_state:
        st.session_id = "default"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of dicts: {role, content, steps}
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}


_init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    session_id_input = st.text_input(
        "Session ID",
        value=st.session_state.get("session_id", "default"),
        help="Conversations are persisted per session ID.",
    )

    if st.button("New Session"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_id
        st.session_state.chat_history = []
        st.session_state.user_profile = {}
        st.success(f"Started new session: {new_id}")
        st.rerun()
    else:
        st.session_state.session_id = session_id_input

    st.divider()
    st.markdown("**User Profile**")
    profile = load_user_profile(st.session_state.session_id)
    non_internal = {k: v for k, v in profile.items() if not k.startswith("_")}
    if non_internal:
        for key, val in non_internal.items():
            st.markdown(f"- **{key}**: {val}")
    else:
        st.caption("No profile facts recorded yet.")

    st.divider()
    st.markdown("**Example queries**")
    st.caption("- List all categories")
    st.caption("- Intent distribution for BILLING")
    st.caption("- Show 5 examples with keyword 'refund'")
    st.caption("- Summarise the SHIPPING category")
    st.caption("- Suggest a query")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("📊 Customer Service Data Analyst Agent")
st.caption(
    "Ask questions about the Bitext customer service dataset. "
    "The agent will use tools to explore categories, intents, and examples."
)

# Render existing chat history
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("steps"):
            with st.expander("🔍 Agent reasoning steps", expanded=False):
                for step in turn["steps"]:
                    st.markdown(step)

# ---------------------------------------------------------------------------
# Handle user input
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask something about the dataset …")

if user_input:
    # Display the user message immediately
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "steps": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent
    graph = _get_graph()
    session_id = st.session_state.session_id
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }

    profile = load_user_profile(session_id)
    profile["_session_id"] = session_id

    reasoning_steps: list[str] = []
    final_answer = ""

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.caption("Thinking …")

        try:
            for event in graph.stream(
                {
                    "messages": [("user", user_input)],
                    "iteration_count": 0,
                    "query_type": "",
                    "user_profile": profile,
                    "pending_recommendation": None,
                },
                config,
                stream_mode="values",
            ):
                msgs = event.get("messages", [])
                if not msgs:
                    continue

                last_msg = msgs[-1]

                # Capture tool calls
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_name = tc.get("name", "tool")
                        tool_args = tc.get("args", {})
                        step = f"**Tool call:** `{tool_name}({tool_args})`"
                        reasoning_steps.append(step)
                        status_placeholder.caption(f"Calling tool: {tool_name} …")

                # Capture tool observations
                elif isinstance(last_msg, ToolMessage):
                    content = last_msg.content
                    if len(content) > 600:
                        content = content[:600] + "\n… (truncated)"
                    step = f"**Observation:**\n```\n{content}\n```"
                    reasoning_steps.append(step)

                # Capture final AI response
                elif (
                    isinstance(last_msg, AIMessage)
                    and last_msg.content
                    and not last_msg.tool_calls
                ):
                    final_answer = last_msg.content

        except Exception as exc:
            final_answer = f"An error occurred: {exc}"

        status_placeholder.empty()

        if final_answer:
            st.markdown(final_answer)
        else:
            final_answer = "I was unable to generate a response. Please try again."
            st.markdown(final_answer)

        if reasoning_steps:
            with st.expander("🔍 Agent reasoning steps", expanded=False):
                for step in reasoning_steps:
                    st.markdown(step)

    # Persist this turn in chat history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": final_answer,
            "steps": reasoning_steps,
        }
    )
