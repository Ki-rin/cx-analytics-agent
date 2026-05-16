"""Agent state definition for the Customer Service Data Analyst Agent."""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the LangGraph ReAct agent.

    Attributes:
        messages: Conversation history with automatic message merging.
        query_type: Classification of the current query — one of
            "structured", "unstructured", or "out_of_scope".
        iteration_count: Number of ReAct loop iterations completed so far.
        user_profile: Dictionary of persisted facts about the current user.
        pending_recommendation: Optional query suggestion awaiting user
            confirmation (used by the Bonus B query recommender).
    """

    messages: Annotated[list, add_messages]
    query_type: str
    iteration_count: int
    user_profile: dict
    pending_recommendation: Optional[str]
