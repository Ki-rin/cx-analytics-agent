"""FastMCP server exposing Bitext dataset tools over the Model Context Protocol."""

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP

from agent.tools import (
    count_records,
    get_examples,
    get_intent_distribution,
    list_categories,
    list_intents,
    get_category_summary_data,
    recommend_next_query,
)

mcp = FastMCP("cx-analytics")


@mcp.tool()
def categories() -> str:
    """List all top-level categories in the Bitext customer service dataset."""
    return list_categories.invoke({})


@mcp.tool()
def intents(category: str) -> str:
    """List all intents within a given category (case-insensitive).

    Args:
        category: The category name, e.g. "BILLING" or "billing".

    Returns:
        A formatted string listing all intents in that category.
    """
    return list_intents.invoke({"category": category})


@mcp.tool()
def count(category: str = "", intent: str = "") -> str:
    """Count dataset records, optionally filtered by category and/or intent.

    Args:
        category: Optional category filter (case-insensitive).
        intent: Optional intent filter (case-insensitive).

    Returns:
        A string with the record count and a human-readable explanation.
    """
    return count_records.invoke({"category": category, "intent": intent})


@mcp.tool()
def examples(
    n: int = 5,
    category: str = "",
    intent: str = "",
    keyword: str = "",
) -> str:
    """Retrieve example records from the dataset.

    Filters are combined with AND logic.  Keyword is searched in the
    instruction (customer query) text.

    Args:
        n: Number of examples to return (1–50).
        category: Optional category filter (case-insensitive).
        intent: Optional intent filter (case-insensitive).
        keyword: Optional keyword to search in instruction text.

    Returns:
        A formatted string with the requested example records.
    """
    return get_examples.invoke(
        {"n": n, "category": category, "intent": intent, "keyword": keyword}
    )


@mcp.tool()
def intent_distribution(category: str) -> str:
    """Get the intent distribution (counts) for a given category.

    Args:
        category: The category name (case-insensitive).

    Returns:
        A formatted string listing each intent and its record count,
        sorted from most to least common.
    """
    return get_intent_distribution.invoke({"category": category})


@mcp.tool()
def category_summary(category: str, sample_size: int = 30) -> str:
    """Get a sample of customer instructions and agent responses for a category.

    This is useful for generating qualitative summaries of a category.

    Args:
        category: The category name (case-insensitive).
        sample_size: Number of sample rows to return (1–100).

    Returns:
        A formatted string with sample customer queries and agent responses.
    """
    return get_category_summary_data.invoke(
        {"category": category, "sample_size": sample_size}
    )


@mcp.tool()
def suggest_next_query(conversation_summary: str, user_interests: str = "") -> str:
    """Suggest the next query based on what has been explored in the session.

    This tool does NOT execute the suggested query — it only proposes one.

    Args:
        conversation_summary: Brief summary of what has been explored so far.
        user_interests: Known interests or focus areas (may be empty).

    Returns:
        A suggested query string with rationale.
    """
    return recommend_next_query.invoke(
        {
            "conversation_summary": conversation_summary,
            "user_interests": user_interests,
        }
    )


if __name__ == "__main__":
    mcp.run()
