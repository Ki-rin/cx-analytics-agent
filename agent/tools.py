"""LangChain tools for querying the Bitext Customer Service dataset."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from agent.dataset import get_dataframe


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


class ListIntentsInput(BaseModel):
    """Input schema for list_intents."""

    category: str = Field(description="Category name (case-insensitive).")


class CountRecordsInput(BaseModel):
    """Input schema for count_records."""

    category: str = Field(default="", description="Optional category filter (case-insensitive).")
    intent: str = Field(default="", description="Optional intent filter (case-insensitive).")


class GetExamplesInput(BaseModel):
    """Input schema for get_examples."""

    n: int = Field(default=5, description="Number of examples to return (1–50).")
    category: str = Field(default="", description="Optional category filter (case-insensitive).")
    intent: str = Field(default="", description="Optional intent filter (case-insensitive).")
    keyword: str = Field(
        default="",
        description="Optional keyword to search for in the instruction text (case-insensitive).",
    )


class GetIntentDistributionInput(BaseModel):
    """Input schema for get_intent_distribution."""

    category: str = Field(description="Category name (case-insensitive).")


class GetCategorySummaryDataInput(BaseModel):
    """Input schema for get_category_summary_data."""

    category: str = Field(description="Category name (case-insensitive).")
    sample_size: int = Field(
        default=30,
        description="Number of sample rows to return for the LLM to summarise.",
    )


class RecommendNextQueryInput(BaseModel):
    """Input schema for recommend_next_query."""

    conversation_summary: str = Field(
        description="A brief summary of what the user has explored so far in this session."
    )
    user_interests: str = Field(
        description="Known interests or focus areas of the user (may be empty)."
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def list_categories() -> str:
    """Return a sorted list of all unique top-level categories available in the
    Bitext customer service dataset.  Use this tool first to discover what
    categories exist before filtering by category in other tools."""
    try:
        df = get_dataframe()
        categories = sorted(df["category"].unique().tolist())
        return "Available categories:\n" + "\n".join(f"  - {c}" for c in categories)
    except Exception as exc:
        return f"Error retrieving categories: {exc}"


@tool(args_schema=ListIntentsInput)
def list_intents(category: str) -> str:
    """Return a sorted list of all unique intents within the specified category.
    The category match is case-insensitive.  Use list_categories first if you
    are unsure which categories exist."""
    try:
        df = get_dataframe()
        mask = df["category"].str.upper() == category.upper().strip()
        filtered = df[mask]
        if filtered.empty:
            return (
                f"No records found for category '{category}'. "
                "Use list_categories to see available categories."
            )
        intents = sorted(filtered["intent"].unique().tolist())
        return (
            f"Intents in category '{category.upper()}':\n"
            + "\n".join(f"  - {i}" for i in intents)
        )
    except Exception as exc:
        return f"Error retrieving intents: {exc}"


@tool(args_schema=CountRecordsInput)
def count_records(category: str = "", intent: str = "") -> str:
    """Count the number of records in the dataset, optionally filtered by
    category and/or intent.  Filters are combined with AND logic.
    Both category and intent are matched case-insensitively.
    Returns the count along with a human-readable explanation."""
    try:
        df = get_dataframe()
        mask = pd.Series([True] * len(df), index=df.index)

        if category.strip():
            mask &= df["category"].str.upper() == category.upper().strip()
        if intent.strip():
            mask &= df["intent"].str.lower() == intent.lower().strip()

        count = int(mask.sum())
        parts: list[str] = []
        if category.strip():
            parts.append(f"category='{category.upper()}'")
        if intent.strip():
            parts.append(f"intent='{intent.lower()}'")

        if parts:
            description = " AND ".join(parts)
            return f"Records matching {description}: {count}"
        return f"Total records in the dataset: {count}"
    except Exception as exc:
        return f"Error counting records: {exc}"


# Need pandas for the mask above — import at top level
import pandas as pd  # noqa: E402  (placed after tool body for clarity in generation)


@tool(args_schema=GetExamplesInput)
def get_examples(
    n: int = 5,
    category: str = "",
    intent: str = "",
    keyword: str = "",
) -> str:
    """Return up to n example rows from the dataset as a formatted string.
    Optional filters (category, intent, keyword) are combined with AND logic.
    The keyword is searched case-insensitively within the instruction text.
    Use this to inspect real customer queries and agent responses."""
    try:
        df = get_dataframe()
        mask = pd.Series([True] * len(df), index=df.index)

        if category.strip():
            mask &= df["category"].str.upper() == category.upper().strip()
        if intent.strip():
            mask &= df["intent"].str.lower() == intent.lower().strip()
        if keyword.strip():
            mask &= df["instruction"].str.contains(keyword.strip(), case=False, na=False)

        filtered = df[mask]
        if filtered.empty:
            return "No records match the specified filters."

        n = max(1, min(n, 50))
        sample = filtered.head(n)

        lines: list[str] = [f"Showing {len(sample)} example(s):"]
        for idx, (_, row) in enumerate(sample.iterrows(), start=1):
            lines.append(
                f"\n--- Example {idx} ---\n"
                f"Category : {row['category']}\n"
                f"Intent   : {row['intent']}\n"
                f"Instruction: {row['instruction']}\n"
                f"Response : {row['response']}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error retrieving examples: {exc}"


@tool(args_schema=GetIntentDistributionInput)
def get_intent_distribution(category: str) -> str:
    """Return the count of records for each intent within the specified category,
    sorted from most to least common.  Use this to understand which intents are
    most prevalent in a given category."""
    try:
        df = get_dataframe()
        mask = df["category"].str.upper() == category.upper().strip()
        filtered = df[mask]

        if filtered.empty:
            return (
                f"No records found for category '{category}'. "
                "Use list_categories to see available categories."
            )

        distribution = (
            filtered["intent"]
            .value_counts()
            .reset_index()
        )
        distribution.columns = ["intent", "count"]

        lines = [f"Intent distribution for category '{category.upper()}':"]
        for _, row in distribution.iterrows():
            lines.append(f"  {row['intent']}: {row['count']}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error computing intent distribution: {exc}"


@tool(args_schema=GetCategorySummaryDataInput)
def get_category_summary_data(category: str, sample_size: int = 30) -> str:
    """Return a sample of customer instructions and agent responses for the
    specified category so that an LLM can generate a qualitative summary of
    what kinds of customer issues and answers appear in that category.
    Use this when the user asks for a narrative overview of a category."""
    try:
        df = get_dataframe()
        mask = df["category"].str.upper() == category.upper().strip()
        filtered = df[mask]

        if filtered.empty:
            return (
                f"No records found for category '{category}'. "
                "Use list_categories to see available categories."
            )

        sample_size = max(1, min(sample_size, 100))
        sample = filtered.sample(n=min(sample_size, len(filtered)), random_state=42)

        lines = [
            f"Sample of {len(sample)} records from category '{category.upper()}' "
            f"(out of {len(filtered)} total):"
        ]
        for idx, (_, row) in enumerate(sample.iterrows(), start=1):
            lines.append(
                f"\n[{idx}] Intent: {row['intent']}\n"
                f"  Customer: {row['instruction']}\n"
                f"  Agent   : {row['response']}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error retrieving category summary data: {exc}"


@tool(args_schema=RecommendNextQueryInput)
def recommend_next_query(conversation_summary: str, user_interests: str) -> str:
    """Generate a suggested next query for the user based on what they have
    explored so far in the current session and their known interests.
    This tool does NOT execute the query — it only proposes one.
    The agent should present the suggestion to the user and ask for
    confirmation before proceeding."""
    try:
        df = get_dataframe()
        all_categories = sorted(df["category"].unique().tolist())

        # Build a heuristic suggestion based on conversation content
        summary_lower = conversation_summary.lower()
        interests_lower = user_interests.lower()

        # Find any category mentioned in the conversation
        mentioned_categories = [
            c for c in all_categories
            if c.lower() in summary_lower or c.lower() in interests_lower
        ]

        if mentioned_categories:
            cat = mentioned_categories[0]
            intents = sorted(
                df[df["category"].str.upper() == cat]["intent"].unique().tolist()
            )
            if intents:
                suggestion_intent = intents[0]
                recommendation = (
                    f"Based on your interest in '{cat}', I suggest exploring the "
                    f"'{suggestion_intent}' intent. You could ask:\n\n"
                    f"  \"Show me 5 examples of {suggestion_intent} in the {cat} category\"\n\n"
                    f"or\n\n"
                    f"  \"What is the intent distribution in {cat}?\"\n\n"
                    f"Would you like me to run one of these queries?"
                )
                return recommendation

        # Generic fallback suggestion
        top_categories = all_categories[:3]
        recommendation = (
            "Based on your session so far, here are some queries you might find interesting:\n\n"
            + "\n".join(
                f"  - \"Show me the intent distribution for {c}\"" for c in top_categories
            )
            + "\n\nWould you like me to run any of these?"
        )
        return recommendation
    except Exception as exc:
        return f"Error generating recommendation: {exc}"


# Convenience list for importing all tools at once
ALL_TOOLS = [
    list_categories,
    list_intents,
    count_records,
    get_examples,
    get_intent_distribution,
    get_category_summary_data,
    recommend_next_query,
]
