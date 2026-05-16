"""Singleton loader for the Bitext Customer Service dataset."""

import functools
import pandas as pd
from datasets import load_dataset


@functools.lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """Load and cache the Bitext customer service dataset.

    Returns:
        A pandas DataFrame with columns: flags, instruction, category,
        intent, response.  Category values are upper-cased and stripped;
        intent values are lower-cased and stripped for consistent matching.
    """
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    df = ds.to_pandas()

    # Normalize for consistent, case-insensitive lookups
    df["category"] = df["category"].str.upper().str.strip()
    df["intent"] = df["intent"].str.lower().str.strip()

    return df
