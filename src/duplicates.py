"""
duplicates.py - Exact and subset duplicate detection and remediation module for DataSentry.

Features:
- Global exact row duplicate detection
- Key-column / subset duplicate detection
- Detailed metrics: counts, percentages, duplicate clusters
- Flexible deduplication strategies: keep 'first', keep 'last', or drop all
- Preview generation with diff tracking
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def detect_duplicates(df: pd.DataFrame, subset_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect duplicate rows in the DataFrame.

    Args:
        df: Input DataFrame
        subset_columns: Optional list of columns to consider for duplicates.
                        If None or empty, all columns are considered.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "total_rows": 0,
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "unique_rows": 0,
            "has_duplicates": False,
            "subset_used": subset_columns or [],
            "duplicate_indices": [],
            "sample_duplicates": pd.DataFrame()
        }

    subset = [c for c in subset_columns if c in df.columns] if subset_columns else None

    # Identify all rows that have duplicate counterparts (both first and subsequent)
    all_duplicate_mask = df.duplicated(subset=subset, keep=False)
    subsequent_duplicate_mask = df.duplicated(subset=subset, keep="first")

    duplicate_count = int(subsequent_duplicate_mask.sum())
    duplicate_pct = round((duplicate_count / total_rows * 100), 2)
    total_in_duplicate_groups = int(all_duplicate_mask.sum())

    # Get sample duplicate rows (first 10 records belonging to duplicate groups)
    sample_duplicates = df[all_duplicate_mask].head(20).copy() if total_in_duplicate_groups > 0 else pd.DataFrame()

    return {
        "total_rows": total_rows,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "total_in_duplicate_groups": total_in_duplicate_groups,
        "unique_rows": total_rows - duplicate_count,
        "has_duplicates": duplicate_count > 0,
        "subset_used": subset or list(df.columns),
        "duplicate_indices": list(df[subsequent_duplicate_mask].index),
        "sample_duplicates": sample_duplicates
    }


def remove_duplicates(
    df: pd.DataFrame,
    subset_columns: Optional[List[str]] = None,
    keep_strategy: str = "first"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove duplicate rows based on subset_columns and keep_strategy.

    Args:
        df: Input DataFrame
        subset_columns: Columns to consider (None for all)
        keep_strategy: 'first', 'last', or 'none' (drop all duplicates)

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and summary.
    """
    initial_rows = len(df)
    subset = [c for c in subset_columns if c in df.columns] if subset_columns else None

    if keep_strategy == "none":
        cleaned_df = df.drop_duplicates(subset=subset, keep=False).copy()
    else:
        cleaned_df = df.drop_duplicates(subset=subset, keep=keep_strategy).copy()

    final_rows = len(cleaned_df)
    rows_removed = initial_rows - final_rows

    summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": rows_removed,
        "removal_pct": round((rows_removed / initial_rows * 100), 2) if initial_rows > 0 else 0.0,
        "keep_strategy": keep_strategy,
        "subset_used": subset or "all_columns"
    }

    return cleaned_df, summary
