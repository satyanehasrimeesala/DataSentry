"""
test_duplicates.py - Unit tests for duplicate detection and deduplication.
"""

import pandas as pd
from src.duplicates import detect_duplicates, remove_duplicates


def test_detect_duplicates_exact():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"]
    })
    res = detect_duplicates(df)

    assert res["total_rows"] == 4
    assert res["duplicate_count"] == 1
    assert res["duplicate_pct"] == 25.0
    assert res["has_duplicates"] is True
    assert len(res["sample_duplicates"]) == 2  # Both rows in duplicate group


def test_remove_duplicates_keep_first():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"]
    })
    cleaned_df, summary = remove_duplicates(df, keep_strategy="first")

    assert len(cleaned_df) == 3
    assert summary["rows_removed"] == 1
    assert list(cleaned_df["id"]) == [1, 2, 3]


def test_remove_duplicates_keep_none():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"]
    })
    cleaned_df, summary = remove_duplicates(df, keep_strategy="none")

    # Both instances of 2 should be dropped
    assert len(cleaned_df) == 2
    assert summary["rows_removed"] == 2
    assert list(cleaned_df["id"]) == [1, 3]


def test_detect_duplicates_subset():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "detail": ["X", "Y1", "Y2", "Z"]
    })
    # Not exact duplicate across all columns
    res_all = detect_duplicates(df)
    assert res_all["duplicate_count"] == 0

    # But duplicate on 'id' subset
    res_sub = detect_duplicates(df, subset_columns=["id"])
    assert res_sub["duplicate_count"] == 1
