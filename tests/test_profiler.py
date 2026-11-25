"""
test_profiler.py - Comprehensive unit tests for dataset profiling module.
"""

import pandas as pd
import numpy as np
from src.profiler import profile_dataset, profile_column


def test_profile_dataset_basic():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["apple", "banana", "apple", "cherry", "banana"],
        "dt": pd.date_range("2026-01-01", periods=5, freq="D"),
        "bool_col": [True, False, True, True, False]
    })

    res = profile_dataset(df)

    assert res["total_rows"] == 5
    assert res["total_columns"] == 4
    assert res["total_cells"] == 20
    assert res["total_missing_cells"] == 0
    assert res["numeric_columns_count"] == 1
    assert "num" in res["numeric_columns"]
    assert "cat" in res["categorical_columns"]
    assert "dt" in res["datetime_columns"]
    assert "bool_col" in res["boolean_columns"]


def test_profile_column_with_missing_and_skew():
    s = pd.Series([10.0, 20.0, np.nan, 40.0, 50.0, 1000.0], name="test_col")
    res = profile_column(s)

    assert res["null_count"] == 1
    assert res["null_percentage"] > 0
    assert res["inferred_type"] == "numeric"
    assert "mean" in res
    assert "skewness" in res


def test_profile_empty_dataframe():
    df = pd.DataFrame()
    res = profile_dataset(df)

    assert res["total_rows"] == 0
    assert res["total_columns"] == 0
    assert res["total_missing_cells"] == 0
