"""
test_missing_values.py - Unit tests for missing value analysis and imputation.
"""

import pandas as pd
import numpy as np
from src.missing_values import analyze_missing_values, impute_or_drop_missing


def test_analyze_missing_values_detection():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["A", "", "C", "D"],
        "complete": [10, 20, 30, 40]
    })
    res = analyze_missing_values(df, empty_strings_as_null=True)

    assert res["total_rows"] == 4
    assert res["total_columns"] == 3
    # num has 1 NaN, cat has 1 empty string (treated as null) -> total 2
    assert res["total_missing_cells"] == 2
    assert res["columns_with_missing_count"] == 2


def test_mean_imputation():
    df = pd.DataFrame({"val": [10.0, 20.0, np.nan, 30.0]})
    # Mean of [10, 20, 30] is 20.0
    cleaned_df, summary = impute_or_drop_missing(df, strategy="mean", target_columns=["val"])

    assert cleaned_df["val"].isna().sum() == 0
    assert cleaned_df["val"].iloc[2] == 20.0
    assert summary["resolved_missing_cells"] == 1


def test_median_imputation():
    df = pd.DataFrame({"val": [1.0, 2.0, np.nan, 100.0]})
    # Median of [1, 2, 100] is 2.0
    cleaned_df, summary = impute_or_drop_missing(df, strategy="median", target_columns=["val"])

    assert cleaned_df["val"].isna().sum() == 0
    assert cleaned_df["val"].iloc[2] == 2.0


def test_mode_imputation():
    df = pd.DataFrame({"dept": ["HR", "Sales", "Sales", None, "Engineering"]})
    cleaned_df, summary = impute_or_drop_missing(df, strategy="mode", target_columns=["dept"])

    assert cleaned_df["dept"].isna().sum() == 0
    assert cleaned_df["dept"].iloc[3] == "Sales"


def test_drop_rows_and_columns():
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [5, 6, 7, 8],
        "C": [np.nan, np.nan, np.nan, 9]
    })
    # Drop rows
    df_rows, s_rows = impute_or_drop_missing(df, strategy="drop_rows", target_columns=["A"])
    assert len(df_rows) == 3
    assert s_rows["rows_removed"] == 1

    # Drop column by threshold (C has 75% missing > 50% threshold)
    df_cols, s_cols = impute_or_drop_missing(df, strategy="drop_columns_by_thresh", drop_threshold_pct=50.0)
    assert "C" not in df_cols.columns
    assert "A" in df_cols.columns
