"""
test_outliers.py - Unit tests for statistical outlier detection and winsorization.
"""

import pandas as pd
from src.outliers import detect_outliers_iqr, detect_outliers_zscore, remediate_outliers


def test_detect_outliers_iqr():
    # Regular values ~10-20, with extreme outlier 999
    data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 999]
    df = pd.DataFrame({"metric": data})

    res = detect_outliers_iqr(df, target_columns=["metric"], multiplier=1.5)
    col_info = res["columns"]["metric"]

    assert col_info["status"] == "analyzed"
    assert col_info["outlier_count"] == 1
    assert 11 in col_info["outlier_indices"]  # Index of 999 is 11


def test_remediate_outliers_cap():
    data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 999.0]
    df = pd.DataFrame({"metric": data})

    cleaned_df, summary = remediate_outliers(df, strategy="cap", target_columns=["metric"], multiplier=1.5)

    assert len(cleaned_df) == len(df)  # No rows removed
    assert summary["affected_cells"] == 1
    # Clamped to upper bound (which is around ~31)
    assert cleaned_df["metric"].max() < 100.0


def test_remediate_outliers_drop():
    data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 999.0]
    df = pd.DataFrame({"metric": data})

    cleaned_df, summary = remediate_outliers(df, strategy="drop", target_columns=["metric"], multiplier=1.5)

    assert len(cleaned_df) == 11
    assert summary["rows_removed"] == 1
    assert 999.0 not in cleaned_df["metric"].values


def test_outliers_constant_column():
    df = pd.DataFrame({"const": [5, 5, 5, 5, 5]})
    res = detect_outliers_iqr(df, target_columns=["const"])
    assert res["columns"]["const"]["status"] == "constant_or_zero_iqr"
    assert res["columns"]["const"]["outlier_count"] == 0
