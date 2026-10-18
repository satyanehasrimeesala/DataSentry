"""
test_quality_score.py - Unit tests for explainable Data Quality Score.
"""

import pandas as pd
import numpy as np
from src.quality_score import calculate_quality_score


def test_perfect_quality_score():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Evan"],
        "score": [80.0, 85.0, 90.0, 88.0, 92.0]
    })
    res = calculate_quality_score(df)

    assert res["overall_score"] == 100.0
    assert res["grade"] == "Excellent"
    assert res["dimensions"]["completeness"] == 100.0
    assert res["dimensions"]["uniqueness"] == 100.0
    assert len(res["deductions"]) == 0


def test_degraded_quality_score():
    # Dataset with exact duplicates, missing values, and outliers
    df = pd.DataFrame({
        "id": [1, 2, 2, 2, 5, 6, 7, 8, 9, 10],
        "val": [10.0, 20.0, 20.0, 20.0, np.nan, np.nan, np.nan, 17.0, 18.0, 9999.0],
        "cat": ["A", "B", "B", "B", "C", "D", "E", "F", "G", "H"]
    })
    res = calculate_quality_score(df)

    assert res["overall_score"] < 90.0
    assert res["dimensions"]["completeness"] < 100.0
    assert res["dimensions"]["uniqueness"] < 100.0
    assert len(res["deductions"]) > 0


def test_empty_dataset_quality_score():
    df = pd.DataFrame()
    res = calculate_quality_score(df)

    assert res["overall_score"] == 0.0
    assert res["grade"] == "Critical"
