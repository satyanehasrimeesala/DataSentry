"""
test_consistency.py - Unit tests for consistency checks, text normalization, and custom rules.
"""

import pandas as pd
from src.consistency import (
    analyze_consistency,
    normalize_text,
    convert_column_type,
    validate_custom_rules
)


def test_analyze_consistency_whitespace_and_casing():
    df = pd.DataFrame({
        "dept": ["  Sales ", "sales", "SALES", "Engineering"]
    })
    res = analyze_consistency(df)

    assert "dept" in res["whitespace_issues"]
    assert "dept" in res["casing_inconsistencies"]


def test_normalize_text():
    df = pd.DataFrame({
        "dept": ["  Sales ", "marketing  ", "HUMAN   RESOURCES "]
    })
    cleaned_df, summary = normalize_text(
        df,
        target_columns=["dept"],
        strip_whitespace=True,
        case_style="title",
        collapse_spaces=True
    )

    assert cleaned_df["dept"].iloc[0] == "Sales"
    assert cleaned_df["dept"].iloc[1] == "Marketing"
    assert cleaned_df["dept"].iloc[2] == "Human Resources"
    assert summary["affected_cells"] == 3


def test_convert_column_type():
    df = pd.DataFrame({
        "age_str": ["25", "30", "invalid", "45"]
    })
    cleaned_df, summary = convert_column_type(df, column="age_str", target_type="numeric")

    assert pd.api.types.is_numeric_dtype(cleaned_df["age_str"])
    assert summary["invalid_coerced_to_nan"] == 1
    assert pd.isna(cleaned_df["age_str"].iloc[2])


def test_validate_custom_rules():
    df = pd.DataFrame({
        "age": [25, 150, -5, 30],
        "dept": ["Sales", "InvalidDept", "Engineering", "Sales"]
    })
    rules = [
        {"column": "age", "type": "range", "min": 0, "max": 120},
        {"column": "dept", "type": "allowed_values", "allowed": ["Sales", "Engineering"]}
    ]
    res = validate_custom_rules(df, rules)

    assert res["rules_evaluated"] == 2
    # age has 2 violations (150 and -5), dept has 1 ("InvalidDept") -> 3 total
    assert res["total_violations"] == 3
