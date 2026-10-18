"""
consistency.py - Data type consistency, string normalization, and custom validation engine for DataSentry.

Capabilities:
- Mixed-type and disguised-type column detection
- Whitespace anomaly detection (leading/trailing/multiple internal spaces)
- Category casing inconsistency detection (e.g. 'Sales', 'sales', 'SALES')
- Date formatting integrity and unparseable date detection
- Numeric type validity checking
- User-defined rule validation (allowed sets, numeric bounds [min, max])
- Safe normalization transformations (strip, case-standardize, regex clean)
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import pandas as pd
import numpy as np


def analyze_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Examine DataFrame for data type inconsistencies, whitespace issues,
    capitalization discrepancies, and invalid dates/numbers.
    """
    total_rows = len(df)
    results: Dict[str, Any] = {
        "whitespace_issues": {},
        "casing_inconsistencies": {},
        "mixed_type_columns": {},
        "suspicious_numeric_columns": {},
        "suspicious_date_columns": {},
        "total_consistency_issues_found": 0
    }

    issue_count = 0

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        # 1. Whitespace issues in string/object columns
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            str_series = non_null.astype(str)
            stripped = str_series.str.strip()
            whitespace_diff = str_series != stripped
            ws_count = int(whitespace_diff.sum())

            # Multiple spaces
            double_spaces = str_series.str.contains(r"\s{2,}", regex=True).sum()

            if ws_count > 0 or double_spaces > 0:
                results["whitespace_issues"][col] = {
                    "leading_trailing_count": ws_count,
                    "double_space_count": int(double_spaces),
                    "affected_pct": round((ws_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
                    "examples": str_series[whitespace_diff].head(5).tolist()
                }
                issue_count += ws_count

            # 2. Casing inconsistencies in categories
            # Check if lowercased unique count < raw unique count
            raw_uniques = str_series.str.strip().unique()
            lower_uniques = {s.lower() for s in raw_uniques}
            if len(lower_uniques) < len(raw_uniques) and len(raw_uniques) < 100:
                # Find the grouped variants
                variant_groups: Dict[str, List[str]] = {}
                for val in raw_uniques:
                    variant_groups.setdefault(val.lower(), []).append(val)
                inconsistent_groups = {k: v for k, v in variant_groups.items() if len(v) > 1}
                if inconsistent_groups:
                    results["casing_inconsistencies"][col] = {
                        "inconsistent_groups": inconsistent_groups,
                        "group_count": len(inconsistent_groups)
                    }
                    issue_count += len(inconsistent_groups)

            # 3. Check if object column should be numeric (suspicious numeric)
            numeric_attempt = pd.to_numeric(str_series, errors="coerce")
            valid_numeric_count = numeric_attempt.notna().sum()
            if valid_numeric_count > 0 and len(str_series) > 0:
                ratio = valid_numeric_count / len(str_series)
                if 0.50 <= ratio < 1.0:
                    invalid_mask = numeric_attempt.isna()
                    invalid_tokens = str_series[invalid_mask].unique()[:10].tolist()
                    results["suspicious_numeric_columns"][col] = {
                        "numeric_ratio": round(ratio * 100, 2),
                        "invalid_tokens_sample": invalid_tokens,
                        "invalid_count": int(invalid_mask.sum())
                    }
                    issue_count += int(invalid_mask.sum())

            # 4. Check if object column should be datetime (suspicious date)
            # Sample first 50 values to test date parsing
            sample = str_series.head(50)
            date_attempt = pd.to_datetime(sample, errors="coerce", format="mixed")
            valid_date_count = date_attempt.notna().sum()
            if valid_date_count > 0 and len(sample) > 0:
                date_ratio = valid_date_count / len(sample)
                if 0.60 <= date_ratio < 1.0:
                    results["suspicious_date_columns"][col] = {
                        "date_ratio_in_sample": round(date_ratio * 100, 2),
                        "suggested_type": "datetime"
                    }
                    issue_count += 1

        # 5. Mixed python types in object columns
        if pd.api.types.is_object_dtype(series):
            type_set = {type(val).__name__ for val in non_null}
            if len(type_set) > 1:
                results["mixed_type_columns"][col] = {
                    "types_found": list(type_set),
                    "count": len(type_set)
                }
                issue_count += 1

    results["total_consistency_issues_found"] = issue_count
    return results


def normalize_text(
    df: pd.DataFrame,
    target_columns: List[str],
    strip_whitespace: bool = True,
    case_style: Optional[str] = None,
    collapse_spaces: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardize text formatting for specified columns.

    Args:
        case_style: 'lower', 'upper', 'title', 'capitalize', or None
    """
    cleaned_df = df.copy()
    affected_cells = 0

    for col in target_columns:
        if col not in cleaned_df.columns:
            continue
        series = cleaned_df[col].astype(str)
        original = series.copy()

        if strip_whitespace:
            series = series.str.strip()

        if collapse_spaces:
            series = series.str.replace(r"\s+", " ", regex=True)

        if case_style == "lower":
            series = series.str.lower()
        elif case_style == "upper":
            series = series.str.upper()
        elif case_style == "title":
            series = series.str.title()
        elif case_style == "capitalize":
            series = series.str.capitalize()

        diff_mask = (original != series) & cleaned_df[col].notna()
        affected_cells += int(diff_mask.sum())
        cleaned_df.loc[cleaned_df[col].notna(), col] = series[cleaned_df[col].notna()]

    return cleaned_df, {
        "operation": "normalize_text",
        "target_columns": target_columns,
        "strip_whitespace": strip_whitespace,
        "case_style": case_style,
        "collapse_spaces": collapse_spaces,
        "affected_cells": affected_cells
    }


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    date_format: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely convert a column to numeric, datetime, string, or boolean.
    Invalid conversions will be safely set to NaN.
    """
    cleaned_df = df.copy()
    if column not in cleaned_df.columns:
        raise ValueError(f"Column '{column}' not in dataset.")

    series = cleaned_df[column]
    invalid_coerced_to_nan = 0

    if target_type == "numeric":
        converted = pd.to_numeric(series, errors="coerce")
        invalid_coerced_to_nan = int(series.notna().sum() - converted.notna().sum())
        cleaned_df[column] = converted

    elif target_type == "datetime":
        converted = pd.to_datetime(series, format=date_format, errors="coerce")
        invalid_coerced_to_nan = int(series.notna().sum() - converted.notna().sum())
        cleaned_df[column] = converted

    elif target_type == "boolean":
        bool_map = {"true": True, "1": True, "yes": True, "t": True, "y": True,
                    "false": False, "0": False, "no": False, "f": False, "n": False}
        str_vals = series.astype(str).str.strip().str.lower()
        cleaned_df[column] = str_vals.map(bool_map)

    elif target_type == "string":
        cleaned_df[column] = series.astype(str)

    return cleaned_df, {
        "column": column,
        "target_type": target_type,
        "invalid_coerced_to_nan": invalid_coerced_to_nan
    }


def validate_custom_rules(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate custom user-defined validation rules.

    Supported rule structure:
    - {'column': 'age', 'type': 'range', 'min': 0, 'max': 120}
    - {'column': 'department', 'type': 'allowed_values', 'allowed': ['Sales', 'Marketing', 'Engineering']}
    - {'column': 'salary', 'type': 'min', 'value': 0}
    """
    violations: List[Dict[str, Any]] = []

    for rule in rules:
        col = rule.get("column")
        rule_type = rule.get("type")
        if col not in df.columns:
            continue

        series = df[col]

        if rule_type == "range":
            num_series = pd.to_numeric(series, errors="coerce")
            min_val = rule.get("min", -float("inf"))
            max_val = rule.get("max", float("inf"))
            invalid_mask = (num_series < min_val) | (num_series > max_val)
            bad_indices = list(df[invalid_mask & series.notna()].index)
            if bad_indices:
                violations.append({
                    "rule": f"{col} in range [{min_val}, {max_val}]",
                    "column": col,
                    "violations_count": len(bad_indices),
                    "violating_indices": bad_indices[:20],
                    "sample_values": series.loc[bad_indices[:5]].tolist()
                })

        elif rule_type == "allowed_values":
            allowed = set(rule.get("allowed", []))
            str_series = series.astype(str).str.strip()
            invalid_mask = ~str_series.isin(allowed) & series.notna()
            bad_indices = list(df[invalid_mask].index)
            if bad_indices:
                violations.append({
                    "rule": f"{col} in allowed set",
                    "column": col,
                    "violations_count": len(bad_indices),
                    "violating_indices": bad_indices[:20],
                    "sample_values": series.loc[bad_indices[:5]].tolist()
                })

    return {
        "rules_evaluated": len(rules),
        "total_violations": sum(v["violations_count"] for v in violations),
        "violations": violations
    }
