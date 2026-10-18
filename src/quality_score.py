"""
quality_score.py - Transparent, explainable Data Quality Score calculation engine for DataSentry.

Computes four defensible dimensions (0 to 100):
1. Completeness: Non-null ratio across all data cells
2. Uniqueness: Non-duplicate row ratio
3. Validity: Adherence to statistical bounds (outliers) and valid type parsing
4. Consistency: Clean string formatting, normalized casing, and date integrity

Aggregates dimensions via configurable weights with grade badges and impact explanations.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .missing_values import analyze_missing_values
from .duplicates import detect_duplicates
from .outliers import detect_outliers_iqr
from .consistency import analyze_consistency


DEFAULT_WEIGHTS = {
    "completeness": 0.35,
    "uniqueness": 0.25,
    "validity": 0.20,
    "consistency": 0.20
}


def calculate_quality_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate an explainable Data Quality Score from 0 to 100.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    else:
        # Normalize weights so they sum to 1.0
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            weights = DEFAULT_WEIGHTS.copy()

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    if total_rows == 0 or total_cols == 0:
        return {
            "overall_score": 0.0,
            "grade": "Critical",
            "dimensions": {
                "completeness": 0.0,
                "uniqueness": 0.0,
                "validity": 0.0,
                "consistency": 0.0
            },
            "weights": weights,
            "deductions": ["Dataset is empty."],
            "summary_text": "Empty dataset cannot be scored."
        }

    # 1. Completeness Dimension
    missing_analysis = analyze_missing_values(df)
    total_missing = missing_analysis["total_missing_cells"]
    completeness_score = max(0.0, 100.0 * (1.0 - (total_missing / total_cells)))

    # 2. Uniqueness Dimension
    dup_analysis = detect_duplicates(df)
    dup_count = dup_analysis["duplicate_count"]
    uniqueness_score = max(0.0, 100.0 * (1.0 - (dup_count / total_rows)))

    # 3. Validity Dimension (Outliers in numeric data)
    outlier_analysis = detect_outliers_iqr(df)
    total_outliers = outlier_analysis["total_outlier_points"]
    # Evaluated against numeric cells only
    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cells = total_rows * len(numeric_cols)
    if numeric_cells > 0:
        validity_score = max(0.0, 100.0 * (1.0 - (total_outliers / numeric_cells)))
    else:
        validity_score = 100.0

    # 4. Consistency Dimension (Whitespace, casing, mixed types)
    consistency_analysis = analyze_consistency(df)
    total_consistency_issues = consistency_analysis["total_consistency_issues_found"]
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    string_cells = total_rows * len(string_cols)
    if string_cells > 0:
        consistency_score = max(0.0, 100.0 * (1.0 - (total_consistency_issues / string_cells)))
    else:
        consistency_score = 100.0

    # Overall weighted composite
    overall = (
        (completeness_score * weights["completeness"]) +
        (uniqueness_score * weights["uniqueness"]) +
        (validity_score * weights["validity"]) +
        (consistency_score * weights["consistency"])
    )
    overall = round(max(0.0, min(100.0, overall)), 1)

    # Determine Grade and Color
    if overall >= 90.0:
        grade = "Excellent"
        grade_color = "#28a745"
    elif overall >= 75.0:
        grade = "Good"
        grade_color = "#17a2b8"
    elif overall >= 60.0:
        grade = "Moderate"
        grade_color = "#ffc107"
    elif overall >= 40.0:
        grade = "Poor"
        grade_color = "#fd7e14"
    else:
        grade = "Critical"
        grade_color = "#dc3545"

    # Identify primary score deductions
    deductions = []
    if total_missing > 0:
        deductions.append(
            f"Completeness deduction: {total_missing} missing cells ({missing_analysis['missing_cell_pct']}% of total data)."
        )
    if dup_count > 0:
        deductions.append(
            f"Uniqueness deduction: {dup_count} duplicate rows ({dup_analysis['duplicate_pct']}% of records)."
        )
    if total_outliers > 0:
        deductions.append(
            f"Validity deduction: {total_outliers} statistical outliers detected via IQR (1.5x multiplier)."
        )
    if total_consistency_issues > 0:
        deductions.append(
            f"Consistency deduction: {total_consistency_issues} anomalies found (whitespace / casing / date issues)."
        )

    return {
        "overall_score": overall,
        "grade": grade,
        "grade_color": grade_color,
        "dimensions": {
            "completeness": round(completeness_score, 1),
            "uniqueness": round(uniqueness_score, 1),
            "validity": round(validity_score, 1),
            "consistency": round(consistency_score, 1)
        },
        "weights": weights,
        "deductions": deductions,
        "raw_counts": {
            "total_cells": total_cells,
            "missing_cells": total_missing,
            "duplicate_rows": dup_count,
            "outlier_points": total_outliers,
            "consistency_issues": total_consistency_issues
        },
        "disclaimer": (
            "DataSentry Quality Score is an application-defined analytical metric based on "
            "completeness, uniqueness, statistical validity, and textual consistency. "
            "It is intended for guidance and comparative assessment before and after cleaning."
        )
    }
