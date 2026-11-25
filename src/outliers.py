"""
outliers.py - Statistical Outlier Detection and Remediation engine for DataSentry.

Implements:
- Interquartile Range (IQR) method (with configurable multiplier k=1.5 or 3.0)
- Z-Score method (threshold |z| > 3.0)
- Bounds calculation: Q1, Q3, IQR, Lower Bound, Upper Bound
- Safe handling for constant columns, small samples, and NaN values
- Remediation strategies: Drop outlier rows, Cap (Winsorize) at bounds, or Keep
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def detect_outliers_iqr(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Lower Bound = Q1 - (multiplier * IQR)
    Upper Bound = Q3 + (multiplier * IQR)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_columns is None:
        target_columns = list(numeric_df.columns)
    else:
        target_columns = [c for c in target_columns if c in numeric_df.columns]

    total_rows = len(df)
    results_by_column: Dict[str, Any] = {}
    total_outlier_points = 0
    all_outlier_indices = set()

    for col in target_columns:
        series = df[col].dropna()
        n_valid = len(series)

        if n_valid < 4:
            results_by_column[col] = {
                "status": "skipped_too_few_samples",
                "message": f"Sample size ({n_valid}) too small for reliable IQR.",
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None,
                "outlier_indices": []
            }
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = float(q3 - q1)

        if iqr == 0:
            results_by_column[col] = {
                "status": "constant_or_zero_iqr",
                "message": "IQR is 0 (values are constant or predominantly identical).",
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "q1": q1,
                "q3": q3,
                "iqr": 0.0,
                "lower_bound": q1,
                "upper_bound": q3,
                "outlier_indices": []
            }
            continue

        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        # Flag outliers in original DataFrame (preserving index)
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = list(df[outlier_mask].index)
        outlier_count = len(outlier_indices)
        outlier_pct = round((outlier_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        total_outlier_points += outlier_count
        all_outlier_indices.update(outlier_indices)

        results_by_column[col] = {
            "status": "analyzed",
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "outlier_indices": outlier_indices,
            "low_outliers_count": int((df[col] < lower_bound).sum()),
            "high_outliers_count": int((df[col] > upper_bound).sum())
        }

    return {
        "method": "IQR",
        "multiplier": multiplier,
        "total_rows": total_rows,
        "total_numeric_columns_checked": len(target_columns),
        "total_outlier_points": total_outlier_points,
        "unique_rows_with_outliers": len(all_outlier_indices),
        "unique_rows_with_outliers_pct": round((len(all_outlier_indices) / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "all_outlier_row_indices": sorted(list(all_outlier_indices)),
        "columns": results_by_column
    }


def detect_outliers_zscore(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect outliers using standard Z-score method (|z| > threshold).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_columns is None:
        target_columns = list(numeric_df.columns)
    else:
        target_columns = [c for c in target_columns if c in numeric_df.columns]

    total_rows = len(df)
    results_by_column: Dict[str, Any] = {}
    all_outlier_indices = set()
    total_outlier_points = 0

    for col in target_columns:
        series = df[col].dropna()
        if len(series) < 4 or series.std() == 0:
            results_by_column[col] = {
                "status": "skipped",
                "outlier_count": 0,
                "outlier_indices": []
            }
            continue

        mean = series.mean()
        std = series.std()
        z_scores = (df[col] - mean).abs() / std
        outlier_mask = z_scores > threshold
        outlier_indices = list(df[outlier_mask].index)
        outlier_count = len(outlier_indices)

        total_outlier_points += outlier_count
        all_outlier_indices.update(outlier_indices)

        results_by_column[col] = {
            "status": "analyzed",
            "mean": round(float(mean), 4),
            "std": round(float(std), 4),
            "threshold": threshold,
            "outlier_count": outlier_count,
            "outlier_pct": round((outlier_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
            "outlier_indices": outlier_indices
        }

    return {
        "method": "Z-Score",
        "threshold": threshold,
        "total_rows": total_rows,
        "total_outlier_points": total_outlier_points,
        "unique_rows_with_outliers": len(all_outlier_indices),
        "columns": results_by_column
    }


def detect_outliers_modified_zscore(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    threshold: float = 3.5
) -> Dict[str, Any]:
    """
    Detect outliers using the Modified Z-Score method with Median Absolute Deviation (MAD),
    as recommended by Iglewicz and Hoaglin (1993) and IEEE Access (2024, Ref Paper 3):
    
        M_i = 0.6745 * |x_i - median(x)| / MAD
        where MAD = median(|x_i - median(x)|)
        
    If MAD == 0, falls back to Mean Absolute Deviation about the Median.
    Threshold typically set to 3.5.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_columns is None:
        target_columns = list(numeric_df.columns)
    else:
        target_columns = [c for c in target_columns if c in numeric_df.columns]

    total_rows = len(df)
    results_by_column: Dict[str, Any] = {}
    all_outlier_indices = set()
    total_outlier_points = 0

    for col in target_columns:
        series = df[col].dropna()
        if len(series) < 4:
            results_by_column[col] = {
                "status": "skipped_too_few_samples",
                "outlier_count": 0,
                "outlier_indices": []
            }
            continue

        median = float(series.median())
        abs_deviations = (series - median).abs()
        mad = float(abs_deviations.median())

        if mad == 0.0:
            # Fallback to Mean Absolute Deviation to avoid division by zero
            mad = float(abs_deviations.mean())
            if mad == 0.0:
                results_by_column[col] = {
                    "status": "constant_values",
                    "outlier_count": 0,
                    "outlier_indices": []
                }
                continue

        # Effective equivalent bounds for Winsorization / capping:
        # |x - median| <= (threshold * mad) / 0.6745
        delta = (threshold * mad) / 0.6745
        lower_bound = median - delta
        upper_bound = median + delta

        # Modified Z-scores
        mod_z = (0.6745 * (df[col] - median).abs()) / mad
        outlier_mask = mod_z > threshold
        outlier_indices = list(df[outlier_mask].index)
        outlier_count = len(outlier_indices)

        total_outlier_points += outlier_count
        all_outlier_indices.update(outlier_indices)

        results_by_column[col] = {
            "status": "analyzed",
            "median": round(median, 4),
            "mad": round(mad, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "threshold": threshold,
            "outlier_count": outlier_count,
            "outlier_pct": round((outlier_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
            "outlier_indices": outlier_indices
        }

    return {
        "method": "Modified Z-Score (MAD)",
        "threshold": threshold,
        "total_rows": total_rows,
        "total_outlier_points": total_outlier_points,
        "unique_rows_with_outliers": len(all_outlier_indices),
        "columns": results_by_column
    }


def remediate_outliers(
    df: pd.DataFrame,
    strategy: str,
    target_columns: List[str],
    method: str = "IQR",
    multiplier: float = 1.5,
    threshold: float = 3.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remediate detected outliers using IQR, Z-Score, or Modified Z-Score (MAD).

    Strategies:
    - 'drop': Drop rows containing outliers in target_columns.
    - 'cap': Winsorize / clamp values to [lower_bound, upper_bound].
    - 'set_null': Convert outliers to NaN (to be handled later by missing-value engine).
    """
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    affected_cells = 0
    warnings: List[str] = []

    if method.upper() == "MODIFIED_ZSCORE" or "MODIFIED" in method.upper() or "MAD" in method.upper():
        detection = detect_outliers_modified_zscore(cleaned_df, target_columns, threshold=threshold)
    elif method.upper() == "ZSCORE" or "Z-SCORE" in method.upper():
        detection = detect_outliers_zscore(cleaned_df, target_columns, threshold=threshold if threshold != 3.5 else 3.0)
    else:
        detection = detect_outliers_iqr(cleaned_df, target_columns, multiplier=multiplier)

    if strategy == "drop":
        rows_to_drop = set()
        for col in target_columns:
            if col in detection["columns"] and detection["columns"][col].get("status") == "analyzed":
                rows_to_drop.update(detection["columns"][col]["outlier_indices"])
        cleaned_df = cleaned_df.drop(index=list(rows_to_drop)).copy()
        rows_removed = initial_rows - len(cleaned_df)
        if initial_rows > 0 and rows_removed / initial_rows > 0.15:
            warnings.append(f"Caution: Dropping outliers removed {rows_removed} rows ({rows_removed / initial_rows * 100:.1f}%).")

    elif strategy == "cap":
        for col in target_columns:
            col_info = detection["columns"].get(col, {})
            if col_info.get("status") == "analyzed":
                lb = col_info.get("lower_bound")
                ub = col_info.get("upper_bound")
                if lb is not None and ub is not None:
                    low_mask = cleaned_df[col] < lb
                    high_mask = cleaned_df[col] > ub
                    affected_cells += int(low_mask.sum() + high_mask.sum())
                    cleaned_df[col] = cleaned_df[col].clip(lower=lb, upper=ub)

    elif strategy == "set_null":
        for col in target_columns:
            col_info = detection["columns"].get(col, {})
            if col_info.get("status") == "analyzed":
                outlier_indices = col_info.get("outlier_indices", [])
                affected_cells += len(outlier_indices)
                cleaned_df.loc[outlier_indices, col] = np.nan

    summary = {
        "strategy": strategy,
        "method": method,
        "initial_rows": initial_rows,
        "final_rows": len(cleaned_df),
        "rows_removed": initial_rows - len(cleaned_df),
        "affected_cells": affected_cells,
        "target_columns": target_columns,
        "warnings": warnings
    }

    return cleaned_df, summary
