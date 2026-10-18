"""
profiler.py - Comprehensive dataset profiling and statistical inspection for DataSentry.

Calculates:
- Global dataset dimensions and memory profile
- Inferred semantic types vs physical pandas dtypes
- Column-by-column descriptive statistics, distributions, and cardinality
- Quick summaries for numeric, categorical, boolean, and datetime fields
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an exhaustive profile of the provided DataFrame.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols if total_rows > 0 else 0

    total_missing_cells = int(df.isna().sum().sum())
    missing_cell_pct = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_row_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    memory_bytes = int(df.memory_usage(deep=True).sum())

    columns_profile: Dict[str, Any] = {}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []

    for col in df.columns:
        series = df[col]
        col_prof = profile_column(series)
        columns_profile[col] = col_prof

        col_type = col_prof["inferred_type"]
        if col_type == "numeric":
            numeric_cols.append(col)
        elif col_type == "datetime":
            datetime_cols.append(col)
        elif col_type == "boolean":
            boolean_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_cell_pct": missing_cell_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_pct": duplicate_row_pct,
        "memory_bytes": memory_bytes,
        "numeric_columns_count": len(numeric_cols),
        "categorical_columns_count": len(categorical_cols),
        "datetime_columns_count": len(datetime_cols),
        "boolean_columns_count": len(boolean_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "columns": columns_profile
    }


def profile_column(series: pd.Series) -> Dict[str, Any]:
    """
    Profile an individual pandas Series.
    """
    total_count = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / total_count * 100), 2) if total_count > 0 else 0.0
    valid_count = total_count - null_count

    non_null_series = series.dropna()
    unique_count = int(non_null_series.nunique())
    unique_pct = round((unique_count / valid_count * 100), 2) if valid_count > 0 else 0.0

    physical_dtype = str(series.dtype)
    inferred_type = infer_semantic_type(series)

    info: Dict[str, Any] = {
        "name": str(series.name),
        "physical_dtype": physical_dtype,
        "inferred_type": inferred_type,
        "total_count": total_count,
        "null_count": null_count,
        "null_percentage": null_pct,
        "valid_count": valid_count,
        "unique_count": unique_count,
        "unique_percentage": unique_pct,
        "is_constant": unique_count == 1,
        "is_all_null": null_count == total_count
    }

    # Additional type-specific metrics
    if inferred_type == "numeric" and valid_count > 0:
        # Cast safely to float for calculations
        num_vals = pd.to_numeric(non_null_series, errors="coerce").dropna()
        if len(num_vals) > 0:
            q1 = float(num_vals.quantile(0.25))
            q3 = float(num_vals.quantile(0.75))
            iqr = q3 - q1
            info.update({
                "min": float(num_vals.min()),
                "max": float(num_vals.max()),
                "mean": round(float(num_vals.mean()), 4),
                "std": round(float(num_vals.std()), 4) if len(num_vals) > 1 else 0.0,
                "median": float(num_vals.median()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "zeros_count": int((num_vals == 0).sum()),
                "zeros_pct": round(float((num_vals == 0).sum() / len(num_vals) * 100), 2),
                "negatives_count": int((num_vals < 0).sum()),
                "negatives_pct": round(float((num_vals < 0).sum() / len(num_vals) * 100), 2),
            })
            if len(num_vals) > 2:
                skew = num_vals.skew()
                info["skewness"] = round(float(skew), 4) if not np.isnan(skew) else 0.0
            else:
                info["skewness"] = 0.0
    elif inferred_type in ["categorical", "string"] and valid_count > 0:
        str_series = non_null_series.astype(str)
        # Leading/trailing whitespace count
        stripped = str_series.str.strip()
        whitespace_count = int((str_series != stripped).sum())
        info["whitespace_count"] = whitespace_count
        info["whitespace_pct"] = round((whitespace_count / valid_count * 100), 2)

        # Top values frequency
        val_counts = str_series.value_counts().head(5).to_dict()
        info["top_values"] = {str(k): int(v) for k, v in val_counts.items()}
        mode_val = str_series.mode()
        info["mode"] = str(mode_val.iloc[0]) if not mode_val.empty else None

    return info


def infer_semantic_type(series: pd.Series) -> str:
    """
    Infer semantic type (numeric, boolean, datetime, categorical).
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # For object/string dtypes, check if values can be parsed as numbers or dates
    sample = series.dropna().head(100)
    if sample.empty:
        return "string"

    # Check if boolean represented as string (True, False, Yes, No)
    bool_strings = {"true", "false", "yes", "no", "t", "f", "1", "0"}
    sample_str = sample.astype(str).str.strip().str.lower()
    if set(sample_str.unique()).issubset(bool_strings) and len(sample_str.unique()) <= 3:
        return "boolean"

    # Try numeric conversion
    numeric_converted = pd.to_numeric(sample, errors="coerce")
    if numeric_converted.notna().sum() / len(sample) >= 0.85:
        return "numeric"

    # Try datetime conversion
    try:
        dt_converted = pd.to_datetime(sample, errors="coerce")
        if dt_converted.notna().sum() / len(sample) >= 0.85:
            return "datetime"
    except Exception:
        pass

    return "categorical"
