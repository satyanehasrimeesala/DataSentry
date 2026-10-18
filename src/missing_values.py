"""
missing_values.py - Missing value detection, analysis, and imputation engine for DataSentry.

Features:
- Comprehensive null/empty-string detection
- Granular column-level and matrix-level missing statistics
- Safe imputation methods: mean, median, mode, constant, ffill/bfill, KNN
- Threshold-based dropping (row/column missing percentage)
- Safety guards against entirely empty columns or missing modes
- High-loss warnings for destructive operations
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def analyze_missing_values(df: pd.DataFrame, empty_strings_as_null: bool = True) -> Dict[str, Any]:
    """
    Analyze missing values across all columns of the dataset.

    Returns:
        Dict containing total missing cells, percentages, affected row count,
        and per-column missing statistics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols if total_rows > 0 else 0

    working_df = df.copy()
    if empty_strings_as_null:
        # Treat empty strings and pure whitespace as null
        for col in working_df.select_dtypes(include=["object", "string"]).columns:
            working_df[col] = working_df[col].replace(r"^\s*$", np.nan, regex=True)

    null_mask = working_df.isna()
    col_missing_counts = null_mask.sum().to_dict()
    rows_with_missing = int(null_mask.any(axis=1).sum())
    total_missing_cells = int(null_mask.sum().sum())

    missing_by_column: List[Dict[str, Any]] = []
    columns_with_missing: List[str] = []

    for col in df.columns:
        count = int(col_missing_counts[col])
        pct = round((count / total_rows * 100), 2) if total_rows > 0 else 0.0
        if count > 0:
            columns_with_missing.append(col)

        missing_by_column.append({
            "column": col,
            "missing_count": count,
            "missing_pct": pct,
            "dtype": str(df[col].dtype),
            "is_all_missing": count == total_rows
        })

    # Sort descending by missing percentage
    missing_by_column.sort(key=lambda x: x["missing_count"], reverse=True)

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_cell_pct": round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0,
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": round((rows_with_missing / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "columns_with_missing_count": len(columns_with_missing),
        "columns_with_missing": columns_with_missing,
        "missing_by_column": missing_by_column
    }


def impute_or_drop_missing(
    df: pd.DataFrame,
    strategy: str,
    target_columns: Optional[List[str]] = None,
    custom_value: Optional[Any] = None,
    drop_threshold_pct: float = 50.0,
    knn_neighbors: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply chosen missing value handling strategy.

    Strategies:
    - 'drop_rows': Drop rows with any missing values in target_columns.
    - 'drop_columns': Drop specified target_columns.
    - 'drop_columns_by_thresh': Drop columns exceeding drop_threshold_pct missing.
    - 'drop_rows_by_thresh': Drop rows exceeding drop_threshold_pct missing.
    - 'mean': Impute numeric target_columns with mean.
    - 'median': Impute numeric target_columns with median.
    - 'mode': Impute target_columns with mode.
    - 'constant': Impute target_columns with custom_value.
    - 'ffill': Forward-fill missing values.
    - 'bfill': Backward-fill missing values.
    - 'knn': KNN multivariate imputation for numeric columns.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and execution summary.
    """
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    initial_cols = len(cleaned_df.columns)
    initial_missing = int(cleaned_df.isna().sum().sum())

    if target_columns is None:
        target_columns = list(cleaned_df.columns)
    else:
        target_columns = [col for col in target_columns if col in cleaned_df.columns]

    warnings: List[str] = []
    affected_cells = 0

    if strategy == "drop_rows":
        subset_to_check = target_columns if target_columns else None
        cleaned_df = cleaned_df.dropna(subset=subset_to_check)
        rows_dropped = initial_rows - len(cleaned_df)
        if rows_dropped / initial_rows > 0.15:
            warnings.append(
                f"Caution: Dropping rows removed {rows_dropped} rows ({rows_dropped / initial_rows * 100:.1f}% of data)."
            )

    elif strategy == "drop_columns":
        cols_to_drop = [col for col in target_columns if col in cleaned_df.columns]
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        if len(cols_to_drop) / initial_cols > 0.3:
            warnings.append(f"Caution: Dropped {len(cols_to_drop)} columns out of {initial_cols}.")

    elif strategy == "drop_columns_by_thresh":
        missing_pcts = (cleaned_df.isna().sum() / len(cleaned_df)) * 100
        cols_to_drop = missing_pcts[missing_pcts > drop_threshold_pct].index.tolist()
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            warnings.append(f"Dropped {len(cols_to_drop)} columns with > {drop_threshold_pct}% missing: {cols_to_drop}")

    elif strategy == "drop_rows_by_thresh":
        row_missing_pcts = (cleaned_df.isna().sum(axis=1) / len(cleaned_df.columns)) * 100
        keep_mask = row_missing_pcts <= drop_threshold_pct
        cleaned_df = cleaned_df[keep_mask]
        rows_dropped = initial_rows - len(cleaned_df)
        if rows_dropped > 0:
            warnings.append(f"Dropped {rows_dropped} rows having > {drop_threshold_pct}% missing features.")

    elif strategy in ["mean", "median"]:
        for col in target_columns:
            if col not in cleaned_df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                warnings.append(f"Skipped non-numeric column '{col}' for {strategy} imputation.")
                continue

            num_missing = int(cleaned_df[col].isna().sum())
            if num_missing == 0:
                continue

            if cleaned_df[col].dropna().empty:
                warnings.append(f"Column '{col}' is entirely null. Cannot compute {strategy}.")
                continue

            replacement = cleaned_df[col].mean() if strategy == "mean" else cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(replacement)
            affected_cells += num_missing

    elif strategy == "mode":
        for col in target_columns:
            if col not in cleaned_df.columns:
                continue
            num_missing = int(cleaned_df[col].isna().sum())
            if num_missing == 0:
                continue

            modes = cleaned_df[col].dropna().mode()
            if modes.empty:
                warnings.append(f"Column '{col}' has no valid mode. Left unchanged.")
                continue

            replacement = modes.iloc[0]
            cleaned_df[col] = cleaned_df[col].fillna(replacement)
            affected_cells += num_missing

    elif strategy == "constant":
        if custom_value is None:
            custom_value = "Missing"
        for col in target_columns:
            if col not in cleaned_df.columns:
                continue
            num_missing = int(cleaned_df[col].isna().sum())
            if num_missing == 0:
                continue

            # Cast custom value appropriately if column is numeric
            val = custom_value
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                try:
                    val = float(custom_value)
                except (ValueError, TypeError):
                    warnings.append(f"Could not convert custom value '{custom_value}' to numeric for column '{col}'. Using 'Missing'.")
                    val = "Missing"
            cleaned_df[col] = cleaned_df[col].fillna(val)
            affected_cells += num_missing

    elif strategy in ["ffill", "bfill"]:
        for col in target_columns:
            if col not in cleaned_df.columns:
                continue
            num_missing = int(cleaned_df[col].isna().sum())
            if num_missing == 0:
                continue
            if strategy == "ffill":
                cleaned_df[col] = cleaned_df[col].ffill()
            else:
                cleaned_df[col] = cleaned_df[col].bfill()
            new_missing = int(cleaned_df[col].isna().sum())
            affected_cells += (num_missing - new_missing)
            if new_missing > 0:
                warnings.append(f"Column '{col}' still has {new_missing} missing values after {strategy} (boundary gaps).")

    elif strategy == "knn":
        # Apply KNN Imputer on numeric columns in target_columns
        num_cols = [c for c in target_columns if c in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[c])]
        if not num_cols:
            warnings.append("No numeric columns available for KNN imputation.")
        else:
            missing_in_num = int(cleaned_df[num_cols].isna().sum().sum())
            if missing_in_num > 0:
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                imputed_matrix = imputer.fit_transform(cleaned_df[num_cols])
                cleaned_df[num_cols] = imputed_matrix
                affected_cells += missing_in_num

    final_missing = int(cleaned_df.isna().sum().sum())
    final_rows = len(cleaned_df)
    final_cols = len(cleaned_df.columns)

    summary = {
        "strategy": strategy,
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "initial_columns": initial_cols,
        "final_columns": final_cols,
        "columns_removed": initial_cols - final_cols,
        "initial_missing_cells": initial_missing,
        "final_missing_cells": final_missing,
        "resolved_missing_cells": initial_missing - final_missing,
        "affected_cells": affected_cells,
        "warnings": warnings
    }

    return cleaned_df, summary
