"""
data_loader.py - Secure and robust dataset ingestion module for DataSentry.

Supports:
- CSV (with encoding detection and separator sniffing)
- Excel (.xlsx, .xls) with multi-sheet inspection
- File size, memory usage, and shape profiling
- Comprehensive validation for empty, malformed, or corrupt files
"""

import os
import io
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd


class DataLoaderError(Exception):
    """Custom exception for data loading and validation issues."""
    pass


def get_file_metadata(file_source: Union[str, io.BytesIO, Any]) -> Dict[str, Any]:
    """
    Extract file metadata such as name, size in bytes, and extension.
    """
    if isinstance(file_source, str):
        file_name = os.path.basename(file_source)
        file_size = os.path.getsize(file_source) if os.path.exists(file_source) else 0
        extension = os.path.splitext(file_name)[1].lower()
    else:
        file_name = getattr(file_source, "name", "uploaded_file")
        file_size = getattr(file_source, "size", 0)
        if file_size == 0 and hasattr(file_source, "getbuffer"):
            file_size = file_source.getbuffer().nbytes
        extension = os.path.splitext(file_name)[1].lower()

    return {
        "file_name": file_name,
        "file_size_bytes": file_size,
        "file_size_formatted": format_bytes(file_size),
        "extension": extension
    }


def format_bytes(bytes_count: int) -> str:
    """Format byte counts to human-readable string (KB, MB, GB)."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.2f} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_count / (1024 * 1024 * 1024):.2f} GB"


def get_excel_sheet_names(file_source: Union[str, io.BytesIO, Any]) -> List[str]:
    """
    Retrieve sheet names for an Excel file without loading entire data.
    """
    try:
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        excel_file = pd.ExcelFile(file_source)
        return excel_file.sheet_names
    except Exception as e:
        raise DataLoaderError(f"Failed to inspect Excel sheets: {str(e)}")


def load_dataset(
    file_source: Union[str, io.BytesIO, Any],
    sheet_name: Optional[Union[str, int]] = 0,
    encoding: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely load a dataset from CSV or Excel file source.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Loaded DataFrame, metadata dictionary)
    Raises:
        DataLoaderError: If file is missing, empty, malformed, or unreadable.
    """
    metadata = get_file_metadata(file_source)
    ext = metadata["extension"]

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise DataLoaderError(
            f"Unsupported file format '{ext}'. DataSentry accepts only .csv, .xlsx, or .xls files."
        )

    if metadata["file_size_bytes"] == 0:
        raise DataLoaderError(
            f"The selected file '{metadata['file_name']}' is empty (0 bytes)."
        )

    df: Optional[pd.DataFrame] = None

    try:
        if hasattr(file_source, "seek"):
            file_source.seek(0)

        if ext == ".csv":
            # Attempt loading with given or fallback encodings
            encodings_to_try = [encoding] if encoding else ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
            last_err = None
            for enc in encodings_to_try:
                try:
                    if hasattr(file_source, "seek"):
                        file_source.seek(0)
                    df = pd.read_csv(file_source, encoding=enc)
                    metadata["encoding_used"] = enc
                    break
                except UnicodeDecodeError as ue:
                    last_err = ue
                    continue
                except pd.errors.EmptyDataError:
                    raise DataLoaderError(f"CSV file '{metadata['file_name']}' contains no columns or data.")
                except Exception as ex:
                    last_err = ex
                    break

            if df is None:
                raise DataLoaderError(
                    f"Could not parse CSV file '{metadata['file_name']}'. Error: {str(last_err)}"
                )

        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(file_source, sheet_name=sheet_name, engine="openpyxl" if ext == ".xlsx" else None)
                metadata["sheet_used"] = sheet_name
            except Exception as xe:
                raise DataLoaderError(
                    f"Could not parse Excel file '{metadata['file_name']}' (sheet: {sheet_name}): {str(xe)}"
                )

        if df is None or df.empty:
            if df is not None and len(df.columns) == 0:
                raise DataLoaderError("The dataset contains zero columns.")
            raise DataLoaderError(
                f"The dataset '{metadata['file_name']}' has no rows of data."
            )

        # Sanitize column names: convert all to string, strip whitespace
        import re
        raw_cols = [str(col).strip() for col in df.columns]
        duplicate_fixed = False
        new_cols = []
        col_counts = {}

        for col in raw_cols:
            match = re.match(r"^(.*?)\.(\d+)$", col)
            base_col = match.group(1) if match and match.group(1) in raw_cols else col
            if base_col != col:
                duplicate_fixed = True

            if base_col in col_counts:
                duplicate_fixed = True
                col_counts[base_col] += 1
                new_cols.append(f"{base_col}_{col_counts[base_col]}")
            else:
                col_counts[base_col] = 0
                new_cols.append(base_col)

        df.columns = new_cols
        metadata["duplicate_columns_fixed"] = duplicate_fixed

        # Compute in-memory usage
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())
        metadata.update({
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_usage_bytes": memory_usage_bytes,
            "memory_usage_formatted": format_bytes(memory_usage_bytes),
            "column_names": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
        })

        return df.copy(), metadata

    except DataLoaderError:
        raise
    except Exception as e:
        raise DataLoaderError(f"Unexpected error while reading '{metadata['file_name']}': {str(e)}")
