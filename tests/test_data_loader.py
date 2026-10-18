"""
test_data_loader.py - Unit tests for data loading and file validation.
"""

import io
import pytest
import pandas as pd
from src.data_loader import (
    load_dataset,
    get_file_metadata,
    get_excel_sheet_names,
    DataLoaderError,
    format_bytes
)


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(1048576) == "1.00 MB"


def test_load_valid_csv():
    csv_data = io.BytesIO(b"id,name,score\n1,Alice,95.5\n2,Bob,88.0\n")
    csv_data.name = "test.csv"
    df, meta = load_dataset(csv_data)

    assert len(df) == 2
    assert list(df.columns) == ["id", "name", "score"]
    assert meta["rows"] == 2
    assert meta["columns"] == 3
    assert meta["extension"] == ".csv"


def test_load_empty_file_raises_error():
    empty_data = io.BytesIO(b"")
    empty_data.name = "empty.csv"
    with pytest.raises(DataLoaderError, match="empty"):
        load_dataset(empty_data)


def test_unsupported_extension_raises_error():
    bad_data = io.BytesIO(b"some text data")
    bad_data.name = "data.txt"
    with pytest.raises(DataLoaderError, match="Unsupported file format"):
        load_dataset(bad_data)


def test_duplicate_column_names_handled():
    # CSV with duplicate column headers
    csv_data = io.BytesIO(b"col,col,val\n1,2,3\n4,5,6\n")
    csv_data.name = "dup_cols.csv"
    df, meta = load_dataset(csv_data)

    assert len(df.columns) == 3
    assert len(set(df.columns)) == 3  # All unique
    assert meta["duplicate_columns_fixed"] is True


def test_excel_loading_and_sheet_listing():
    df_sample = pd.DataFrame({"A": [10, 20], "B": ["x", "y"]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_sample.to_excel(writer, sheet_name="DataSheet", index=False)
        df_sample.to_excel(writer, sheet_name="SecondSheet", index=False)
    buf.seek(0)
    buf.name = "test.xlsx"

    sheets = get_excel_sheet_names(buf)
    assert "DataSheet" in sheets
    assert "SecondSheet" in sheets

    df_loaded, meta = load_dataset(buf, sheet_name="DataSheet")
    assert len(df_loaded) == 2
    assert list(df_loaded.columns) == ["A", "B"]
