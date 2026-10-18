"""
test_reporting.py - Unit tests for comparison report generation and binary/text export.
"""

import pandas as pd
import numpy as np
from src.reporting import (
    generate_comparison_report,
    export_dataframe_to_csv_bytes,
    export_dataframe_to_excel_bytes,
    export_report_to_json_bytes,
    export_report_to_text
)


def test_reporting_and_exports():
    raw_df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": [10.0, 20.0, 20.0, np.nan]
    })
    clean_df = pd.DataFrame({
        "id": [1, 2, 3],
        "val": [10.0, 20.0, 30.0]
    })
    audit_log = [
        {"step": 1, "timestamp": "2026-10-18 12:00:00", "action": "REMOVE_DUPLICATES", "description": "Deduplicated", "rows_before": 4, "rows_after": 3},
        {"step": 2, "timestamp": "2026-10-18 12:01:00", "action": "MISSING_VALUES_MEAN", "description": "Imputed", "rows_before": 3, "rows_after": 3}
    ]

    report = generate_comparison_report(raw_df, clean_df, audit_log)

    assert report["dimensions_summary"]["initial_rows"] == 4
    assert report["dimensions_summary"]["final_rows"] == 3
    assert report["duplicates"]["removed_count"] == 1
    assert report["missing_values"]["resolved_count"] == 1
    assert report["quality_score"]["score_delta"] > 0

    # Test CSV export bytes
    csv_bytes = export_dataframe_to_csv_bytes(clean_df)
    assert len(csv_bytes) > 0
    assert b"id,val" in csv_bytes

    # Test Excel export bytes
    excel_bytes = export_dataframe_to_excel_bytes(clean_df)
    assert len(excel_bytes) > 0
    assert excel_bytes[:2] == b"PK"  # Zip header of xlsx file

    # Test JSON export bytes
    json_bytes = export_report_to_json_bytes(report)
    assert len(json_bytes) > 0

    # Test Text report export
    text_report = export_report_to_text(report, original_filename="raw_test.csv")
    assert "DATASENTRY - DATASET QUALITY INVESTIGATION REPORT" in text_report
    assert "REMOVE_DUPLICATES" in text_report
