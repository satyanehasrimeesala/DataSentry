"""
reporting.py - Comparative Before-and-After analysis and export generator for DataSentry.

Features:
- Side-by-side metric comparison (Rows, Columns, Missing, Duplicates, Outliers, Scores)
- JSON and TXT audit report generation
- In-memory CSV and Excel binary stream exporters
- Clean export filename builder
"""

import io
import json
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from .missing_values import analyze_missing_values
from .duplicates import detect_duplicates
from .outliers import detect_outliers_iqr
from .consistency import analyze_consistency
from .quality_score import calculate_quality_score


def generate_comparison_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    audit_log: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comprehensive comparison between raw and cleaned DataFrames.
    """
    raw_score = calculate_quality_score(raw_df)
    clean_score = calculate_quality_score(cleaned_df)

    raw_missing = analyze_missing_values(raw_df)
    clean_missing = analyze_missing_values(cleaned_df)

    raw_dup = detect_duplicates(raw_df)
    clean_dup = detect_duplicates(cleaned_df)

    raw_outliers = detect_outliers_iqr(raw_df)
    clean_outliers = detect_outliers_iqr(cleaned_df)

    raw_consistency = analyze_consistency(raw_df)
    clean_consistency = analyze_consistency(cleaned_df)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dimensions_summary": {
            "initial_rows": len(raw_df),
            "final_rows": len(cleaned_df),
            "rows_delta": len(cleaned_df) - len(raw_df),
            "initial_columns": len(raw_df.columns),
            "final_columns": len(cleaned_df.columns),
            "columns_delta": len(cleaned_df.columns) - len(raw_df.columns),
        },
        "quality_score": {
            "initial_score": raw_score["overall_score"],
            "final_score": clean_score["overall_score"],
            "score_delta": round(clean_score["overall_score"] - raw_score["overall_score"], 1),
            "initial_grade": raw_score["grade"],
            "final_grade": clean_score["grade"],
            "initial_dimensions": raw_score["dimensions"],
            "final_dimensions": clean_score["dimensions"],
        },
        "missing_values": {
            "initial_count": raw_missing["total_missing_cells"],
            "final_count": clean_missing["total_missing_cells"],
            "resolved_count": raw_missing["total_missing_cells"] - clean_missing["total_missing_cells"],
            "initial_pct": raw_missing["missing_cell_pct"],
            "final_pct": clean_missing["missing_cell_pct"],
        },
        "duplicates": {
            "initial_count": raw_dup["duplicate_count"],
            "final_count": clean_dup["duplicate_count"],
            "removed_count": raw_dup["duplicate_count"] - clean_dup["duplicate_count"],
        },
        "outliers": {
            "initial_count": raw_outliers["total_outlier_points"],
            "final_count": clean_outliers["total_outlier_points"],
            "resolved_count": raw_outliers["total_outlier_points"] - clean_outliers["total_outlier_points"],
        },
        "consistency": {
            "initial_issues": raw_consistency["total_consistency_issues_found"],
            "final_issues": clean_consistency["total_consistency_issues_found"],
            "resolved_issues": raw_consistency["total_consistency_issues_found"] - clean_consistency["total_consistency_issues_found"],
        },
        "audit_log": audit_log
    }


def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export DataFrame as utf-8 CSV bytes buffer."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")


def export_dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Cleaned_Data") -> bytes:
    """Export DataFrame as Excel .xlsx bytes buffer."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def export_report_to_json_bytes(report_dict: Dict[str, Any]) -> bytes:
    """Export quality and audit report as formatted JSON bytes."""
    json_str = json.dumps(report_dict, indent=2, default=str)
    return json_str.encode("utf-8")


def export_report_to_text(report_dict: Dict[str, Any], original_filename: str) -> str:
    """Format full quality and audit report into clean readable text."""
    lines = [
        "=" * 70,
        "             DATASENTRY - DATASET QUALITY INVESTIGATION REPORT",
        "=" * 70,
        f"Generated At      : {report_dict.get('timestamp')}",
        f"Original File     : {original_filename}",
        f"Initial Dimensions: {report_dict['dimensions_summary']['initial_rows']} rows x {report_dict['dimensions_summary']['initial_columns']} columns",
        f"Cleaned Dimensions: {report_dict['dimensions_summary']['final_rows']} rows x {report_dict['dimensions_summary']['final_columns']} columns",
        "-" * 70,
        "1. QUALITY SCORE OVERVIEW",
        "-" * 70,
        f"Initial Score     : {report_dict['quality_score']['initial_score']} / 100 ({report_dict['quality_score']['initial_grade']})",
        f"Cleaned Score     : {report_dict['quality_score']['final_score']} / 100 ({report_dict['quality_score']['final_grade']})",
        f"Score Improvement : {report_dict['quality_score']['score_delta']:+0.1f} points",
        "",
        "Dimension Breakdown (Initial -> Final):",
        f"  - Completeness  : {report_dict['quality_score']['initial_dimensions']['completeness']} -> {report_dict['quality_score']['final_dimensions']['completeness']}",
        f"  - Uniqueness    : {report_dict['quality_score']['initial_dimensions']['uniqueness']} -> {report_dict['quality_score']['final_dimensions']['uniqueness']}",
        f"  - Validity      : {report_dict['quality_score']['initial_dimensions']['validity']} -> {report_dict['quality_score']['final_dimensions']['validity']}",
        f"  - Consistency   : {report_dict['quality_score']['initial_dimensions']['consistency']} -> {report_dict['quality_score']['final_dimensions']['consistency']}",
        "-" * 70,
        "2. DEFECT REMEDIATION SUMMARY",
        "-" * 70,
        f"Missing Cells     : {report_dict['missing_values']['initial_count']} -> {report_dict['missing_values']['final_count']} (Resolved: {report_dict['missing_values']['resolved_count']})",
        f"Duplicate Rows    : {report_dict['duplicates']['initial_count']} -> {report_dict['duplicates']['final_count']} (Removed: {report_dict['duplicates']['removed_count']})",
        f"Outlier Points    : {report_dict['outliers']['initial_count']} -> {report_dict['outliers']['final_count']} (Resolved: {report_dict['outliers']['resolved_count']})",
        f"Consistency Flags : {report_dict['consistency']['initial_issues']} -> {report_dict['consistency']['final_issues']} (Resolved: {report_dict['consistency']['resolved_issues']})",
        "-" * 70,
        "3. APPLIED CLEANING OPERATIONS AUDIT LOG",
        "-" * 70
    ]

    audit_log = report_dict.get("audit_log", [])
    if not audit_log:
        lines.append("  (No cleaning operations were executed.)")
    else:
        for entry in audit_log:
            lines.append(
                f"  Step {entry.get('step')}: [{entry.get('timestamp')}] {entry.get('action')} - {entry.get('description')}"
            )
            lines.append(
                f"         Rows: {entry.get('rows_before')} -> {entry.get('rows_after')} | Cols: {entry.get('columns_before', entry.get('columns_after'))} -> {entry.get('columns_after')}"
            )

    lines.extend([
        "=" * 70,
        "End of DataSentry Investigation Report",
        "=" * 70
    ])

    return "\n".join(lines)
