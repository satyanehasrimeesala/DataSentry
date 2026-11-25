"""
app.py - DataSentry: Intelligent Dataset Quality Investigator & Research Suite
Modern, decoupled, production-grade Streamlit application.
"""

import io
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Core Analytical Modules
from src.cleaning import CleaningWorkflowManager

# Application Configuration & UI Theming
from src.config import OUTLIER_CONFIG, UI_CONFIG
from src.consistency import (
    analyze_consistency,
)
from src.data_loader import DataLoaderError, get_excel_sheet_names, load_dataset
from src.duplicates import detect_duplicates
from src.missing_values import analyze_missing_values
from src.outliers import (
    detect_outliers_iqr,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
)
from src.profiler import profile_dataset
from src.quality_score import calculate_quality_score
from src.reporting import (
    export_dataframe_to_csv_bytes,
    export_dataframe_to_excel_bytes,
    export_report_to_html,
    export_report_to_text,
    generate_comparison_report,
)
from src.ui.theme import get_grade_color, get_plotly_template, inject_custom_css

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE INITIALIZATION & CACHING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{UI_CONFIG['app_name']} — Dataset Quality Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified design system
inject_custom_css()
plotly_theme = get_plotly_template()


# Session State Management
if "workflow_manager" not in st.session_state:
    st.session_state.workflow_manager = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = {}


def load_file_into_session(file_obj, sheet_name=0, filename: Optional[str] = None):
    """Load and register dataset into session workflow manager."""
    try:
        df, meta = load_dataset(file_obj, sheet_name=sheet_name)
        st.session_state.workflow_manager = CleaningWorkflowManager(df)
        st.session_state.current_file_name = filename or meta.get("file_name", "dataset")
        st.session_state.file_metadata = meta
        st.toast(f"Loaded '{st.session_state.current_file_name}' ({len(df):,} rows, {len(df.columns)} cols)", icon="✅")
    except DataLoaderError as dle:
        st.error(f"Data Loading Error: {dle!s}")
    except Exception as e:
        st.error(f"Unexpected Error during ingestion: {e!s}")


@st.cache_data(show_spinner=False)
def get_cached_profile(df: pd.DataFrame):
    """Cached dataset profiling computation to prevent redundant recalculations."""
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def get_cached_quality_score(df: pd.DataFrame):
    """Cached quality score computation."""
    return calculate_quality_score(df)


# ------------------------------------------------------------------------------
# 2. SIDEBAR NAVIGATION & PERSISTENT STATUS INDICATOR
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <h2>🛡️ {UI_CONFIG['app_name']}</h2>
            <p>{UI_CONFIG['app_tagline']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Persistent Health & Progress Card
    if st.session_state.workflow_manager is not None:
        mgr = st.session_state.workflow_manager
        curr_df = mgr.current_data
        score_info = get_cached_quality_score(curr_df)
        score_val = score_info["overall_score"]
        grade_letter = score_info["grade"]
        badge_cls = f"grade-badge-{grade_letter.lower()}"

        st.markdown(
            f"""
            <div class="status-pill">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-size:0.75rem; text-transform:uppercase; color:#64748B; font-weight:700;">Active Dataset</span>
                    <span class="grade-badge {badge_cls}">{score_val}/100 ({grade_letter})</span>
                </div>
                <div style="font-weight:700; color:#0F172A; font-size:0.95rem; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">
                    {st.session_state.current_file_name}
                </div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:4px;">
                    {len(curr_df):,} rows &bull; {len(curr_df.columns)} columns
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_u, col_r = st.columns(2)
        with col_u:
            if st.button("↩️ Undo", use_container_width=True, disabled=(mgr.history_length == 0), help="Revert the last transformation"):
                res = mgr.undo()
                if res:
                    st.toast(f"Reverted step: {res['reverted_action'].get('action')}", icon="↩️")
                    st.rerun()
        with col_r:
            if st.button("🔄 Reset", use_container_width=True, disabled=(mgr.history_length == 0), help="Restore dataset to original uploaded state"):
                mgr.reset_to_original()
                st.toast("Restored to original raw data!", icon="🔄")
                st.rerun()
    else:
        st.info("No active dataset. Upload a file or choose a benchmark dataset below.")

    st.markdown("---")

    menu_pages = [
        "🏠 Upload & Overview",
        "📊 Dataset Profiler",
        "🔍 Missing Values",
        "👥 Duplicate Rows",
        "📈 Outlier Detection",
        "🧹 Consistency & Types",
        "🛡️ Quality Score",
        "🔬 Research Lab (GARDA+)",
        "📋 Before-After & Export",
    ]
    selected_page = st.radio("Navigation", menu_pages, label_visibility="collapsed")

    st.markdown("---")
    st.caption("100% Local & Cost-Free &bull; Python 3.10+")


# ------------------------------------------------------------------------------
# 3. GLOBAL TOP METRIC STRIP (When Dataset is Active)
# ------------------------------------------------------------------------------
if st.session_state.workflow_manager is not None:
    mgr = st.session_state.workflow_manager
    raw_df = mgr.raw_data
    curr_df = mgr.current_data

    raw_score = get_cached_quality_score(raw_df)["overall_score"]
    curr_score = get_cached_quality_score(curr_df)["overall_score"]
    score_delta = round(curr_score - raw_score, 1)

    raw_missing = int(raw_df.isna().sum().sum())
    curr_missing = int(curr_df.isna().sum().sum())
    missing_delta = curr_missing - raw_missing

    raw_dups = int(raw_df.duplicated().sum())
    curr_dups = int(curr_df.duplicated().sum())
    dups_delta = curr_dups - raw_dups

    # Top KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows & Columns", f"{len(curr_df):,} × {len(curr_df.columns)}", f"{len(curr_df) - len(raw_df):+d} rows")
    k2.metric("Missing Cells", f"{curr_missing:,}", f"{missing_delta:+d} cells", delta_color="inverse")
    k3.metric("Duplicate Rows", f"{curr_dups:,}", f"{dups_delta:+d} rows", delta_color="inverse")
    k4.metric("Quality Score", f"{curr_score}/100", f"{score_delta:+0.1f} pts")
    k5.metric("Audit History", f"{mgr.history_length} operations", f"Step {mgr.history_length}")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PAGE 1: UPLOAD & OVERVIEW
# ------------------------------------------------------------------------------
if selected_page == "🏠 Upload & Overview":
    st.markdown(
        """
        <div class="app-hero-header">
            <h1>🛡️ DataSentry: Intelligent Dataset Quality Investigator</h1>
            <p>Upload, inspect, diagnose, and remediate dataset defects with explainable mathematical scoring and local execution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_info = st.columns([1.2, 0.8], gap="large")

    with col_up:
        st.markdown("<div class='section-header'>📂 Ingest Dataset (CSV or Excel)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["csv", "xlsx", "xls"],
            help="Supported: CSV, XLSX, XLS. Files remain 100% local on your machine.",
            label_visibility="collapsed",
        )

        sheet_selection = 0
        if uploaded_file is not None:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext in [".xlsx", ".xls"]:
                try:
                    sheets = get_excel_sheet_names(uploaded_file)
                    if len(sheets) > 1:
                        sheet_selection = st.selectbox("Select Excel Worksheet:", sheets)
                except Exception as ex:
                    st.warning(f"Unable to read Excel worksheets: {ex}")

            if st.button("🚀 Process & Ingest Dataset", type="primary", use_container_width=True):
                load_file_into_session(uploaded_file, sheet_name=sheet_selection)
                st.rerun()

        st.markdown("<div class='section-header'>🧪 Or Load a Pre-Configured Benchmark Dataset</div>", unsafe_allow_html=True)
        st.caption("Click any dataset card below to load a real-world benchmark into the workspace:")

        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                """
                <div class="sample-dataset-card">
                    <div>
                        <h4>📡 Intel Lab IoT Sensors</h4>
                        <p>1,005 rows &bull; 8 cols<br>Modeled after Berkeley Intel Lab telemetry with physical sensor correlations and outages.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Intel Lab IoT", use_container_width=True, key="btn_load_iot"):
                path = os.path.join("data", "intel_lab_iot_sensors.csv")
                with open(path, "rb") as f:
                    load_file_into_session(io.BytesIO(f.read()), filename="intel_lab_iot_sensors.csv")
                    st.rerun()

        with b2:
            st.markdown(
                """
                <div class="sample-dataset-card">
                    <div>
                        <h4>🌍 Global Air Quality</h4>
                        <p>803 rows &bull; 8 cols<br>EPA metropolitan air pollution telemetry across 8 global capitals with smog spikes.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Air Quality", use_container_width=True, key="btn_load_air"):
                path = os.path.join("data", "global_air_quality_cities.csv")
                with open(path, "rb") as f:
                    load_file_into_session(io.BytesIO(f.read()), filename="global_air_quality_cities.csv")
                    st.rerun()

        with b3:
            st.markdown(
                """
                <div class="sample-dataset-card">
                    <div>
                        <h4>🛍️ Customer Churn</h4>
                        <p>604 rows &bull; 8 cols<br>Demographic and sales transactions with casing anomalies, subset duplicates, and outliers.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Customer Churn", use_container_width=True, key="btn_load_churn"):
                path = os.path.join("data", "ecommerce_customer_churn_messy.csv")
                with open(path, "rb") as f:
                    load_file_into_session(io.BytesIO(f.read()), filename="ecommerce_customer_churn_messy.csv")
                    st.rerun()

    with col_info:
        st.markdown("<div class='section-header'>🌟 Architectural Highlights</div>", unsafe_allow_html=True)
        st.markdown(
            """
            - **100% Offline & Private**: Zero external network traffic, telemetry, or remote API calls.
            - **Transparent Mathematical Scoring**: 4-dimensional Quality Score (0–100) combining Completeness (35%), Uniqueness (25%), Validity (20%), and Consistency (20%).
            - **Non-Destructive Workflow**: Immutable raw copy preserved with linear Undo and Reset protection.
            - **IEEE Research Extension**: Implements base GARDA (IEEE IoT-J 2026) and novel **GARDA+** with Adaptive FD-Granulation, sign-coupled LDR, and entropy error defense.
            - **Multi-Format Reporting**: Cleaned CSV/XLSX, detailed text audit logs, and printable executive HTML reports.
            """
        )

    if st.session_state.workflow_manager is not None:
        st.markdown("---")
        st.markdown("<div class='section-header'>👀 Active Dataset Quick Preview</div>", unsafe_allow_html=True)
        st.dataframe(st.session_state.workflow_manager.current_data.head(15), use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 2: DATASET PROFILER
# ------------------------------------------------------------------------------
elif selected_page == "📊 Dataset Profiler":
    st.markdown("<div class='section-header'>📊 Automated Dataset Profiler & Statistical Breakdown</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        df = st.session_state.workflow_manager.current_data
        profile = get_cached_profile(df)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Memory Usage", f"{profile['memory_bytes'] / 1024:.1f} KB")
        p2.metric("Numeric Columns", f"{profile['numeric_columns_count']}")
        p3.metric("Categorical Columns", f"{profile['categorical_columns_count']}")
        p4.metric("Datetime Columns", f"{profile['datetime_columns_count']}")

        st.markdown("#### Column-by-Column Statistical Profiling")
        col_summary_rows = []
        for col_name, col_stats in profile["columns"].items():
            col_summary_rows.append({
                "Column Name": col_name,
                "Inferred Type": col_stats["inferred_type"],
                "Missing Cells": col_stats["null_count"],
                "Missing %": col_stats["null_pct"] / 100.0,
                "Unique Values": col_stats["distinct_count"],
                "Mean / Top": col_stats["stats"].get("mean", col_stats["stats"].get("top_value", "—")),
                "Std Dev": col_stats["stats"].get("std", "—"),
                "Min": col_stats["stats"].get("min", "—"),
                "Max": col_stats["stats"].get("max", "—"),
            })

        st.dataframe(
            pd.DataFrame(col_summary_rows),
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    "Missing %",
                    help="Percentage of null or empty values",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Interactive Distributions
        num_cols = profile["numeric_columns"]
        if num_cols:
            st.markdown("#### Feature Distribution Explorer")
            selected_feat = st.selectbox("Select Numeric Feature for Distribution Analysis:", num_cols)
            fig_hist = px.histogram(
                df,
                x=selected_feat,
                nbins=35,
                marginal="box",
                title=f"Empirical Distribution of '{selected_feat}'",
            )
            fig_hist.update_layout(template=plotly_theme)
            st.plotly_chart(fig_hist, use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 3: MISSING VALUES
# ------------------------------------------------------------------------------
elif selected_page == "🔍 Missing Values":
    st.markdown("<div class='section-header'>🔍 Missing Value Diagnostics & Multi-Strategy Imputation</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data
        missing_analysis = analyze_missing_values(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Missing Cells", f"{missing_analysis['total_missing_cells']:,}")
        m2.metric("Missing Cell Ratio", f"{missing_analysis['missing_cell_pct']}%")
        m3.metric("Affected Rows", f"{missing_analysis['rows_with_missing']:,}")

        if missing_analysis["total_missing_cells"] == 0:
            st.success("🎉 Complete dataset! Zero missing cells found across all attributes.")
        else:
            st.markdown("#### Column-Level Missing Value Matrix")
            col_missing_table = []
            for col, count in missing_analysis["missing_by_column"].items():
                if count > 0:
                    pct = round((count / len(df)) * 100, 2)
                    col_missing_table.append({"Column": col, "Missing Count": count, "Missing Ratio": pct / 100.0})

            st.dataframe(
                pd.DataFrame(col_missing_table),
                column_config={
                    "Missing Ratio": st.column_config.ProgressColumn("Missing Ratio", format="%.1f%%", min_value=0, max_value=1)
                },
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown("#### 🛠️ Apply Imputation or Drop Strategy")
            c_strat, c_opts = st.columns([1.2, 1.8], gap="large")

            with c_strat:
                strategy = st.selectbox(
                    "Select Strategy:",
                    [
                        "mean",
                        "median",
                        "mode",
                        "constant",
                        "ffill",
                        "bfill",
                        "knn",
                        "drop_rows",
                        "drop_columns",
                        "drop_columns_by_thresh",
                    ],
                    format_func=lambda s: {
                        "mean": "Mean Imputation (Numeric)",
                        "median": "Median Imputation (Robust to Outliers)",
                        "mode": "Mode Imputation (Categorical/Frequent)",
                        "constant": "Constant Value Replacement",
                        "ffill": "Forward Fill (Time Series / Sensor)",
                        "bfill": "Backward Fill (Time Series / Sensor)",
                        "knn": "KNN Multivariate Imputation",
                        "drop_rows": "Drop Rows with Missing Values",
                        "drop_columns": "Drop Specific Columns with Missing Data",
                        "drop_columns_by_thresh": "Drop Columns Exceeding Missing % Threshold",
                    }[s],
                )

            with c_opts:
                target_cols = None
                custom_val = None
                thresh_val = 50.0
                knn_k = 5

                avail_cols = missing_analysis["columns_with_missing"] or list(df.columns)
                if strategy in ["mean", "median", "mode", "constant", "ffill", "bfill", "drop_columns"]:
                    target_cols = st.multiselect("Target Columns:", avail_cols, default=avail_cols)
                if strategy == "constant":
                    custom_val = st.text_input("Constant Value:", value="Missing")
                if strategy == "drop_columns_by_thresh":
                    thresh_val = st.slider("Missingness Cutoff Threshold (%):", 10.0, 90.0, 40.0, 5.0)
                if strategy == "knn":
                    knn_k = st.slider("Number of Nearest Neighbors (K):", 2, 15, 5)

            if strategy == "drop_rows" and missing_analysis["rows_with_missing_pct"] > 15:
                st.warning(f"⚠️ Caution: Dropping rows will delete {missing_analysis['rows_with_missing']} rows ({missing_analysis['rows_with_missing_pct']}%) of your data.")

            if st.button("Confirm & Apply Imputation Action", type="primary"):
                with st.spinner("Applying imputation transformation..."):
                    result = mgr.apply_missing_value_action(
                        strategy=strategy,
                        target_columns=target_cols,
                        custom_value=custom_val,
                        drop_threshold_pct=thresh_val,
                        knn_neighbors=knn_k,
                    )
                    st.success(f"Applied successfully! Resolved {result['details']['resolved_missing_cells']} missing cells.")
                    time.sleep(0.5)
                    st.rerun()


# ------------------------------------------------------------------------------
# PAGE 4: DUPLICATE ROWS
# ------------------------------------------------------------------------------
elif selected_page == "👥 Duplicate Rows":
    st.markdown("<div class='section-header'>👥 Exact & Subset Duplicate Row Detection</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data

        c_opt, _ = st.columns([2, 1])
        with c_opt:
            use_subset = st.checkbox("Check duplicates based on specific primary key columns", value=False)
            subset_cols = None
            if use_subset:
                subset_cols = st.multiselect("Select Key Columns:", list(df.columns))

        dup_res = detect_duplicates(df, subset_columns=subset_cols)

        d1, d2, d3 = st.columns(3)
        d1.metric("Total Records", f"{dup_res['total_rows']:,}")
        d2.metric("Duplicate Rows", f"{dup_res['duplicate_count']:,}", f"{dup_res['duplicate_pct']}%")
        d3.metric("Unique Records", f"{dup_res['unique_rows']:,}")

        if not dup_res["has_duplicates"]:
            st.success("🎉 Dataset is clean of duplicate records.")
        else:
            st.markdown("#### Sample Duplicate Groups")
            st.dataframe(dup_res["sample_duplicates"], use_container_width=True)

            st.markdown("---")
            st.markdown("#### Deduplication Strategy")
            keep_rule = st.radio(
                "Preservation Policy:",
                ["first", "last", "none"],
                format_func=lambda x: {
                    "first": "Keep First Occurrence (Standard)",
                    "last": "Keep Last Occurrence",
                    "none": "Purge All Instances of Duplicate Records",
                }[x],
                horizontal=True,
            )

            if st.button("🗑️ Execute Deduplication", type="primary"):
                result = mgr.apply_duplicate_removal(subset_columns=subset_cols, keep_strategy=keep_rule)
                st.success(f"Removed {result['details']['rows_removed']} duplicate rows!")
                time.sleep(0.5)
                st.rerun()


# ------------------------------------------------------------------------------
# PAGE 5: OUTLIER DETECTION
# ------------------------------------------------------------------------------
elif selected_page == "📈 Outlier Detection":
    st.markdown("<div class='section-header'>📈 Statistical Outlier Detection & Winsorization Studio</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if not numeric_cols:
            st.info("No numeric features available for outlier detection.")
        else:
            c1, c2, c3 = st.columns([1.5, 1.2, 1], gap="medium")
            with c1:
                selected_num = st.multiselect("Numeric Columns to Inspect:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
            with c2:
                method = st.selectbox(
                    "Detection Methodology:",
                    [
                        "Modified Z-Score (MAD / Hybrid Scale - IEEE Access 2024)",
                        "IQR (Interquartile Range)",
                        "Z-Score (|z| > 3.0)",
                    ],
                )
            with c3:
                multiplier = OUTLIER_CONFIG["iqr_standard_multiplier"]
                thresh = OUTLIER_CONFIG["modified_zscore_mad_threshold"]
                if "IQR" in method:
                    multiplier = st.selectbox("IQR Multiplier (k):", [1.5, 3.0], format_func=lambda x: f"{x}x ({'Mild/Standard' if x == 1.5 else 'Extreme Only'})")
                elif "Modified" in method:
                    thresh = st.slider("Modified Z-Score Cutoff:", 2.5, 5.0, 3.5, 0.5, help="Standard MAD threshold recommended by Iglewicz and Hoaglin")
                else:
                    thresh = st.slider("Z-Score Cutoff (σ):", 2.0, 4.0, 3.0, 0.5)

            if selected_num:
                if "Modified" in method:
                    out_res = detect_outliers_modified_zscore(df, target_columns=selected_num, threshold=thresh)
                    method_key = "MODIFIED_ZSCORE"
                elif "IQR" in method:
                    out_res = detect_outliers_iqr(df, target_columns=selected_num, multiplier=multiplier)
                    method_key = "IQR"
                else:
                    out_res = detect_outliers_zscore(df, target_columns=selected_num, threshold=thresh)
                    method_key = "ZSCORE"

                o1, o2, o3 = st.columns(3)
                o1.metric("Outlier Data Points", f"{out_res['total_outlier_points']:,}")
                o2.metric("Affected Unique Rows", f"{out_res['unique_rows_with_outliers']:,}")
                o3.metric("Evaluated Features", f"{len(selected_num)}")

                out_summary = []
                for c in selected_num:
                    info = out_res["columns"].get(c, {})
                    if info.get("status") == "analyzed":
                        row_dict = {
                            "Feature": c,
                            "Outliers Found": info.get("outlier_count"),
                            "Outlier %": f"{info.get('outlier_pct')}%",
                            "Lower Bound": info.get("lower_bound", "—"),
                            "Upper Bound": info.get("upper_bound", "—"),
                        }
                        out_summary.append(row_dict)
                if out_summary:
                    st.dataframe(pd.DataFrame(out_summary), use_container_width=True, hide_index=True)

                st.markdown("#### Interactive Box & Outlier Distribution")
                box_fig = px.box(df[selected_num], points="outliers", title="Feature Box Plots with Identified Outliers")
                box_fig.update_layout(template=plotly_theme, height=380)
                st.plotly_chart(box_fig, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 🛠️ Outlier Remediation Strategy")
                r_c1, r_c2 = st.columns([1.5, 1])
                with r_c1:
                    remed_opt = st.radio(
                        "Remediation Action:",
                        ["cap", "drop", "set_null"],
                        format_func=lambda x: {
                            "cap": "Winsorize (Cap): Clamp extreme values into [Lower Bound, Upper Bound]",
                            "drop": "Drop Rows containing identified outliers",
                            "set_null": "Nullify: Convert outliers to NaN for later imputation",
                        }[x],
                    )
                with r_c2:
                    if st.button("Apply Outlier Remediation", type="primary"):
                        res = mgr.apply_outlier_remediation(
                            strategy=remed_opt,
                            target_columns=selected_num,
                            method=method_key,
                            multiplier=multiplier,
                            threshold=thresh,
                        )
                        st.success(f"Outlier remediation complete! Clamped/affected cells: {res['details']['affected_cells']}.")
                        time.sleep(0.5)
                        st.rerun()


# ------------------------------------------------------------------------------
# PAGE 6: CONSISTENCY & TYPES
# ------------------------------------------------------------------------------
elif selected_page == "🧹 Consistency & Types":
    st.markdown("<div class='section-header'>🧹 Data Consistency, Normalization & Custom Validation Rules</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data
        cons_analysis = analyze_consistency(df)

        c1, c2 = st.columns(2)
        c1.metric("Consistency Issues Found", f"{cons_analysis['total_consistency_issues_found']:,}")
        c2.metric("Whitespace Anomaly Columns", f"{len(cons_analysis['whitespace_issues'])}")

        st.markdown("#### Text Normalization & Casing Harmonization")
        str_cols = list(df.select_dtypes(include=["object", "string"]).columns)
        if str_cols:
            t_col1, t_col2 = st.columns([1.5, 1])
            with t_col1:
                target_str = st.multiselect("Select String Columns to Normalize:", str_cols, default=str_cols[:min(3, len(str_cols))])
                case_action = st.selectbox("Standardize Casing:", ["none", "lower", "upper", "title", "capitalize"])
            with t_col2:
                strip_ws = st.checkbox("Strip leading/trailing whitespace", value=True)
                collapse_sp = st.checkbox("Collapse multiple internal spaces", value=True)

                if st.button("Apply Text Normalization", type="primary"):
                    res = mgr.apply_text_normalization(
                        target_columns=target_str,
                        strip_whitespace=strip_ws,
                        case_style=None if case_action == "none" else case_action,
                        collapse_spaces=collapse_sp,
                    )
                    st.success(f"Normalized {res['details']['affected_cells']} cells across {len(target_str)} columns.")
                    time.sleep(0.5)
                    st.rerun()


# ------------------------------------------------------------------------------
# PAGE 7: QUALITY SCORECARD
# ------------------------------------------------------------------------------
elif selected_page == "🛡️ Quality Score":
    st.markdown("<div class='section-header'>🛡️ Explainable 4-Dimensional Quality Scorecard</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        df = st.session_state.workflow_manager.current_data
        score_data = get_cached_quality_score(df)

        sc_left, sc_right = st.columns([1.2, 1.8], gap="large")
        with sc_left:
            grade_color = get_grade_color(score_data["overall_score"])
            badge_cls = f"grade-badge-{score_data['grade'].lower()}"
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:24px; text-align:center;">
                    <div style="font-size:0.85rem; text-transform:uppercase; color:#64748B; font-weight:700;">Overall Quality Score</div>
                    <div style="font-size:3.8rem; font-weight:800; color:{grade_color}; line-height:1.1;">
                        {score_data['overall_score']} <span style="font-size:1.5rem; color:#94A3B8;">/ 100</span>
                    </div>
                    <div style="margin-top:10px;">
                        <span class="grade-badge {badge_cls}" style="font-size:1.1rem; padding:6px 16px;">Grade {score_data['grade']}</span>
                    </div>
                    <p style="font-size:0.85rem; color:#64748B; margin-top:12px;">{score_data['disclaimer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sc_right:
            st.markdown("#### Mathematical Formulation & Weights")
            st.latex(r"\text{Score} = 0.35 \cdot \text{Comp} + 0.25 \cdot \text{Uniq} + 0.20 \cdot \text{Val} + 0.20 \cdot \text{Cons}")
            dims = score_data["dimensions"]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Completeness", f"{dims['completeness']}%", "35% Weight")
            d2.metric("Uniqueness", f"{dims['uniqueness']}%", "25% Weight")
            d3.metric("Validity", f"{dims['validity']}%", "20% Weight")
            d4.metric("Consistency", f"{dims['consistency']}%", "20% Weight")

        st.markdown("---")
        rad_col, ded_col = st.columns([1.2, 1.8], gap="large")

        with rad_col:
            st.markdown("#### Quality Radar Fingerprint")
            cats = ["Completeness", "Uniqueness", "Validity", "Consistency"]
            vals = [dims["completeness"], dims["uniqueness"], dims["validity"], dims["consistency"]]

            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                line_color=grade_color,
                name="Quality",
            ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=350,
                margin=dict(l=30, r=30, t=20, b=20),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        with ded_col:
            st.markdown("#### Transparent Deductions & Defect Audit")
            if score_data["deductions"]:
                for d in score_data["deductions"]:
                    st.write(f"- ❌ {d}")
            else:
                st.success("Zero deductions! Dataset exhibits perfect quality across all evaluated criteria.")


# ------------------------------------------------------------------------------
# PAGE 8: RESEARCH LAB (GARDA BASE & UPGRADED GARDA+)
# ------------------------------------------------------------------------------
elif selected_page == "🔬 Research Lab (GARDA+)":
    st.markdown("<div class='section-header'>🔬 IEEE Research Extension: GARDA vs Proposed GARDA+</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="stat-card" style="margin-bottom:20px;">
            <div style="font-weight:700; color:#1E3A8A; font-size:1.05rem;">IEEE Internet of Things Journal Base Paper & Upgraded Algorithm</div>
            <p style="font-size:0.85rem; color:#475569; margin:4px 0 0 0;">
                <b>Base Paper:</b> <i>"GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks"</i> (D'Aniello et al., 2026).<br>
                <b>Our Upgrades:</b> Adaptive Freedman-Diaconis Granulation, Sign-Coupled Localized Density Reconstruction (LDR), and Shannon Entropy Error Defense.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_theory, t_bench, t_apply = st.tabs([
        "📖 Algorithmic Innovations",
        "🧪 Ground-Truth Benchmark Lab",
        "⚡ Impute Active Dataset with GARDA+",
    ])

    with t_theory:
        th1, th2 = st.columns(2, gap="large")
        with th1:
            st.markdown("### 📄 Base Paper: GARDA (IEEE 2026)")
            st.markdown(
                """
                1. **Granulation**: Discretizes sensor streams into fixed intervals using K-Means.
                2. **Robust Association Rules (RAR)**: Mines rules on incomplete matrices using disabled-sets:
                """
            )
            st.latex(r"\text{Conf}(X \to Y) = \frac{|\sigma(X \cup Y)|}{|\sigma(X)| - |\text{Dis}(Y) \cap \sigma(X)|}")
            st.markdown(
                r"""
                3. **Limitation**: Reconstructs continuous values as flat interval midpoints $(a+b)/2$, discarding intra-granule trajectory and causing systematic quantization error $O(\lambda^2/12)$.
                """
            )
        with th2:
            st.markdown("### 🚀 Our Innovation: GARDA+")
            st.markdown(
                r"""
                1. **Adaptive Dispersion Granulation**: Automatically sizes bins via Freedman-Diaconis rule $h = 2 \cdot \text{IQR} \cdot n^{-1/3}$.
                2. **Sign-Coupled Local Density Reconstruction (LDR)**:
                """
            )
            st.latex(r"\hat{y} = b_{\text{low}} + \alpha_{\text{coupled}} \cdot (b_{\text{high}} - b_{\text{low}})")
            st.markdown(
                r"""
                3. **Entropy Gating (EG-Defense)**: Prunes ambiguous rules where Shannon conditional entropy $H(Y \mid X) > \theta_H$.
                """
            )

    with t_bench:
        st.markdown("#### Synthetic Error Injection & Ground-Truth Benchmarking")
        st.caption("Takes complete numeric rows, injects controlled synthetic missingness, and evaluates reconstruction accuracy against true values.")

        if st.session_state.workflow_manager is None:
            st.info("Please ingest or select a dataset on the Upload & Overview page.")
        else:
            df = st.session_state.workflow_manager.current_data
            num_cols = list(df.select_dtypes(include=[np.number]).columns)

            if len(num_cols) < 2:
                st.error("At least 2 numeric features required for multivariate benchmark.")
            else:
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    sel_bench = st.multiselect("Benchmark Features:", num_cols, default=num_cols[:min(4, len(num_cols))])
                with b_c2:
                    err_type = st.selectbox("Error Pattern:", ["MCAR (Random Sensor Drops)", "Consecutive Gap (Sensor Outage Blackout)"])
                with b_c3:
                    miss_ratio = st.slider("Missingness Ratio (%):", 5, 50, 20, 5) / 100.0

                if st.button("🚀 Run Comprehensive 7-Model Benchmark Suite", type="primary"):
                    with st.spinner("Executing benchmark across Baselines, Multi-SNN, Base GARDA, and Upgraded GARDA+..."):
                        clean_bench_df = df[sel_bench].dropna()
                        if len(clean_bench_df) < 10:
                            st.error("Need at least 10 complete rows without NaN for controlled benchmark.")
                        else:
                            from src.garda_upgraded import benchmark_imputation_methods
                            err_key = "mcar" if "MCAR" in err_type else "consecutive"
                            res = benchmark_imputation_methods(
                                df_complete=clean_bench_df,
                                numeric_cols=sel_bench,
                                error_model=err_key,
                                missing_rate=miss_ratio,
                                consecutive_gap_size=max(3, int(len(clean_bench_df) * 0.15)),
                            )
                            st.success(f"Benchmark finished! Evaluated on {res['injected_missing_count']} injected missing cells.")

                            res_table = pd.DataFrame(res["results_table"])
                            st.dataframe(res_table, use_container_width=True, hide_index=True)

                            # Plotly Bar Chart: RMSE
                            fig_b = px.bar(
                                res_table,
                                x="model",
                                y="rmse",
                                color="model",
                                labels={"rmse": "Root Mean Squared Error (RMSE)", "model": "Algorithm"},
                                title="Imputation Accuracy Benchmark (Lower RMSE = Superior Performance)",
                            )
                            fig_b.update_layout(template=plotly_theme, showlegend=False, height=380)
                            st.plotly_chart(fig_b, use_container_width=True)

    with t_apply:
        st.markdown("#### Impute Missing Values in Active Dataset using GARDA+")
        if st.session_state.workflow_manager is not None:
            df = st.session_state.workflow_manager.current_data
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            missing_count = int(df[num_cols].isna().sum().sum()) if num_cols else 0

            st.write(f"Missing numeric cells currently: **{missing_count:,}**")
            if missing_count > 0:
                if st.button("Apply GARDA+ Imputation to Dataset", type="primary"):
                    with st.spinner("Mining rules and executing localized continuous reconstruction..."):
                        from src.garda_upgraded import UpgradedGardaImputer
                        imputer = UpgradedGardaImputer()
                        imputed_df, summ = imputer.fit_transform(df, num_cols)
                        st.session_state.workflow_manager.apply_custom_dataframe(
                            new_df=imputed_df,
                            action_name="GARDA_PLUS_IMPUTATION",
                            description=f"Applied GARDA+ imputation (Rules applied: {summ['rule_imputed_count']}, Fallback: {summ['fallback_imputed_count']})",
                            details=summ,
                        )
                        st.success(f"GARDA+ Complete! Mined {summ['rules_mined_count']} rules in {summ['elapsed_time_sec']}s.")
                        time.sleep(0.5)
                        st.rerun()
            else:
                st.info("Zero missing numeric cells in the active dataset.")


# ------------------------------------------------------------------------------
# PAGE 9: BEFORE-AFTER & EXPORT
# ------------------------------------------------------------------------------
elif selected_page == "📋 Before-After & Export":
    st.markdown("<div class='section-header'>📋 Before-and-After Comparison & Production Export</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.info("Please ingest or select a dataset on the Upload & Overview page.")
    else:
        mgr = st.session_state.workflow_manager
        raw_df = mgr.raw_data
        curr_df = mgr.current_data
        audit_log = mgr.audit_log

        comp = generate_comparison_report(raw_df, curr_df, audit_log)

        # Delta Cards
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Quality Score", f"{comp['quality_score']['final_score']}/100", f"{comp['quality_score']['score_delta']:+0.1f} pts")
        d2.metric("Missing Resolved", f"{comp['missing_values']['resolved_count']:,} cells", f"{comp['missing_values']['final_count']} remaining")
        d3.metric("Duplicates Purged", f"{comp['duplicates']['removed_count']:,} rows", f"{comp['duplicates']['final_count']} remaining")
        d4.metric("Outliers Remediated", f"{comp['outliers']['resolved_count']:,} points", f"{comp['outliers']['final_count']} remaining")

        st.markdown("---")
        st.markdown("#### Chronological Cleaning Audit Log")
        if not audit_log:
            st.info("No cleaning operations executed yet. Current dataset is identical to raw upload.")
        else:
            log_table = []
            for item in audit_log:
                log_table.append({
                    "Step": item.get("step"),
                    "Action": item.get("action"),
                    "Description": item.get("description"),
                    "Rows Delta": f"{item.get('rows_before')} → {item.get('rows_after')}",
                    "Timestamp": item.get("timestamp"),
                })
            st.dataframe(pd.DataFrame(log_table), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 💾 Export Cleaned Dataset & Reports")

        orig_fn = st.session_state.current_file_name or "dataset"
        base_fn = os.path.splitext(orig_fn)[0]
        ts_str = time.strftime("%Y%m%d_%H%M%S")

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            csv_b = export_dataframe_to_csv_bytes(curr_df)
            st.download_button("📥 Cleaned CSV", data=csv_b, file_name=f"DataSentry_Cleaned_{base_fn}_{ts_str}.csv", mime="text/csv", use_container_width=True)
        with e2:
            xlsx_b = export_dataframe_to_excel_bytes(curr_df)
            st.download_button("📥 Cleaned Excel (.xlsx)", data=xlsx_b, file_name=f"DataSentry_Cleaned_{base_fn}_{ts_str}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with e3:
            txt_rep = export_report_to_text(comp, original_filename=orig_fn)
            st.download_button("📄 Audit Log (.txt)", data=txt_rep, file_name=f"DataSentry_Report_{base_fn}_{ts_str}.txt", mime="text/plain", use_container_width=True)
        with e4:
            html_rep = export_report_to_html(comp, original_filename=orig_fn)
            st.download_button("🖨️ Executive Report (HTML)", data=html_rep.encode("utf-8"), file_name=f"DataSentry_Executive_{base_fn}_{ts_str}.html", mime="text/html", use_container_width=True)
