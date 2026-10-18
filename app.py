"""
app.py - DataSentry: Intelligent Dataset Quality Investigator
Full-featured, interactive Streamlit Web Application.
"""

import os
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local modules
from src.data_loader import (
    load_dataset,
    get_excel_sheet_names,
    get_file_metadata,
    DataLoaderError,
    format_bytes
)
from src.profiler import profile_dataset
from src.missing_values import analyze_missing_values
from src.duplicates import detect_duplicates
from src.outliers import detect_outliers_iqr, detect_outliers_zscore
from src.consistency import analyze_consistency, validate_custom_rules
from src.quality_score import calculate_quality_score
from src.cleaning import CleaningWorkflowManager
from src.reporting import (
    generate_comparison_report,
    export_dataframe_to_csv_bytes,
    export_dataframe_to_excel_bytes,
    export_report_to_json_bytes,
    export_report_to_text
)
from src.garda_base import GardaImputer
from src.garda_upgraded import UpgradedGardaImputer, benchmark_imputation_methods


# Configure Streamlit page
st.set_page_config(
    page_title="DataSentry — Dataset Quality Investigator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-excellent { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-good { background-color: #E1EFFE; color: #1E429F; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-moderate { background-color: #FEF08A; color: #713F12; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-poor { background-color: #FFEDD5; color: #9A3412; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .badge-critical { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .banner-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if "workflow_manager" not in st.session_state:
    st.session_state.workflow_manager = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = {}


def load_file_into_session(file_obj, sheet_name=0):
    """Load and register dataset into session workflow manager."""
    try:
        df, meta = load_dataset(file_obj, sheet_name=sheet_name)
        st.session_state.workflow_manager = CleaningWorkflowManager(df)
        st.session_state.current_file_name = meta.get("file_name", "dataset")
        st.session_state.file_metadata = meta
        st.success(f"Successfully loaded '{st.session_state.current_file_name}' ({meta['rows']} rows, {meta['columns']} cols)!")
    except DataLoaderError as dle:
        st.error(f"Error loading file: {str(dle)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("### 🛡️ DataSentry")
    st.caption("Intelligent Dataset Quality Investigator")
    st.markdown("---")

    menu_options = [
        "🏠 Upload & Overview",
        "📊 Dataset Profiler",
        "🔍 Missing Values",
        "👥 Duplicate Rows",
        "📈 Outlier Detection",
        "🧹 Consistency & Types",
        "🛡️ Quality Score",
        "🔬 Research Lab (GARDA Base & Upgraded)",
        "📋 Before-After & Export"
    ]
    selected_page = st.radio("Navigation Menu", menu_options, index=0)

    st.markdown("---")
    if st.session_state.workflow_manager is not None:
        mgr = st.session_state.workflow_manager
        st.markdown(f"**Active Dataset:** `{st.session_state.current_file_name}`")
        st.markdown(f"**Current Rows:** `{len(mgr.current_data)}`")
        st.markdown(f"**Current Cols:** `{len(mgr.current_data.columns)}`")
        st.markdown(f"**History Steps:** `{mgr.history_length}`")

        col_u, col_r = st.columns(2)
        with col_u:
            if st.button("↩️ Undo", use_container_width=True, disabled=(mgr.history_length == 0)):
                res = mgr.undo()
                if res:
                    st.toast(f"Undid action: {res['reverted_action'].get('action')}")
                    st.rerun()
        with col_r:
            if st.button("🔄 Reset", use_container_width=True, disabled=(mgr.history_length == 0)):
                mgr.reset_to_original()
                st.toast("Dataset restored to raw uploaded state!")
                st.rerun()
    else:
        st.info("No dataset active. Upload a CSV or Excel file to get started.")

    st.markdown("---")
    st.caption("Pre-Final Year CSE Data Science Project | 100% Local Execution")


# ---------------- PAGE 1: UPLOAD & OVERVIEW ----------------
if selected_page == "🏠 Upload & Overview":
    st.markdown("<div class='main-title'>DataSentry: Dataset Quality Investigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Upload, inspect, diagnose, and clean datasets with complete transparency and local offline processing.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("#### 📂 Upload Dataset (CSV or Excel)")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Accepted formats: .csv, .xlsx, .xls. Files are processed locally on your machine."
        )

        sheet_selection = 0
        if uploaded_file is not None:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext in [".xlsx", ".xls"]:
                try:
                    sheets = get_excel_sheet_names(uploaded_file)
                    if len(sheets) > 1:
                        sheet_selection = st.selectbox("Select Worksheet:", sheets)
                except Exception as ex:
                    st.warning(f"Could not inspect Excel worksheets: {str(ex)}")

            if st.button("🚀 Process & Load Dataset", type="primary", use_container_width=True):
                load_file_into_session(uploaded_file, sheet_name=sheet_selection)
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🧪 Or Load Pre-configured Sample Datasets")
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("Load 'Messy Business Data' (CSV)", use_container_width=True):
                path = os.path.join("sample_data", "sample_messy_dataset.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        buf = io.BytesIO(f.read())
                        buf.name = "sample_messy_dataset.csv"
                        load_file_into_session(buf)
                        st.rerun()
                else:
                    st.error("Sample dataset file not found on disk.")
        with sc2:
            if st.button("Load 'IoT Correlated Sensors' (CSV)", use_container_width=True):
                path = os.path.join("sample_data", "iot_sensor_multivariate.csv")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        buf = io.BytesIO(f.read())
                        buf.name = "iot_sensor_multivariate.csv"
                        load_file_into_session(buf)
                        st.rerun()
                else:
                    st.error("IoT sample dataset not found on disk.")

    with col2:
        st.markdown("#### 🌟 Key Capabilities")
        st.markdown("""
        - **Accurate Profiling**: Inferred semantic types, distribution moments, zero/negative counts.
        - **Missing Value Handling**: Dropping, Mean, Median, Mode, Constant, FFill/BFill, and KNN Imputation.
        - **Deduplication**: Exact and key-column subset duplicate detection and preview.
        - **Outlier Remediation**: IQR (configurable $k$) & Z-score detection with clipping or removal.
        - **Text Normalization**: Stripping whitespace, standardizing casing, regex cleaning.
        - **Explainable Quality Score**: Weighted composite of Completeness, Uniqueness, Validity, and Consistency (0–100).
        - **IEEE Research Extension**: Base GARDA (Algorithm 1) vs Proposed Upgraded GARDA+ (Adaptive Granulation, Local Density Reconstruction & Entropy Gating).
        - **Reversible Cleaning**: Full Undo stack, reset-to-original, and downloadable CSV/XLSX/Report exports.
        """)

    # If dataset active, show overview metrics
    if st.session_state.workflow_manager is not None:
        st.markdown("---")
        st.markdown("### 📋 Active Dataset Summary")
        df_curr = st.session_state.workflow_manager.current_data
        prof = profile_dataset(df_curr)
        q_score = calculate_quality_score(df_curr)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Rows", f"{prof['total_rows']:,}")
        m2.metric("Columns", f"{prof['total_columns']:,}")
        m3.metric("Missing Cells", f"{prof['total_missing_cells']:,}", f"{prof['missing_cell_pct']}%")
        m4.metric("Duplicates", f"{prof['duplicate_rows']:,}", f"{prof['duplicate_row_pct']}%")
        m5.metric("Memory", format_bytes(prof["memory_bytes"]))
        m6.metric("Quality Score", f"{q_score['overall_score']}/100", q_score["grade"])

        st.markdown("#### Preview (First 5 Rows)")
        st.dataframe(df_curr.head(5), use_container_width=True)


# ---------------- PAGE 2: DATASET PROFILER ----------------
elif selected_page == "📊 Dataset Profiler":
    st.markdown("<div class='main-title'>Dataset Profiler & Statistical Overview</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset in the 'Upload & Overview' tab first.")
    else:
        df = st.session_state.workflow_manager.current_data
        prof = profile_dataset(df)

        tab_preview, tab_cols, tab_stats = st.tabs(["Data Preview", "Column Metadata", "Descriptive Statistics"])

        with tab_preview:
            c1, c2 = st.columns([1, 3])
            with c1:
                view_mode = st.radio("View Mode", ["Head (First N)", "Tail (Last N)", "Random Sample"])
                n_rows = st.slider("Number of rows to display", 5, min(100, len(df)), 10)
            with c2:
                if view_mode == "Head (First N)":
                    st.dataframe(df.head(n_rows), use_container_width=True)
                elif view_mode == "Tail (Last N)":
                    st.dataframe(df.tail(n_rows), use_container_width=True)
                else:
                    st.dataframe(df.sample(min(n_rows, len(df))), use_container_width=True)

        with tab_cols:
            col_rows = []
            for col_name, c_info in prof["columns"].items():
                col_rows.append({
                    "Column Name": col_name,
                    "Inferred Type": c_info["inferred_type"],
                    "Physical Dtype": c_info["physical_dtype"],
                    "Null Count": c_info["null_count"],
                    "Null %": f"{c_info['null_percentage']}%",
                    "Unique Values": c_info["unique_count"],
                    "Constant?": "Yes" if c_info["is_constant"] else "No"
                })
            st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

        with tab_stats:
            num_cols = prof["numeric_columns"]
            if num_cols:
                st.markdown("##### Numeric Features Statistics")
                num_stats = []
                for c in num_cols:
                    info = prof["columns"][c]
                    num_stats.append({
                        "Column": c,
                        "Mean": info.get("mean"),
                        "Std Dev": info.get("std"),
                        "Min": info.get("min"),
                        "Q1 (25%)": info.get("q1"),
                        "Median (50%)": info.get("median"),
                        "Q3 (75%)": info.get("q3"),
                        "Max": info.get("max"),
                        "IQR": info.get("iqr"),
                        "Skewness": info.get("skewness"),
                        "Zeros Count": info.get("zeros_count"),
                        "Negatives Count": info.get("negatives_count")
                    })
                st.dataframe(pd.DataFrame(num_stats), use_container_width=True)
            else:
                st.info("No numeric columns detected in the dataset.")


# ---------------- PAGE 3: MISSING VALUES ----------------
elif selected_page == "🔍 Missing Values":
    st.markdown("<div class='main-title'>Missing-Value Diagnostics & Imputation Workspace</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset first.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data
        missing_analysis = analyze_missing_values(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Missing Cells", f"{missing_analysis['total_missing_cells']:,}")
        c2.metric("Missing Cell Ratio", f"{missing_analysis['missing_cell_pct']}%")
        c3.metric("Rows with Missing Data", f"{missing_analysis['rows_with_missing']:,}")
        c4.metric("Columns Affected", f"{missing_analysis['columns_with_missing_count']}/{missing_analysis['total_columns']}")

        if missing_analysis["total_missing_cells"] == 0:
            st.success("🎉 Excellent! This dataset currently contains zero missing values.")
        else:
            # Interactive Plotly Bar Chart
            missing_df = pd.DataFrame(missing_analysis["missing_by_column"])
            fig = px.bar(
                missing_df[missing_df["missing_count"] > 0],
                x="column",
                y="missing_pct",
                text="missing_count",
                labels={"column": "Column Name", "missing_pct": "Missing Percentage (%)"},
                title="Missing Data Distribution Across Affected Columns",
                color="missing_pct",
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🛠️ Interactive Imputation & Remediation")

            col_action, col_opts = st.columns([1, 1.5])
            with col_action:
                strategy = st.selectbox(
                    "Select Cleaning Strategy",
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
                        "drop_columns_by_thresh"
                    ],
                    format_func=lambda s: {
                        "mean": "Numeric Mean Imputation",
                        "median": "Numeric Median Imputation",
                        "mode": "Categorical/General Mode Imputation",
                        "constant": "Fill with Custom Constant Value",
                        "ffill": "Forward Fill (Time Series / Sensor)",
                        "bfill": "Backward Fill (Time Series / Sensor)",
                        "knn": "KNN Multivariate Imputation (scikit-learn)",
                        "drop_rows": "Drop Rows with Missing Values",
                        "drop_columns": "Drop Specific Columns with Missing Data",
                        "drop_columns_by_thresh": "Drop Columns Exceeding Missing % Threshold"
                    }[s]
                )

            with col_opts:
                target_cols = None
                custom_val = None
                thresh_val = 50.0
                knn_k = 5

                if strategy in ["mean", "median", "mode", "constant", "ffill", "bfill", "drop_columns"]:
                    available = missing_analysis["columns_with_missing"] if missing_analysis["columns_with_missing"] else list(df.columns)
                    target_cols = st.multiselect("Apply to Columns:", available, default=available)

                if strategy == "constant":
                    custom_val = st.text_input("Enter Constant Replacement Value:", value="Unknown")

                if strategy == "drop_columns_by_thresh":
                    thresh_val = st.slider("Missing Percentage Threshold (%):", 10.0, 90.0, 40.0, 5.0)

                if strategy == "knn":
                    knn_k = st.slider("Number of Neighbors (K):", 2, 15, 5)

            # Warning calculation
            if strategy == "drop_rows":
                rows_to_drop = missing_analysis["rows_with_missing"]
                pct_loss = missing_analysis["rows_with_missing_pct"]
                if pct_loss > 15:
                    st.warning(f"⚠️ Warning: Dropping rows will eliminate {rows_to_drop} rows ({pct_loss}% of the dataset)!")

            if st.button("✅ Confirm & Apply Imputation", type="primary"):
                result = mgr.apply_missing_value_action(
                    strategy=strategy,
                    target_columns=target_cols,
                    custom_value=custom_val,
                    drop_threshold_pct=thresh_val,
                    knn_neighbors=knn_k
                )
                st.success(f"Operation applied successfully! Resolved {result['details']['resolved_missing_cells']} missing cells.")
                if result['details'].get('warnings'):
                    for w in result['details']['warnings']:
                        st.warning(w)
                time.sleep(1)
                st.rerun()


# ---------------- PAGE 4: DUPLICATE ROWS ----------------
elif selected_page == "👥 Duplicate Rows":
    st.markdown("<div class='main-title'>Duplicate Row Detection & Remediation</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset first.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data

        st.markdown("#### Detection Settings")
        c1, c2 = st.columns([2, 1])
        with c1:
            use_subset = st.checkbox("Check duplicates based on specific key columns (optional)", value=False)
            subset_cols = None
            if use_subset:
                subset_cols = st.multiselect("Select Key Columns for Duplicate Matching:", list(df.columns))

        dup_analysis = detect_duplicates(df, subset_columns=subset_cols)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows", f"{dup_analysis['total_rows']:,}")
        m2.metric("Duplicate Rows", f"{dup_analysis['duplicate_count']:,}", f"{dup_analysis['duplicate_pct']}%")
        m3.metric("Unique Records", f"{dup_analysis['unique_rows']:,}")

        if not dup_analysis["has_duplicates"]:
            st.success("🎉 No duplicate records found in the dataset.")
        else:
            st.markdown("#### Sample Duplicate Groups (First 15 Instances)")
            st.dataframe(dup_analysis["sample_duplicates"], use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🛠️ Deduplication Options")
            keep_opt = st.radio(
                "Which occurrence should be preserved?",
                ["first", "last", "none"],
                format_func=lambda x: {
                    "first": "Keep First Occurrence (Standard)",
                    "last": "Keep Last Occurrence",
                    "none": "Drop All Duplicates (Purge all records in duplicate groups)"
                }[x],
                horizontal=True
            )

            if st.button("🗑️ Remove Duplicate Rows", type="primary"):
                result = mgr.apply_duplicate_removal(subset_columns=subset_cols, keep_strategy=keep_opt)
                st.success(f"Removed {result['details']['rows_removed']} duplicate rows!")
                time.sleep(1)
                st.rerun()


# ---------------- PAGE 5: OUTLIER DETECTION ----------------
elif selected_page == "📈 Outlier Detection":
    st.markdown("<div class='main-title'>Statistical Outlier Detection & Winsorization</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset first.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data
        num_cols = list(df.select_dtypes(include=[np.number]).columns)

        if not num_cols:
            st.info("No numeric columns available in the dataset for outlier analysis.")
        else:
            st.markdown("""
            > [!NOTE]
            > **Statistical Note**: An outlier is an observation that lies an abnormal distance from other values.
            > Outliers are not inherently errors; they may represent genuine rare phenomena or measurement spikes.
            """)

            c1, c2, c3 = st.columns([1.5, 1, 1])
            with c1:
                selected_num_cols = st.multiselect("Select Numeric Columns to Analyze:", num_cols, default=num_cols[:min(4, len(num_cols))])
            with c2:
                method = st.selectbox("Detection Method:", ["IQR (Interquartile Range)", "Z-Score (|z| > 3.0)"])
            with c3:
                multiplier = 1.5
                if "IQR" in method:
                    multiplier = st.selectbox("IQR Multiplier (k):", [1.5, 3.0], format_func=lambda x: f"{x}x ({'Mild/Standard' if x == 1.5 else 'Extreme Only'})")

            if selected_num_cols:
                iqr_res = detect_outliers_iqr(df, target_columns=selected_num_cols, multiplier=multiplier)

                m1, m2, m3 = st.columns(3)
                m1.metric("Outlier Data Points", f"{iqr_res['total_outlier_points']:,}")
                m2.metric("Affected Unique Rows", f"{iqr_res['unique_rows_with_outliers']:,}", f"{iqr_res['unique_rows_with_outliers_pct']}%")
                m3.metric("Columns Evaluated", f"{len(selected_num_cols)}")

                # Table of bounds
                outlier_summary_table = []
                for c in selected_num_cols:
                    info = iqr_res["columns"].get(c, {})
                    if info.get("status") == "analyzed":
                        outlier_summary_table.append({
                            "Column": c,
                            "Q1 (25%)": info.get("q1"),
                            "Q3 (75%)": info.get("q3"),
                            "IQR": info.get("iqr"),
                            "Lower Bound": info.get("lower_bound"),
                            "Upper Bound": info.get("upper_bound"),
                            "Outliers Count": info.get("outlier_count"),
                            "Outliers %": f"{info.get('outlier_pct')}%"
                        })
                if outlier_summary_table:
                    st.dataframe(pd.DataFrame(outlier_summary_table), use_container_width=True)

                # Plotly Box Plot
                st.markdown("#### Interactive Box Plot Visualization")
                box_fig = px.box(df[selected_num_cols], title="Feature Distributions and Outlier Points")
                box_fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(box_fig, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 🛠️ Outlier Remediation")
                r_col1, r_col2 = st.columns([1.5, 1])
                with r_col1:
                    remediation_action = st.radio(
                        "Remediation Strategy:",
                        ["cap", "drop", "set_null"],
                        format_func=lambda x: {
                            "cap": "Cap (Winsorize): Clamp values into [Lower Bound, Upper Bound]",
                            "drop": "Drop Rows containing outliers in selected columns",
                            "set_null": "Convert outliers to NaN (resolve later via missing value workspace)"
                        }[x]
                    )
                with r_col2:
                    if st.button("Apply Outlier Remediation", type="primary"):
                        res = mgr.apply_outlier_remediation(
                            strategy=remediation_action,
                            target_columns=selected_num_cols,
                            multiplier=multiplier
                        )
                        st.success(f"Remediation applied! Affected cells: {res['details']['affected_cells']}.")
                        if res['details'].get('warnings'):
                            for w in res['details']['warnings']:
                                st.warning(w)
                        time.sleep(1)
                        st.rerun()


# ---------------- PAGE 6: CONSISTENCY & TYPES ----------------
elif selected_page == "🧹 Consistency & Types":
    st.markdown("<div class='main-title'>Data Consistency, String Normalization & Validation Rules</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset first.")
    else:
        mgr = st.session_state.workflow_manager
        df = mgr.current_data
        consistency_res = analyze_consistency(df)

        st.metric("Total Inconsistencies Detected", consistency_res["total_consistency_issues_found"])

        t_clean, t_types, t_rules = st.tabs(["Text Normalization", "Type Conversions", "Custom Validation Rules"])

        with t_clean:
            st.markdown("##### 🔤 String Cleaning & Standardization")
            ws_issues = consistency_res["whitespace_issues"]
            casing_issues = consistency_res["casing_inconsistencies"]

            if ws_issues:
                st.warning(f"Leading/trailing whitespace detected in {len(ws_issues)} columns: {list(ws_issues.keys())}")
            if casing_issues:
                st.warning(f"Inconsistent category capitalization found in {len(casing_issues)} columns: {list(casing_issues.keys())}")
                for col_name, grp in casing_issues.items():
                    st.caption(f"**{col_name} variants:** {grp['inconsistent_groups']}")

            str_cols = list(df.select_dtypes(include=["object", "string"]).columns)
            if str_cols:
                sel_str_cols = st.multiselect("Select String Columns to Normalize:", str_cols, default=str_cols)
                c_ws = st.checkbox("Strip leading & trailing whitespace", value=True)
                c_col = st.checkbox("Collapse multiple consecutive spaces into single space", value=True)
                c_case = st.selectbox("Standardize Case Style:", ["None (Keep case)", "lower", "upper", "title", "capitalize"])
                actual_case = None if c_case.startswith("None") else c_case

                if st.button("Apply Text Normalization", type="primary"):
                    res = mgr.apply_text_normalization(
                        target_columns=sel_str_cols,
                        strip_whitespace=c_ws,
                        case_style=actual_case,
                        collapse_spaces=c_col
                    )
                    st.success(f"Normalized text across {len(sel_str_cols)} columns! Affected cells: {res['details']['affected_cells']}.")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info("No string/text columns found in dataset.")

        with t_types:
            st.markdown("##### 🔄 Convert Column Data Types")
            col_to_conv = st.selectbox("Select Column to Convert:", list(df.columns))
            target_t = st.selectbox("Target Data Type:", ["numeric", "datetime", "boolean", "string"])

            dt_fmt = None
            if target_t == "datetime":
                dt_fmt = st.text_input("Date format (leave blank for automatic inference):", value="")
                if not dt_fmt.strip():
                    dt_fmt = None

            if st.button("Convert Column Type"):
                res = mgr.apply_type_conversion(column=col_to_conv, target_type=target_t, date_format=dt_fmt)
                st.success(f"Converted '{col_to_conv}' to {target_t}!")
                if res['details'].get('invalid_coerced_to_nan', 0) > 0:
                    st.warning(f"{res['details']['invalid_coerced_to_nan']} values could not be parsed and were coerced to NaN.")
                time.sleep(1)
                st.rerun()

        with t_rules:
            st.markdown("##### 📐 Define and Evaluate Validation Rules")
            st.caption("Add domain rules to verify business logic constraints.")

            rule_col = st.selectbox("Rule Target Column:", list(df.columns), key="rule_col")
            rule_type = st.radio("Constraint Type:", ["Numeric Range", "Allowed Value Set"], horizontal=True)

            rules_list = []
            if rule_type == "Numeric Range":
                rc1, rc2 = st.columns(2)
                with rc1:
                    min_v = st.number_input("Minimum Allowed Value:", value=0.0)
                with rc2:
                    max_v = st.number_input("Maximum Allowed Value:", value=100.0)
                rules_list.append({"column": rule_col, "type": "range", "min": min_v, "max": max_v})
            else:
                allowed_str = st.text_input("Allowed values (comma-separated):", value="Engineering, Sales, Marketing")
                allowed_tokens = [t.strip() for t in allowed_str.split(",") if t.strip()]
                rules_list.append({"column": rule_col, "type": "allowed_values", "allowed": allowed_tokens})

            if st.button("Evaluate Validation Rules"):
                val_res = validate_custom_rules(df, rules_list)
                if val_res["total_violations"] == 0:
                    st.success("🎉 All rows satisfy the defined validation rules!")
                else:
                    st.error(f"Found {val_res['total_violations']} rule violations!")
                    for v in val_res["violations"]:
                        st.write(f"- **{v['rule']}**: {v['violations_count']} violating rows. Sample invalid entries: {v['sample_values']}")


# ---------------- PAGE 7: QUALITY SCORE ----------------
elif selected_page == "🛡️ Quality Score":
    st.markdown("<div class='main-title'>Explainable Data Quality Score</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.workflow_manager.current_data
        score_data = calculate_quality_score(df)

        # Main Score Header
        col_score, col_badge = st.columns([1.5, 2.5])
        with col_score:
            st.markdown(
                f"<div style='font-size: 3.5rem; font-weight: 800; color: {score_data['grade_color']};'>"
                f"{score_data['overall_score']} <span style='font-size: 1.5rem; color: #64748B;'>/ 100</span></div>",
                unsafe_allow_html=True
            )
            badge_class = f"badge-{score_data['grade'].lower()}"
            st.markdown(f"<span class='{badge_class}' style='font-size: 1.1rem;'>Grade: {score_data['grade']}</span>", unsafe_allow_html=True)

        with col_badge:
            st.markdown("##### Score Formulation & Dimensions")
            st.markdown("""
            DataSentry computes a composite score across four transparent dimensions:
            - **Completeness (35%)**: Ratio of non-null cells across the full matrix.
            - **Uniqueness (25%)**: Ratio of unique, non-duplicated rows.
            - **Validity (20%)**: Statistical compliance (IQR outlier absence) and numeric validity.
            - **Consistency (20%)**: Absence of whitespace anomalies, casing mismatches, and parsing issues.
            """)

        st.markdown("---")
        st.markdown("#### Dimension Breakdown")
        d1, d2, d3, d4 = st.columns(4)
        dims = score_data["dimensions"]
        d1.metric("Completeness", f"{dims['completeness']}%", "Weight: 35%")
        d2.metric("Uniqueness", f"{dims['uniqueness']}%", "Weight: 25%")
        d3.metric("Validity", f"{dims['validity']}%", "Weight: 20%")
        d4.metric("Consistency", f"{dims['consistency']}%", "Weight: 20%")

        # Plotly Radar Chart for dimensions
        radar_categories = ["Completeness", "Uniqueness", "Validity", "Consistency"]
        radar_values = [dims["completeness"], dims["uniqueness"], dims["validity"], dims["consistency"]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            name="Current Dataset",
            line_color=score_data["grade_color"]
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Primary Score Deductions")
        if score_data["deductions"]:
            for d in score_data["deductions"]:
                st.write(f"- ❌ {d}")
        else:
            st.success("No deductions! Perfect dataset quality across all evaluated criteria.")

        st.info(score_data["disclaimer"])


# ---------------- PAGE 8: RESEARCH LAB (GARDA BASE & UPGRADED) ----------------
elif selected_page == "🔬 Research Lab (GARDA Base & Upgraded)":
    st.markdown("<div class='main-title'>Research Extension: GARDA & Proposed Upgraded GARDA+</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='banner-box'>
        <h3>IEEE IoT Journal Base Paper & Upgraded Algorithm</h3>
        <p><b>Base Paper:</b> <i>"GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks"</i> (D'Aniello et al., 2026).<br>
        <b>Our Upgraded Contribution (GARDA+):</b> Adaptive Dispersion Granulation (FD-KMeans), Local Density Continuous Reconstruction (LDR), and Conditional Entropy-Gated Error Defense.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_benchmark, tab_impute = st.tabs([
        "📖 Algorithmic Foundations",
        "🧪 Synthetic Benchmark Lab (Ground-Truth Evaluation)",
        "⚡ Impute Active Dataset with GARDA+"
    ])

    with tab_overview:
        col_b, col_u = st.columns(2)
        with col_b:
            st.markdown("### 📄 Base Paper: GARDA (IEEE 2026)")
            st.markdown("""
            **Core Mechanism**:
            1. **Granulation**: Discretizes continuous sensor time-series into interval granules using Equal-Width or K-Means.
            2. **Robust Association Rules (RAR)**: Mines rules on incomplete data using disabled-set tracking:
               $$\\text{Sup}(X) = \\frac{|\\sigma(X)|}{|S| - |\\text{Dis}(X)|}$$
               $$\\text{Conf}(X \\to Y) = \\frac{|\\sigma(X \\cup Y)|}{|\\sigma(X)| - |\\text{Dis}(Y) \\cap \\sigma(X)|}$$
            3. **Two-Pass Imputation**: High-confidence mining pass followed by an iterative relaxation pass ($r_{\\text{sup}}, r_{\\text{conf}}$).
            4. **Reconstruction**: Maps granules back to continuous values using interval midpoints or centroids.
            5. **Hybrid Fallback**: Unresolved values filled via Bayesian Ridge Regression.
            """)
        with col_u:
            st.markdown("### 🚀 Upgraded Algorithm: GARDA+")
            st.markdown("""
            **Our Novel Upgrades Addressing Paper Limitations**:
            1. **Adaptive Freedman-Diaconis Granulation**:
               Eliminates manual $\\sigma/n$ grid searching by dynamically scaling granule width $h = 2 \\cdot \\text{IQR} \\cdot n^{-1/3}$, preventing over/under-granulation.
            2. **Localized Density Continuous Reconstruction (LDR)**:
               Replaces flat static midpoints with continuous intra-granule interpolation derived from the antecedent's relative percentile position $\\alpha \\in [0, 1]$.
            3. **Entropy-Gated Error Cascade Defense**:
               Calculates Shannon entropy $H(Y \\mid X) = -\\sum P \\log_2 P$ to prune ambiguous rules before they corrupt subsequent regression phases.
            """)

    with tab_benchmark:
        st.markdown("### 🧪 Experimental Evaluation on Controlled Ground Truth")
        st.caption("Takes complete numeric rows, injects controlled synthetic missingness, and evaluates reconstruction accuracy against ground truth.")

        if st.session_state.workflow_manager is None:
            st.warning("Please upload or load a dataset first.")
        else:
            df = st.session_state.workflow_manager.current_data
            num_cols = list(df.select_dtypes(include=[np.number]).columns)

            if len(num_cols) < 2:
                st.error("At least 2 numeric/sensor columns are required for association-rule imputation benchmarking.")
            else:
                b1, b2, b3 = st.columns(3)
                with b1:
                    sel_bench_cols = st.multiselect("Benchmark Features:", num_cols, default=num_cols[:min(4, len(num_cols))])
                with b2:
                    error_mod = st.selectbox("Error Model:", ["MCAR (Random Sensor Missingness)", "Consecutive Gap (Sensor Outage/Blackout)"])
                with b3:
                    missing_rate = st.slider("Missingness Ratio (%):", 5, 50, 20, 5) / 100.0

                if st.button("🚀 Run Comprehensive Benchmark Suite", type="primary"):
                    with st.spinner("Executing benchmark across Baselines, Base GARDA, and Upgraded GARDA+..."):
                        try:
                            clean_test_df = df[sel_bench_cols].dropna()
                            if len(clean_test_df) < 10:
                                st.error("Need at least 10 complete rows without NaN to run ground-truth validation.")
                            else:
                                err_key = "mcar" if "MCAR" in error_mod else "consecutive"
                                bench_results = benchmark_imputation_methods(
                                    df_complete=clean_test_df,
                                    numeric_cols=sel_bench_cols,
                                    error_model=err_key,
                                    missing_rate=missing_rate,
                                    consecutive_gap_size=max(3, int(len(clean_test_df) * 0.15))
                                )

                                st.success(f"Benchmark completed! Evaluated on {bench_results['injected_missing_count']} injected missing cells.")

                                # Comparison Table
                                res_df = pd.DataFrame(bench_results["results_table"])
                                st.dataframe(res_df, use_container_width=True)

                                # Chart: RMSE Comparison
                                fig_rmse = px.bar(
                                    res_df,
                                    x="model",
                                    y="rmse",
                                    color="model",
                                    labels={"model": "Imputation Model", "rmse": "Root Mean Squared Error (RMSE)"},
                                    title=f"Imputation Error (RMSE) Comparison ({error_mod} @ {int(missing_rate*100)}% Missingness)"
                                )
                                fig_rmse.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig_rmse, use_container_width=True)

                                # Highlight GARDA+ Improvement
                                st.metric(
                                    "GARDA+ Improvement Over Base Paper GARDA",
                                    f"{bench_results['garda_plus_improvement_pct']}% RMSE Reduction",
                                    f"Top Performing Model: {bench_results['best_model']}"
                                )

                        except Exception as e:
                            st.error(f"Benchmark failed: {str(e)}")

    with tab_impute:
        st.markdown("### ⚡ Direct Imputation via GARDA+")
        st.caption("Apply the upgraded association-rule engine to fill actual missing cells in your active dataset.")

        if st.session_state.workflow_manager is not None:
            df = st.session_state.workflow_manager.current_data
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            missing_in_num = int(df[num_cols].isna().sum().sum()) if num_cols else 0

            st.write(f"Missing numeric cells currently: **{missing_in_num}**")

            if missing_in_num > 0:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    sup_val = st.slider("Initial Support Threshold:", 0.20, 0.70, 0.35, 0.05)
                with col_m2:
                    conf_val = st.slider("Initial Confidence Threshold:", 0.50, 0.95, 0.75, 0.05)

                if st.button("Apply GARDA+ Imputation to Dataset", type="primary"):
                    with st.spinner("Mining granular association rules and reconstructing continuous values..."):
                        imputer = UpgradedGardaImputer(init_sup=sup_val, init_conf=conf_val)
                        imputed_df, summary = imputer.fit_transform(df, num_cols)

                        st.session_state.workflow_manager.apply_custom_dataframe(
                            new_df=imputed_df,
                            action_name="GARDA_PLUS_IMPUTATION",
                            description=f"Applied GARDA+ imputation (Rules applied: {summary['rule_imputed_count']}, Fallback: {summary['fallback_imputed_count']})",
                            details=summary
                        )
                        st.success(f"GARDA+ Imputation Complete! Mined {summary['rules_mined_count']} rules in {summary['elapsed_time_sec']}s.")
                        time.sleep(1)
                        st.rerun()
            else:
                st.info("No missing numeric values to impute in the active dataset.")


# ---------------- PAGE 9: BEFORE-AFTER & EXPORT ----------------
elif selected_page == "📋 Before-After & Export":
    st.markdown("<div class='main-title'>Before-and-After Investigation Report & Cleaned Export</div>", unsafe_allow_html=True)

    if st.session_state.workflow_manager is None:
        st.warning("Please upload a dataset first.")
    else:
        mgr = st.session_state.workflow_manager
        raw_df = mgr.raw_data
        curr_df = mgr.current_data
        audit_log = mgr.audit_log

        comp = generate_comparison_report(raw_df, curr_df, audit_log)

        # Delta Summary Cards
        st.markdown("#### 🔄 Quality Improvement Overview")
        d1, d2, d3, d4 = st.columns(4)

        score_diff = comp["quality_score"]["score_delta"]
        score_arrow = "normal" if score_diff >= 0 else "inverse"
        d1.metric(
            "Quality Score",
            f"{comp['quality_score']['final_score']}/100",
            f"{score_diff:+0.1f} pts ({comp['quality_score']['initial_score']} -> {comp['quality_score']['final_score']})"
        )

        missing_diff = comp["missing_values"]["resolved_count"]
        d2.metric(
            "Missing Cells Resolved",
            f"{comp['missing_values']['final_count']:,}",
            f"-{missing_diff} resolved" if missing_diff > 0 else "0"
        )

        dup_diff = comp["duplicates"]["removed_count"]
        d3.metric(
            "Duplicates Removed",
            f"{comp['duplicates']['final_count']:,}",
            f"-{dup_diff} purged" if dup_diff > 0 else "0"
        )

        row_diff = comp["dimensions_summary"]["rows_delta"]
        d4.metric(
            "Row Count",
            f"{comp['dimensions_summary']['final_rows']:,}",
            f"{row_diff:+d} rows"
        )

        st.markdown("---")
        st.markdown("#### 📜 Chronological Cleaning Audit Trail")
        if not audit_log:
            st.info("No cleaning operations performed yet. The dataset is currently identical to the raw uploaded state.")
        else:
            log_entries = []
            for item in audit_log:
                log_entries.append({
                    "Step": item.get("step"),
                    "Timestamp": item.get("timestamp"),
                    "Action": item.get("action"),
                    "Description": item.get("description"),
                    "Rows Before": item.get("rows_before"),
                    "Rows After": item.get("rows_after")
                })
            st.dataframe(pd.DataFrame(log_entries), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 💾 Export Cleaned Dataset and Investigation Report")

        original_fn = st.session_state.current_file_name or "dataset"
        base_name = os.path.splitext(original_fn)[0]
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        exp_c1, exp_c2, exp_c3 = st.columns(3)

        with exp_c1:
            csv_bytes = export_dataframe_to_csv_bytes(curr_df)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_bytes,
                file_name=f"DataSentry_Cleaned_{base_name}_{timestamp_str}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_c2:
            xlsx_bytes = export_dataframe_to_excel_bytes(curr_df)
            st.download_button(
                label="📥 Download Cleaned Excel (.xlsx)",
                data=xlsx_bytes,
                file_name=f"DataSentry_Cleaned_{base_name}_{timestamp_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with exp_c3:
            report_text = export_report_to_text(comp, original_filename=original_fn)
            st.download_button(
                label="📄 Download Quality Report (.txt)",
                data=report_text,
                file_name=f"DataSentry_Report_{base_name}_{timestamp_str}.txt",
                mime="text/plain",
                use_container_width=True
            )
