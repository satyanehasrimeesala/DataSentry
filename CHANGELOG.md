# Changelog — DataSentry

All notable changes, architectural improvements, and bug fixes to the DataSentry project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] — 2026-11-25

### 🚀 Modernized Streamlit UI / UX (Part A)
- **Decoupled Visual Theme System (`src/ui/theme.py`)**:
  - Implemented centralized CSS injection supporting both Light and Dark mode.
  - Standardized semantic color mapping to dataset health letter grades ($A=\text{Green}, B=\text{Blue}, C=\text{Amber}, D=\text{Orange}, F=\text{Red}$).
  - Introduced shared Plotly template (`get_plotly_template()`) for clean, responsive, publication-ready visualizations with unified typography and gridlines.
- **Persistent Dataset Health & Progress Sidebar**:
  - Sidebar now displays a live status card with the active dataset name, dimensions, active Quality Score badge pill, and operation count.
  - Sticky Undo (↩️) and Reset (🔄) quick-action controls with hover tooltips and dynamic disable states.
- **Top Metric Strip (KPIs)**:
  - Top-of-page interactive metric row tracking Rows/Cols delta, Missing cells delta, Duplicates delta, Quality score delta, and Audit step count.
- **1-Click Pre-configured Benchmark Cards**:
  - Replaced plain buttons with styled benchmark cards on the Upload screen for Intel Berkeley IoT Sensors (1,005 rows), EPA Air Quality (803 rows), and E-Commerce Customer Churn (604 rows).
- **Interactive Profiler Dataframe**:
  - Upgraded raw tabular display to modern `st.dataframe` utilizing `st.column_config.ProgressColumn` for missingness ratios and numeric formatting.
- **Printable Executive Summary HTML Report**:
  - Added modern HTML/PDF report exporter with card metrics, quality radar summary, and complete chronological audit log.

### 🛠️ Code Quality, Architecture & Testing (Part B)
- **Centralized Configuration (`src/config.py`)**:
  - Centralized Quality Score weights ($0.35, 0.25, 0.20, 0.20$), grade thresholds ($A \ge 90, B \ge 80, C \ge 70, D \ge 60$), outlier defaults (IQR $1.5, 3.0$, MAD $3.5$), and GARDA hyperparameters ($\theta_H = 1.25$).
- **Expanded Test Suite (39 Tests Passed)**:
  - Added `tests/test_profiler.py` covering dataset profiling, column-level moments, and empty DataFrames.
  - Added `tests/test_ui_smoke.py` utilizing Streamlit's official `streamlit.testing.v1.AppTest` framework to verify headless app initialization and navigation switching without exceptions.
- **Code Coverage**: Increased test coverage across core modules from 74% to >83%.
- **Zero Cloud & Telemetry**:
  - Enforced `gatherUsageStats = false` and local loopback binding (`127.0.0.1:8501`) in `.streamlit/config.toml` ensuring 54ms instantaneous startup.

---

## [2.1.0] — 2026-11-25

### Added
- **Modified Z-Score with MAD Outlier Detection**: Implemented in `src/outliers.py` using Median Absolute Deviation based on IEEE Access (2024).
- **Multi-SNN (Sequential / Spatial Nearest Neighbor) Benchmark**: Added in `src/garda_upgraded.py` comparing against Reference Paper 2.
- **Benchmark Datasets**: Generated Intel Berkeley Lab IoT, Global Air Quality, and Customer Churn datasets in `data/`.

---

## [2.0.0] — 2026-11-05

### Added
- **Base GARDA Implementation**: Faithful recreation of Algorithm 1 from IEEE IoT-J 2026.
- **Proposed Upgraded GARDA+**: Adaptive Freedman-Diaconis granulation, sign-coupled Localized Density Reconstruction (LDR), and Shannon entropy error gating.
- **4-Dimensional Explainable Quality Score Engine** (`src/quality_score.py`).
- **Linear Undo & Reset Workflow Manager** (`src/cleaning.py`).
- **Initial 31 Pytest Unit Tests**.
