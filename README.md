# DataSentry — Intelligent Dataset Quality Investigator & Research Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests-Pytest](https://img.shields.io/badge/Tests-39%20Passed-brightgreen.svg)](tests/)
[![Architecture-Decoupled](https://img.shields.io/badge/Architecture-Modular%20%26%20Decoupled-9cf.svg)](src/)

---

## 📌 1. Project Overview & Problem Statement

Modern machine learning systems and analytical pipelines adhere strictly to the principle of *"Garbage In, Garbage Out"*. Incomplete records, duplicated rows, undetected statistical outliers, and inconsistent casing introduce systematic error variance into downstream algorithms.

**DataSentry** is an end-to-end, completely local and offline **Dataset Quality Investigation & Interactive Remediation Suite** developed as a major Computer Science (Data Science) project.

### Core Distinctions:
1. **100% Offline & Private**: Zero cloud API dependencies, zero external tokens, zero logins, and zero remote telemetry.
2. **Explainable 4-Dimensional Quality Score (0–100)**: Formulates quality transparently as:
   $$\text{Score} = 0.35 \cdot \text{Completeness} + 0.25 \cdot \text{Uniqueness} + 0.20 \cdot \text{Validity} + 0.20 \cdot \text{Consistency}$$
3. **Reversible Remediation Studio**: Manages state with an immutable raw copy and a linear Undo/Reset stack.
4. **IEEE Research Extension**: Implements the base paper algorithm **GARDA** (*IEEE Internet of Things Journal, Sept 2026*) alongside our novel proposed algorithm **GARDA+**, achieving **42.36% lower imputation RMSE** on multivariate sensor networks.
5. **Multi-Format Export**: Generates Cleaned CSV, Cleaned Excel (.xlsx), Text Audit Logs (.txt), and printable **Executive Summary HTML/PDF** reports.

---

## 🏗️ 2. Architectural Design & Separation of Concerns

DataSentry strictly isolates mathematical calculation engines from the user interface presentation layer:

```text
DataSentry/
├── app.py                      # Modernized Streamlit Dashboard (UI Presentation Layer)
├── requirements.txt            # Pinned, tested dependencies for Python 3.10+
├── README.md                   # Comprehensive architectural guide & viva defense notes
├── CHANGELOG.md                # Semantic version history and release notes
├── run_app.bat                 # One-click Windows application launcher
├── run_tests.bat               # One-click automated Pytest runner
│
├── src/                        # Modular Analytical & Research Engines
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Centralized mathematical weights, thresholds, and hyperparameters
│   ├── data_loader.py          # CSV & Excel ingestion, encoding fallback, duplicate header fixing
│   ├── profiler.py             # Semantic type inference, distribution moments, zero/negative metrics
│   ├── missing_values.py       # Diagnostics and 10 imputation strategies (Mean, Median, Mode, KNN, etc.)
│   ├── duplicates.py           # Exact and subset key-based deduplication
│   ├── outliers.py             # IQR, Z-Score, and Modified Z-Score with MAD (IEEE Access 2024)
│   ├── consistency.py          # Casing harmonization, whitespace normalization, custom rule engine
│   ├── quality_score.py        # Explainable 4D score generator with penalty breakdown
│   ├── cleaning.py             # CleaningWorkflowManager with immutable raw copy & Undo stack
│   ├── reporting.py            # Comparative before/after metrics, CSV/XLSX/HTML/TXT exporters
│   ├── garda_base.py           # Faithful IEEE GARDA base algorithm (Algorithm 1)
│   ├── garda_upgraded.py       # Proposed GARDA+ (Adaptive FD-Granulation, LDR, Entropy Defense)
│   │
│   └── ui/                     # UI Presentation Components
│       ├── __init__.py         # UI package exports
│       └── theme.py            # Custom CSS design system, grade color mapping, and Plotly templates
│
├── data/                       # Built-in Industry Benchmark Datasets
│   ├── intel_lab_iot_sensors.csv           # Berkeley Intel Lab sensor telemetry (1,005 rows)
│   ├── global_air_quality_cities.csv       # EPA air pollution network across 8 metropolises (803 rows)
│   └── ecommerce_customer_churn_messy.csv  # Customer behavioral transactions with messy strings (604 rows)
│
├── tests/                      # Automated Verification & Smoke Suite (39 Tests Passed)
│   ├── test_cleaning.py        # Undo/Reset workflow tests
│   ├── test_consistency.py     # String normalization & custom validation tests
│   ├── test_data_loader.py     # CSV/XLSX ingestion and encoding tests
│   ├── test_duplicates.py      # Deduplication strategy tests
│   ├── test_garda.py           # GARDA, GARDA+, and Multi-SNN benchmark tests
│   ├── test_missing_values.py  # Imputation strategy tests
│   ├── test_outliers.py        # IQR, Z-Score, and Modified Z-Score MAD tests
│   ├── test_profiler.py        # Statistical moments and dataset profiling tests
│   ├── test_quality_score.py   # Mathematical scoring boundary tests
│   ├── test_reporting.py       # Exporter and comparison report tests
│   └── test_ui_smoke.py        # Streamlit AppTest headless smoke tests
```

---

## 🔬 3. Research Extension: Base GARDA vs. Proposed GARDA+

### Base Paper Reference
> *"GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks"*  
> *(IEEE Internet of Things Journal, Vol. 13, No. 17, September 2026)*

### Limitations in the Base Paper
1. **Quantization Variance**: Assigns imputed values to the static midpoint $(a+b)/2$ of matched intervals, discarding intra-granule trends ($O(\lambda^2/12)$ variance).
2. **Fixed Cluster Sizing**: Requires empirical grid search for cluster count $k$; fails on skewed or multimodal distributions.
3. **Threshold Relaxation Cascade**: Relaxing support and confidence introduces ambiguous, low-quality association rules.

### Our Proposed Innovation: **GARDA+**
1. **Adaptive Freedman-Diaconis Granulation**: Automatically sizes bins based on sample variance and size:
   $$h = 2 \cdot \text{IQR} \cdot n^{-1/3}$$
2. **Correlation Sign-Coupled Localized Density Reconstruction (LDR)**:
   $$\hat{y} = b_{\text{low}} + \alpha_{\text{coupled}} \cdot (b_{\text{high}} - b_{\text{low}})$$
   $$\alpha_{\text{coupled}} = \begin{cases} \alpha, & \text{if } \text{corr}(X, Y) \ge 0 \\ 1 - \alpha, & \text{if } \text{corr}(X, Y) < 0 \end{cases}$$
3. **Shannon Information Entropy Gating (EG-Defense)**: Prunes ambiguous rules where conditional entropy $H(Y \mid X) > \theta_H$ before delegating to Bayesian Ridge regression.

### Benchmark Empirical Comparison (Intel Lab Sensor Telemetry):
| Imputation Method | Error Scenario | RMSE | NRMSE | Local & Offline? |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Mean)** | Consecutive Outage (25%) | 4.966 | 0.248 | ✅ Yes |
| **KNN Imputation ($k=5$)** | Consecutive Outage (25%) | 3.808 | 0.190 | ✅ Yes |
| **Multi-SNN (Paper 2)** | Consecutive Outage (25%) | 2.842 | 0.142 | ✅ Yes |
| **Base GARDA (Paper 1)** | Consecutive Outage (25%) | 2.072 | 0.103 | ✅ Yes |
| **Proposed GARDA+ (Ours)** | Consecutive Outage (25%) | **1.194** | **0.059** | ✅ **Yes (42.36% Better)** |

---

## 🚀 4. How to Run Locally

### Quick Launch (Windows):
Double-click `run_app.bat` in the project root.

### Manual Launch via Terminal:
```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Launch Streamlit Dashboard
streamlit run app.py
```
Open your browser at: **`http://127.0.0.1:8501`**

---

## 🧪 5. Automated Testing & Verification

Run the complete test suite across all 39 unit and headless UI smoke tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
To check code coverage:
```powershell
.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing
```

---

## 💼 6. Resume Bullet Points (Ready for Applications)

**DataSentry — Intelligent Dataset Quality Investigator & Research Suite**  
*Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Pytest, Git*
- Architected and built an end-to-end, 100% offline dataset quality investigation and remediation platform processing CSV and multi-sheet Excel files with zero cloud API dependencies.
- Engineered an explainable 4-dimensional **Data Quality Score (0–100)** evaluating Completeness (35%), Uniqueness (25%), Validity (20%), and Consistency (20%) with dynamic radar chart breakdowns and deduction auditing.
- Implemented robust defect remediation engines covering 10 missing-value strategies (KNN, Mode, FFill/BFill), exact/subset deduplication, string harmonization, and **Modified Z-Score Outlier Detection using Median Absolute Deviation (MAD)** based on IEEE Access (2024).
- Researched and upgraded the IEEE IoT Journal (2026) **GARDA** algorithm (**GARDA+**), integrating Freedman-Diaconis adaptive granulation, correlation sign-coupled localized density reconstruction (LDR), and Shannon entropy gating, **reducing imputation RMSE by 42.36%** over the base paper.
- Developed a non-destructive state management workflow engine featuring immutable raw storage, linear Undo/Reset stacks, comparative delta metrics, and automated multi-format export (Cleaned CSV/XLSX, Text Audit, and Printable Executive HTML/PDF Reports).

---

## 🎓 7. University Viva & Review Defense Notes

### Q1: What makes DataSentry different from open-source profiling libraries?
> *"Generic profilers produce static, read-only summaries. DataSentry is an active investigative workspace that couples diagnostics with an interactive remediation studio, a non-destructive linear Undo/Reset state engine, an explainable 4D score, and an academic research extension comparing state-of-the-art IEEE imputation algorithms."*

### Q2: Why is the Modified Z-Score with MAD superior to the standard Z-Score?
> *"Standard Z-score relies on the sample mean $\mu$ and standard deviation $\sigma$. Extreme outliers heavily inflate $\sigma$ and skew $\mu$, masking genuine defects. Modified Z-Score replaces mean with the median and replaces standard deviation with Median Absolute Deviation (MAD), making it robust to extreme contamination."*

### Q3: How does GARDA+ prevent error cascade?
> *"When support and confidence thresholds are relaxed, traditional association rule mining can generate high-entropy rules where an antecedent maps unpredictably to diverse consequents. GARDA+ computes the Shannon conditional entropy $H(Y \mid X)$ and suppresses ambiguous rules ($\theta_H > 1.25$), falling back to regularized Bayesian Ridge regression."*

---

## 📜 8. License & Academic Integrity
Licensed under the [MIT License](LICENSE). Built for academic, research, and data science engineering portfolios.
