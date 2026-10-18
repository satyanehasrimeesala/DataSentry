# DataSentry — Intelligent Dataset Quality Investigator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests-Pytest](https://img.shields.io/badge/Tests-31%20Passed-brightgreen.svg)](tests/)

---

## 📌 1. Project Overview & Problem Statement

Modern machine learning models and analytical pipelines adhere strictly to the principle of *"Garbage In, Garbage Out"*. When datasets are compromised by missing values, duplicated entries, extreme statistical outliers, inconsistent text casing, or disguised formats, predictive performance decays, leading to catastrophic errors and unreliable business or engineering decisions.

**DataSentry** is a complete, 100% local, self-contained **Dataset Quality Investigation & Interactive Remediation Suite** developed specifically as a final-year Computer Science (Data Science) major project. 

It provides an end-to-end investigative workspace allowing analysts and developers to:
1. Ingest **CSV** and multi-sheet **Excel (XLSX)** datasets without transmitting any data externally.
2. Automatically diagnose **missingness, exact/subset duplicates, IQR/Z-score outliers, whitespace/casing anomalies, and data type discrepancies**.
3. Calculate an **explainable, transparent Data Quality Score (0 to 100)** broken down into **Completeness, Uniqueness, Validity, and Consistency**.
4. Interactively remediate defects using **safe statistical and machine-learning transformations** with full **Undo** and **Reset** protection.
5. Benchmark and execute our **Research Extension**: A faithful implementation of the IEEE Base Paper **GARDA** (*Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks*, IEEE IoT-J 2026) alongside our novel proposed **Upgraded GARDA+ Algorithm**.
6. Generate a full **Before-and-After Investigation Report** and export cleaned datasets in CSV, XLSX, and formatted TXT/JSON formats.

---

## 🎯 2. Project Objectives

- **Zero Cloud Dependence**: Entirely local offline computation. No API keys, no paid tokens, no logins, and zero remote data exposure.
- **Explainable Quality Assessment**: Formulate quality not as a black box, but as four defensible mathematical indicators.
- **Reversible Cleaning Workflows**: Guard against accidental data destruction using linear state-stack rollbacks and audit logs.
- **Academic Research Integration**: Bridge theoretical research and practical software by implementing and upgrading an IEEE journal algorithm for continuous missing value imputation.

---

## 🏗️ 3. Architecture & Modular Project Structure

DataSentry follows a clean, decoupled design pattern separating analytical logic from the Streamlit UI presentation layer:

```text
DataSentry/
│
├── app.py                      # Main Streamlit web application & interactive dashboard
├── requirements.txt            # Minimal required open-source dependencies
├── README.md                   # Complete architectural, setup, and viva guide
├── .gitignore                  # Git tracking rules
│
├── src/                        # Modular core analytical engine
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # CSV & Excel ingestion, encoding checks, duplicate header handling
│   ├── profiler.py             # Descriptive statistics, semantic type inference, cardinality
│   ├── missing_values.py       # Null analysis & imputation (Mean, Median, Mode, Constant, FFill, KNN)
│   ├── duplicates.py           # Exact & subset duplicate detection and removal
│   ├── outliers.py             # Interquartile Range (IQR) & Z-score detection & Winsorization
│   ├── consistency.py          # Whitespace, casing variations, suspicious types, custom validation rules
│   ├── quality_score.py        # Explainable 4-dimensional Data Quality Score engine (0-100)
│   ├── cleaning.py             # Reversible CleaningWorkflowManager (History stack, Undo, Reset)
│   ├── reporting.py            # Comparative Before-and-After reporting & binary file exporters
│   ├── garda_base.py           # Faithful implementation of IEEE Base Paper: GARDA (Algorithm 1)
│   └── garda_upgraded.py       # Proposed Upgraded GARDA+ (Adaptive Granulation, LDR & Entropy Gating)
│
├── tests/                      # Automated test suite (Pytest - 31 passing unit tests)
│   ├── __init__.py
│   ├── test_data_loader.py     # File ingestion, headers, empty file validation
│   ├── test_missing_values.py  # Detection, thresholds, imputation techniques
│   ├── test_duplicates.py      # Exact and subset duplicate checks
│   ├── test_outliers.py        # IQR and Z-score detection, capping, dropping
│   ├── test_consistency.py     # String normalization, casing, type conversion, custom rules
│   ├── test_quality_score.py   # Perfect, degraded, and empty dataset scoring
│   ├── test_cleaning.py        # Undo, reset-to-original, audit log state tracking
│   ├── test_reporting.py       # Comparison summaries, CSV/Excel/JSON export bytes
│   └── test_garda.py           # Base GARDA, Upgraded GARDA+, and synthetic benchmark validation
│
├── sample_data/                # Controlled sample datasets for demonstration
│   ├── sample_messy_dataset.csv # Multi-defect dataset (missing values, duplicates, outliers, typos)
│   └── iot_sensor_multivariate.csv # Correlated IoT sensor time-series (temp, humidity, pressure)
│
├── BASE PAPER.pdf              # IEEE IoT Journal (2026) GARDA Base Paper
├── Reference Paper - 1.pdf     # IEEE Access (2023) Multi-Spike SNN Paper
└── DataSentry Overview.txt     # High-level project specifications
```

---

## 🔬 4. Base Paper & Proposed Upgraded GARDA+ Algorithm

### A. The Base Paper: GARDA (IEEE IoT Journal, 2026)
*Reference: G. D'Aniello, M. Della Corte, R. Gaeta, and T.-P. Hong, "GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks," IEEE Internet of Things Journal, Vol. 13, No. 17, Sept. 2026.*

**Core Methodology**:
1. **Granulation**: Continuous features $x_i$ are discretized into interval-valued granules using K-Means clusters or Equal-Width intervals.
2. **Robust Association Rules (RAR)**: Mined over incomplete transactions $S$ with disabled set $Dis(X_p)$:
   $$\text{Sup}(X_p) = \frac{|\sigma(X_p)|}{|S| - |Dis(X_p)|}, \quad \text{Conf}(X_p \to Y_{p'}) = \frac{|\sigma(X_p \cup Y_{p'})|}{|\sigma(X_p)| - |Dis(Y_{p'}) \cap \sigma(X_p)|}$$
3. **Iterative Relaxation**: High-confidence mining pass followed by progressive relaxation of thresholds ($r_{\text{sup}}, r_{\text{conf}}$).
4. **Reconstruction & Hybrid Fallback**: Imputed granules are mapped to interval midpoints or centroids; remaining unfilled values are completed using Bayesian Ridge Regression.

### B. Limitations of the Base Paper Identified
1. **Static Flat Reconstruction**: Blindly taking the interval midpoint $(a+b)/2$ discards intra-granule trends, introducing a systematic variance of $O(\lambda^2 / 12)$.
2. **Rigid Binning**: Base GARDA relied on an exhaustive grid search across $\sigma/n$ fractions to pick cluster counts.
3. **Error Cascade**: The authors noted in Section III-D that relaxing support/confidence introduces weak, ambiguous rules that corrupt the data before traditional regression can act.

### C. Our Upgraded Algorithm: GARDA+ (Adaptive-GARDA)
DataSentry introduces three algorithmic enhancements:
1. **Adaptive Freedman-Diaconis Granulation (FD-KMeans)**: Dynamically scales cluster counts based on interquartile range ($h = 2 \cdot \text{IQR} \cdot n^{-1/3}$) with quantile centroid initialization to adapt to skewed sensor measurements.
2. **Confidence-Weighted Localized Density Reconstruction (LDR)**:
   Instead of a static midpoint, GARDA+ calculates the relative position $\alpha \in [0, 1]$ of the antecedent attribute within its own granule:
   $$\alpha = \frac{x_{\text{ant}} - \text{start}_{\text{ant}}}{\text{end}_{\text{ant}} - \text{start}_{\text{ant}}}$$
   Then reconstructs the continuous target:
   $$\hat{y} = \text{start}_{\text{con}} + \alpha \cdot (\text{end}_{\text{con}} - \text{start}_{\text{con}})$$
   This continuous coupling preserves fine-grained physical gradients and reduces RMSE.
3. **Information Entropy Gating (EG-Defense)**:
   Evaluates the posterior Shannon entropy of candidate rules:
   $$H(Y \mid X) = -\sum_{y} P(y \mid X) \log_2 P(y \mid X)$$
   Rules exceeding the entropy threshold ($\theta_H = 1.25$ bits) are suppressed, delegating ambiguous gaps directly to Bayesian Ridge Regression to prevent error propagation.

---

## 📊 5. Transparent Data Quality Score Formulation

DataSentry calculates an explainable Data Quality Score on a scale of **0 to 100**:

$$\text{Quality Score} = (0.35 \times C) + (0.25 \times U) + (0.20 \times V) + (0.20 \times K)$$

| Indicator | Weight | Mathematical Formula | What it Measures |
|---|:---:|---|---|
| **Completeness ($C$)** | 35% | $100 \times \left(1 - \frac{\text{Missing Cells}}{\text{Total Cells}}\right)$ | Presence of empty or null cells across the entire dataset. |
| **Uniqueness ($U$)** | 25% | $100 \times \left(1 - \frac{\text{Duplicate Rows}}{\text{Total Rows}}\right)$ | Proportion of distinct, non-redundant observation records. |
| **Validity ($V$)** | 20% | $100 \times \left(1 - \frac{\text{Outlier Points}}{\text{Total Numeric Cells}}\right)$ | Adherence to statistical boundaries computed via IQR ($1.5 \times$). |
| **Consistency ($K$)** | 20% | $100 \times \left(1 - \frac{\text{Anomalies}}{\text{Total String Cells}}\right)$ | Absence of whitespace padding, mixed casing, and unparseable dates. |

### Quality Grade Scale:
- **90 – 100**: `Excellent` (Green) — Production ready, minimal or no defects.
- **75 – 89**: `Good` (Blue) — Minor defects, highly suitable for exploratory analysis.
- **60 – 74**: `Moderate` (Yellow) — Noticeable data quality degradation; cleaning strongly recommended.
- **40 – 59**: `Poor` (Orange) — Substantial missingness, duplicates, or format corruptions.
- **< 40**: `Critical` (Red) — Severe data integrity loss; model training will suffer garbage-in-garbage-out.

---

## 💻 6. Installation & Launch Instructions (Windows)

### Prerequisites:
- Windows 10 or 11
- Python 3.10, 3.11, 3.12, 3.13, 3.14, or 3.15 installed

### Step 1: Clone or Navigate to the Project
Open **PowerShell** or **Command Prompt**:
```powershell
cd c:\Projects\DataSentry
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
pytest tests -v
```
All **31 unit tests** should pass in under 5 seconds.

### Step 5: Launch the Application
```powershell
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🖥️ 7. Feature Walkthrough & Demonstration Guide

### 1. Upload & Overview
- Click **Browse files** to upload any CSV or XLSX file.
- If an Excel file has multiple sheets, select your desired worksheet from the dropdown.
- Or click **Load 'Messy Business Data'** to test with built-in realistic dirty data.

### 2. Dataset Profiler
- View dynamic row previews (First N, Last N, Random Sample).
- Inspect inferred semantic types (Numeric, Categorical, Boolean, Datetime) alongside physical pandas dtypes.
- Review descriptive statistics: Mean, Std Dev, Min, Q1, Median, Q3, Max, IQR, Skewness, Zeros count, and Negatives count.

### 3. Missing Value Remediation
- View an interactive Plotly bar chart showing missing percentages across columns.
- Select from 10 strategies: **Mean, Median, Mode, Constant, FFill, BFill, KNN Imputation, Drop Rows, Drop Columns, Drop by % Threshold**.
- High-loss warning alerts appear if an action drops more than 15% of records.

### 4. Duplicate Detection
- Check exact row duplicates or select specific business primary keys (e.g. `customer_id`).
- Inspect sample duplicate groups side-by-side.
- Deduplicate by keeping the **First**, **Last**, or **Purging all duplicate instances**.

### 5. Outlier Detection & Winsorization
- Interquartile Range (IQR) detection with configurable multiplier ($k = 1.5$ standard, $k = 3.0$ extreme).
- View Plotly interactive box plots and summary bounds.
- Apply **Winsorization (Capping)** to clamp outliers to $[Q1 - 1.5 \cdot \text{IQR}, Q3 + 1.5 \cdot \text{IQR}]$, drop outlier rows, or convert them to nulls.

### 6. Consistency & Custom Validation Rules
- Detect and strip leading/trailing whitespace and collapse multiple spaces.
- Standardize casing (`lower`, `upper`, `title`, `capitalize`) to fix variations like `Sales`, `sales`, and `SALES`.
- Convert column types safely with automatic invalid coercion to NaN.
- Define custom business validation rules (e.g., `age in [0, 120]`, `department in allowed list`).

### 7. Explainable Quality Score
- View the overall 0–100 score meter, letter grade badge, and radar chart across dimensions.
- Review transparent deduction explanations detailing exactly which defects pulled the score down.

### 8. Research Lab (GARDA Base & Upgraded)
- Read the algorithmic summary comparing the IEEE 2026 Base Paper and our Upgraded GARDA+.
- **Run the Ground-Truth Benchmark Suite**: Injects synthetic MCAR or Consecutive Sensor Outage errors (5% to 50%) into complete data, then benchmarks Mean, Median, KNN, Bayesian Ridge, Base GARDA, and Upgraded GARDA+.
- Directly review RMSE, MAE, NRMSE, R², and Runtime charts.
- Use **Direct GARDA+ Imputation** to repair real missing values in your active dataset.

### 9. Before-and-After Report & Cleaned Export
- Inspect side-by-side metric improvements (Quality Score delta, Missing Cells resolved, Duplicates removed).
- Review the chronological audit log of all operations performed.
- One-click downloads:
  - 📥 **Cleaned CSV**
  - 📥 **Cleaned Excel (.xlsx)**
  - 📄 **Full Quality Investigation Report (.txt)**

---

## 🎓 8. College Review & Viva Presentation Guide

When presenting DataSentry to your university project evaluation panel, use the following structured answers:

### Q1: What is the primary purpose and USP of DataSentry?
> *"DataSentry is an intelligent, completely offline dataset quality investigation and remediation platform. Unlike black-box tools or mock dashboards, it calculates a mathematically explainable Data Quality Score (0–100) across Completeness, Uniqueness, Validity, and Consistency. It protects analysts with a reversible Undo stack and integrates an academic research extension based on the IEEE 2026 GARDA paper."*

### Q2: What is the IEEE Base Paper used in your project?
> *"Our base paper is 'GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks', published in the IEEE Internet of Things Journal (September 2026) by Giuseppe D'Aniello, Mario Della Corte, Rosario Gaeta, and Tzung-Pei Hong. It introduced continuous data granulation and Robust Association Rules (RAR) to impute missing values in correlated sensor streams without deep neural networks."*

### Q3: What is your UPGRADATION of the Base Paper?
> *"We identified three key limitations in the Base Paper and designed **GARDA+**:*
> 1. *Base GARDA required manual grid searching for cluster counts; GARDA+ introduces **Adaptive Freedman-Diaconis Granulation** to automatically size bins from feature variance.*
> 2. *Base GARDA reconstructed values as flat interval midpoints, which discarded intra-interval trends. GARDA+ introduces **Confidence-Weighted Localized Density Reconstruction (LDR)**, interpolating position based on the antecedent's continuous percentile position $\alpha \in [0, 1]$.*
> 3. *Base GARDA suffered from error propagation when relaxing thresholds. GARDA+ introduces **Information Entropy Gating** to suppress ambiguous high-entropy rules and delegate them to Bayesian Ridge Regression."*

### Q4: How is your Data Quality Score calculated?
> *"It is a weighted multi-attribute index:*
> $\text{Score} = 0.35 \times \text{Completeness} + 0.25 \times \text{Uniqueness} + 0.20 \times \text{Validity} + 0.20 \times \text{Consistency}$.
> *Each sub-indicator is bounded between 0 and 100 based on exact cell defect counts. It is fully explainable because the application displays every deduction and allows the analyst to see the score increase in real time as cleaning operations are applied."*

### Q5: How do you prevent accidental data loss?
> *"Through the `CleaningWorkflowManager` in `src/cleaning.py`. It holds an immutable copy of the raw uploaded data and manages a linear state stack. Any operation can be reverted via the **Undo** button, or the entire dataset can be restored via **Reset to Original**."*

---

## ⚖️ 9. Limitations & Future Scope

### Current Limitations:
- **Scalability**: Optimized for single-node in-memory processing up to ~1,000,000 rows (depending on client RAM).
- **Tabular Focus**: Designed specifically for structured tabular and sensor data, not unstructured image/audio blobs.
- **Rule Combinatorics**: Apriori-like candidate generation in Base GARDA is $O(n \cdot 2^k)$, though mitigated in GARDA+ by adaptive bin pruning.

### Future Enhancements:
- **Streaming Ingestion**: Implementing sliding-window adaptive updating for live MQTT/Kafka sensor feeds.
- **Fuzzy/Rough Granulation**: Exploring fuzzy membership functions to replace hard interval boundaries.
- **Automated Rule Synthesis**: Using statistical hypothesis tests to auto-suggest domain validation constraints.

---

## 📜 10. References

1. **Base Paper**: D'Aniello, G., Della Corte, M., Gaeta, R., & Hong, T.-P. (2026). *GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks*. **IEEE Internet of Things Journal**, 13(17), 38900–38914.
2. **Reference Paper 1**: Al-Jamali, N. A. S., Al-Saedi, I. R. K., Zarzoor, A. R., & Li, H. (2023). *A New Imputation Technique Based a Multi-Spike Neural Network to Handle Missing Data in the Internet of Things Network (IoT)*. **IEEE Access**, 11, 112841–112850.
3. Ragel, A., & Crémilleux, B. (1998). *Treatment of missing values for association rules*. In **PAKDD**, Springer, pp. 258–270.
4. Hong, T.-P., & Wu, C.-W. (2011). *Mining rules from an incomplete dataset with a high missing rate*. **Expert Systems with Applications**, 38(4), 3931–3936.
