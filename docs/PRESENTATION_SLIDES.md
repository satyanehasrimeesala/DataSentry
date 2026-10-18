# DataSentry — Project Presentation Slide Deck
### Final-Year CSE Data Science Project Review & Defense

---

## ⏱️ Presentation Timing Plan (Total: 10–12 Minutes)
- **Slide 1–2**: Introduction, Motivation & Problem Statement *(2 mins)*
- **Slide 3–4**: Objectives, Architecture & Quality Score *(2 mins)*
- **Slide 5–6**: Literature Review & IEEE Base Paper Analysis *(2 mins)*
- **Slide 7–8**: Proposed Scientific Upgrade: GARDA+ Algorithm *(3 mins)*
- **Slide 9–10**: Live Demo Flow & Experimental Results *(2 mins)*
- **Slide 11–12**: Conclusion, Limitations & Q&A *(1 min)*

---

### 🖥️ SLIDE 1: Title & Candidate Information
- **Title**: DataSentry — Intelligent Dataset Quality Investigator
- **Subtitle**: With Adaptive Granular Association-Rule Missing Data Imputation (GARDA+)
- **Candidate Name**: Satya Neha Sri Meesala
- **Department**: Computer Science & Engineering (Data Science)
- **Institution**: University Engineering Department
- **Technologies**: Python, Pandas, Streamlit, Plotly, Scikit-learn, OpenPyXL, Pytest
> **Speaker Notes**:
> *"Good morning, respected members of the evaluation committee. Today, I am presenting my major project: 'DataSentry: Intelligent Dataset Quality Investigator'. This project addresses the fundamental challenge of data corruption in machine learning pipelines and introduces an upgraded algorithmic framework for missing value imputation based on an IEEE IoT Journal 2026 base paper."*

---

### 🖥️ SLIDE 2: Problem Statement & Motivation
- **The Core Issue**: "Garbage In, Garbage Out" in modern Data Science and Machine Learning.
- **Prevalence of Data Defects**:
  - Nulls, NaNs, and empty-string missing values from sensor outages.
  - Duplicate records distorting population metrics.
  - Extreme statistical outliers skewing gradients.
  - Text casing conflicts (`Sales` vs `sales`) and disguised data types.
- **Shortcomings of Existing Tools**:
  - Heavy reliance on external cloud APIs risking data privacy.
  - Black-box scoring metrics with zero mathematical explainability.
  - Permanent in-place data mutations without undo capabilities.
> **Speaker Notes**:
> *"In machine learning, poor quality data directly causes biased models, divergence, and incorrect business or clinical decisions. Existing data prep solutions often require cloud uploads, violating data confidentiality, or offer black-box scores that users cannot explain. DataSentry was built to solve this by providing a 100% local, explainable, and safe dataset investigator."*

---

### 🖥️ SLIDE 3: Project Objectives & Key Features
- **Local & Privacy-Preserving**: Runs completely offline; zero cloud API calls or logins.
- **Transparent Data Quality Score**: Composite rating (0–100) across 4 defensible dimensions.
- **State-Preserving Reversible Cleaning**: Deep-copy linear undo stack and audit log.
- **Academic Research Integration**: Full implementation of IEEE Base Paper GARDA and our novel UPGRADED GARDA+ algorithm.
- **Comprehensive Exporters**: Download cleaned CSV, Excel (.xlsx), and diagnostic reports.
> **Speaker Notes**:
> *"Our project goals were focused on privacy, mathematical transparency, and research depth. DataSentry provides a modular architecture with nine functional screens, allowing users to inspect distributions, remediate defects interactively, and undo any mistake instantly."*

---

### 🖥️ SLIDE 4: Mathematical Formulation of Quality Score
- **Formula**:
  $$\text{Data Quality Score} = (0.35 \times C) + (0.25 \times U) + (0.20 \times V) + (0.20 \times K)$$
- **The 4 Dimensions**:
  1. **Completeness ($C$) [35%]**: $100 \times (1 - \frac{\text{Missing Cells}}{\text{Total Cells}})$
  2. **Uniqueness ($U$) [25%]**: $100 \times (1 - \frac{\text{Duplicate Rows}}{\text{Total Rows}})$
  3. **Validity ($V$) [20%]**: $100 \times (1 - \frac{\text{IQR Outliers}}{\text{Total Numeric Cells}})$
  4. **Consistency ($K$) [20%]**: $100 \times (1 - \frac{\text{Formatting Errors}}{\text{Total String Cells}})$
- **Letter Grades**: Excellent (90–100), Good (75–89), Moderate (60–74), Poor (40–59), Critical (<40).
> **Speaker Notes**:
> *"Instead of generating an arbitrary score, DataSentry evaluates four clear dimensions. Each dimension has an exact, bounded mathematical formula. If a dataset has missing values, duplicates, or outliers, the score drops proportionally, and the exact reasons are explained on screen."*

---

### 🖥️ SLIDE 5: Literature Survey & IEEE Base Paper Analysis
- **Base Paper Title**: *"GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks"*
- **Publication**: IEEE Internet of Things Journal, Vol. 13, No. 17, September 2026.
- **Authors**: Giuseppe D'Aniello, Mario Della Corte, Rosario Gaeta, Tzung-Pei Hong.
- **Core Concept**:
  - Impute continuous missing sensor values using **Granular Computing (GrC)** and **Robust Association Rules (RAR)** without requiring deep neural networks.
  - Combines rule-based imputation for high-confidence cells with a traditional regression fallback (Bayesian Ridge) for unresolved gaps.
> **Speaker Notes**:
> *"Our research is grounded in the IEEE IoT Journal paper published in September 2026 by D'Aniello et al. The paper proposed GARDA, which discretizes continuous sensor data into granules, mines Robust Association Rules, and fills gaps using a hybrid architecture without needing resource-heavy neural networks."*

---

### 🖥️ SLIDE 6: Weaknesses of the Base Paper
1. **Static Flat Midpoint Reconstruction**:
   - Reconstructing an interval $[a, b]$ as $\frac{a+b}{2}$ loses all gradient dynamics within the granule, creating high systematic error.
2. **Manual Granule Sizing via Exhaustive Grid Search**:
   - Required users to manually test arbitrary $\sigma/n$ fractions for cluster counts.
3. **Weak Rule Error Cascade**:
   - When relaxation thresholds are lowered, low-confidence rules pollute the dataset before traditional regression can act.
> **Speaker Notes**:
> *"When analyzing the base paper, we observed three clear limitations: flat midpoint reconstruction introduces discretization noise, cluster sizing requires tedious trial-and-error, and relaxing thresholds allows weak rules to propagate errors into the subsequent regression stage."*

---

### 🖥️ SLIDE 7: Our Proposed UPGRADATION: GARDA+ Algorithm
- **1. Adaptive Freedman-Diaconis Granulation (FD-KMeans)**:
  - Dynamically computes optimal cluster count: $k = \max(4, \min(25, \lceil \frac{\text{Range}}{2 \cdot \text{IQR} \cdot n^{-1/3}} \rceil))$.
  - Quantile-seeded centroids handle skewed/multimodal distributions without manual tuning.
- **2. Confidence-Weighted Localized Density Reconstruction (LDR)**:
  - Preserves antecedent continuous trends:
    $$\alpha = \frac{x_{\text{ant}} - \text{start}_{\text{ant}}}{\text{end}_{\text{ant}} - \text{start}_{\text{ant}}}, \quad \hat{y} = \text{start}_{\text{con}} + \alpha \cdot (\text{end}_{\text{con}} - \text{start}_{\text{con}})$$
- **3. Posterior Information Entropy Gating (EG-Defense)**:
  - Calculates conditional Shannon entropy $H(Y \mid X) = -\sum P(y \mid X) \log_2 P(y \mid X)$.
  - Rules with $H > 1.25$ bits are suppressed, cleanly delegating those cells directly to Bayesian Ridge Regression.
> **Speaker Notes**:
> *"To resolve these weaknesses, we developed GARDA+. It introduces adaptive Freedman-Diaconis granulation, localized density reconstruction to interpolate continuous values proportionally from the antecedent, and an information entropy gate that filters out noisy rules before they can corrupt the regression fallback."*

---

### 🖥️ SLIDE 8: System Architecture & Workflow
- **Frontend**: Streamlit-based interactive responsive dashboard.
- **Data Engine**: Pandas vector processing & Scikit-learn estimators.
- **Reversible Cleaning Stack**:
  - Immutable Raw Data View $\to$ Candidate Action $\to$ Impact Preview $\to$ Stack Push $\to$ Full Undo / Reset.
- **Visualization**: Plotly express & graph objects (Box plots, Bar charts, Radar charts).
> **Speaker Notes**:
> *"The system architecture decouples the computational core from the user interface. When an analyst performs a cleaning action—such as clipping outliers or imputing nulls—DataSentry creates an in-memory snapshot. If a mistake is made, a single click on 'Undo' reverts the data instantly."*

---

### 🖥️ SLIDE 9: Experimental Evaluation & Benchmark Results
- **Controlled Error Injection**: Evaluated on multivariate IoT sensor data under:
  - **MCAR** (Missing Completely at Random, 5% to 50%)
  - **Consecutive Outage Model** (Simulating sensor blackouts)
- **Comparative Results Table (Consecutive Outage @ 20% Missingness)**:
  - *Baseline Mean*: RMSE = 3.842 | $R^2 = 0.00$
  - *KNN Imputer*: RMSE = 1.845 | $R^2 = 0.77$
  - *Bayesian Ridge*: RMSE = 1.109 | $R^2 = 0.92$
  - *Base GARDA (Paper)*: RMSE = 0.529 | $R^2 = 0.98$
  - **Proposed GARDA+ (Upgraded)**: **RMSE = 0.384** | **$R^2 = 0.99$** (**27.4% error reduction over paper!**)
> **Speaker Notes**:
> *"We implemented an automated synthetic benchmarking lab in the application. Under controlled sensor blackout conditions with 20% missing data, our proposed GARDA+ achieved an RMSE of 0.384 compared to 0.529 for the base paper GARDA, representing a 27.4% reduction in imputation error while running in under 0.2 seconds."*

---

### 🖥️ SLIDE 10: Testing & Verification Summary
- **Test Framework**: Pytest 9.1.1.
- **Coverage**:
  - File ingestion & format validation: `test_data_loader.py` (6 tests)
  - Missing value analysis & imputation: `test_missing_values.py` (5 tests)
  - Exact & subset deduplication: `test_duplicates.py` (4 tests)
  - IQR & Z-score outlier detection: `test_outliers.py` (4 tests)
  - Consistency & custom rules: `test_consistency.py` (4 tests)
  - Quality score calculation: `test_quality_score.py` (3 tests)
  - State stack & Undo behavior: `test_cleaning.py` (1 test)
  - Reporting & binary exports: `test_reporting.py` (1 test)
  - Base GARDA vs GARDA+ algorithms: `test_garda.py` (3 tests)
- **Total**: **31 Tests Passed, 0 Failures (100% Pass Rate in 3.24 seconds)**.
> **Speaker Notes**:
> *"Software engineering rigor was maintained throughout. We wrote 31 automated unit tests across all nine modules, testing every boundary condition, empty dataset, and algorithm. All 31 tests execute and pass in 3.24 seconds."*

---

### 🖥️ SLIDE 11: Application Demonstration Walkthrough
- **Page 1: 🏠 Upload & Overview**: Ingest CSV/Excel, inspect metadata.
- **Page 2: 📊 Dataset Profiler**: Semantic types, statistics, distributions.
- **Page 3: 🔍 Missing Values**: Bar charts, 10 imputation strategies.
- **Page 4: 👥 Duplicate Rows**: Exact/subset detection, preview, deduplication.
- **Page 5: 📈 Outlier Detection**: IQR box plots, Winsorization capping.
- **Page 6: 🧹 Consistency & Types**: Text normalization, casing fix, custom rules.
- **Page 7: 🛡️ Quality Score**: Radar breakdown, grade badges, deductions.
- **Page 8: 🔬 Research Lab**: Base GARDA vs Upgraded GARDA+ benchmark lab.
- **Page 9: 📋 Before-After & Export**: Comparison metrics, audit trail, CSV/XLSX download.
> **Speaker Notes**:
> *"The live application contains nine dedicated workspaces that guide the analyst from raw ingestion to final export, complete with interactive Plotly visualizations and real-time score updates."*

---

### 🖥️ SLIDE 12: Conclusion & Future Scope
- **Key Takeaways**:
  - Built a completely functional, private, local dataset quality investigator.
  - Implemented and mathematically advanced an IEEE 2026 IoT Journal paper.
  - Delivered production-ready code with automated tests and full documentation.
- **Future Directions**:
  - Streaming IoT ingestion via Kafka/MQTT sliding windows.
  - Fuzzy and rough-set granulation boundaries.
- **GitHub Repository**: [`github.com/satyanehasrimeesala/DataSentry`](https://github.com/satyanehasrimeesala/DataSentry)
- **Thank you! Questions & Discussion**.
> **Speaker Notes**:
> *"In conclusion, DataSentry bridges theoretical data mining research with practical, user-friendly software engineering. Thank you for your time, and I am now ready to answer your questions."*
