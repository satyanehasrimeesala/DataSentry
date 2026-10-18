# DataSentry — Comprehensive Viva Voce Defense Guide
### 25 Technical Questions & Answers for College Major Project Review

---

## 🎯 CATEGORY 1: PROJECT MOTIVATION & HIGH-LEVEL ARCHITECTURE

### Q1: What is the primary objective of your project and what problem does it solve?
**Answer**:
> *"The objective of DataSentry is to provide an intelligent, 100% local, privacy-preserving web application for dataset quality investigation, transparent scoring, and reversible interactive cleaning. It addresses the 'Garbage In, Garbage Out' dilemma in machine learning by detecting missing values, duplicate rows, statistical outliers, formatting inconsistencies, and invalid data types without relying on external cloud APIs or exposing confidential datasets."*

### Q2: Why did you build a local application instead of using existing cloud platforms or SaaS tools?
**Answer**:
> *"Most commercial tools (like Google Cloud Dataprep or AWS Glue DataBrew) require paid enterprise subscriptions, internet access, and sending private organizational datasets to remote servers. This introduces data privacy risks, network latency, and vendor lock-in. DataSentry runs entirely offline on the client's machine using free, open-source Python libraries, ensuring zero cloud costs and zero data egress."*

### Q3: How is your project structured, and what architectural pattern did you follow?
**Answer**:
> *"We followed a decoupled Layered Model-View-Controller (MVC) architecture:*
> - *The Presentation Layer (`app.py`) is implemented in Streamlit and Plotly.*
> - *The Controller Layer (`src/cleaning.py`) manages state transitions, audit logging, and the linear undo stack via the `CleaningWorkflowManager` class.*
> - *The Model/Engine Layer (`src/`) contains independent, single-responsibility modules: `data_loader.py`, `profiler.py`, `missing_values.py`, `duplicates.py`, `outliers.py`, `consistency.py`, `quality_score.py`, `reporting.py`, `garda_base.py`, and `garda_upgraded.py`.*
> - *The Testing Layer (`tests/`) contains 31 automated unit tests written in pytest."*

---

## 📐 CATEGORY 2: DATA QUALITY SCORE & MATHEMATICAL FORMULATION

### Q4: How is your Data Quality Score formulated? Is it an arbitrary heuristic?
**Answer**:
> *"No, it is not an arbitrary heuristic. It is a mathematically bounded, explainable composite index between 0 and 100 calculated as:*
> $$\text{Score} = (0.35 \times \text{Completeness}) + (0.25 \times \text{Uniqueness}) + (0.20 \times \text{Validity}) + (0.20 \times \text{Consistency})$$
> *Each sub-indicator is strictly bounded between 0 and 100 based on exact cell and row defect counts. Every point deduction is documented and displayed with specific on-screen diagnostic explanations."*

### Q5: How do you calculate the Completeness dimension?
**Answer**:
> *"Completeness ($C$) is the ratio of valid, non-null cells across the entire dataset matrix:*
> $$C = 100 \times \left(1 - \frac{\text{Total Missing Cells}}{\text{Total Rows} \times \text{Total Columns}}\right)$$
> *It treats standard IEEE NaNs, nulls, empty strings (`""`), and whitespace-only strings as missing values."*

### Q6: How do you calculate the Validity dimension?
**Answer**:
> *"Validity ($V$) measures adherence to statistical boundaries. We compute the Interquartile Range (IQR = $Q_3 - Q_1$) across all numeric columns. Values falling outside $[Q_1 - 1.5 \cdot \text{IQR}, Q_3 + 1.5 \cdot \text{IQR}]$ are flagged as outliers:*
> $$V = 100 \times \left(1 - \frac{\text{Total Outlier Points}}{\text{Total Rows} \times \text{Total Numeric Columns}}\right)$$
> *If a dataset contains no numeric columns, Validity defaults safely to 100."*

### Q7: How do you calculate the Consistency dimension?
**Answer**:
> *"Consistency ($K$) measures text normalization and data type integrity over string/categorical columns:*
> $$K = 100 \times \left(1 - \frac{\text{Whitespace Errors} + \text{Casing Conflicts} + \text{Unparseable Dates}}{\text{Total Rows} \times \text{Total String Columns}}\right)$$
> *It flags leading/trailing whitespace, multiple consecutive spaces, casing discrepancies (e.g., `Sales` vs `sales`), and invalid date formats."*

---

## 🔬 CATEGORY 3: BASE PAPER DEEP DIVE (GARDA: IEEE IoT-J 2026)

### Q8: What is your Base Paper and who are the authors?
**Answer**:
> *"Our Base Paper is: 'GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks', published in the IEEE Internet of Things Journal (Vol. 13, No. 17, September 2026) by Giuseppe D'Aniello, Mario Della Corte, Rosario Gaeta, and Tzung-Pei Hong."*

### Q9: What was the main motivation behind the GARDA base paper?
**Answer**:
> *"In Internet of Things (IoT) sensor networks, devices operate with limited CPU, memory, and battery power. Deep learning imputation models (like recurrent neural networks or transformers) achieve good accuracy but are computationally prohibitive and consume excessive power. Traditional methods like mean or linear regression are lightweight but perform poorly at high missing rates. GARDA bridged this gap by combining Granular Computing (GrC) and association rules with a regression fallback, achieving high accuracy with low computational overhead."*

### Q10: How does GARDA handle continuous data using Granular Computing?
**Answer**:
> *"Traditional association-rule mining (like Apriori) works only on discrete categorical items. GARDA granulates continuous data into interval-valued sets using Equal-Width binning or K-Means clustering. An observation $x$ belongs to granule $I_j$ if $x \in [\text{start}_j, \text{end}_j]$. Mined rules then associate antecedent granules with consequent granules (e.g., `Temperature_G2 -> Humidity_G4`)."*

### Q11: What are Robust Association Rules (RAR) and why are they necessary?
**Answer**:
> *"Standard Apriori requires complete transactions and would drop rows with missing values, causing severe information loss. GARDA uses Robust Association Rules (RAR, proposed by Ragel & Crémilleux), which compute support and confidence over incomplete data by defining a 'disabled set' $Dis(X)$—the set of transactions where attributes in itemset $X$ are missing. This allows rules to be mined without dropping incomplete observations."*

### Q12: How does the hybrid architecture in Base GARDA work?
**Answer**:
> *"GARDA operates in three stages:*
> 1. *Granulation: Discretizes continuous attributes into interval granules.*
> 2. *GrC Association Rule Imputation: Uses mined high-confidence rules to fill missing intervals, followed by an iterative threshold relaxation pass ($r_{\text{sup}}, r_{\text{conf}}$).*
> 3. *Reconstruction & Hybrid Fallback: Maps granules back to continuous values using interval midpoints or centroids. Any cells that remain unresolved by rules are handed over to Bayesian Ridge Regression."*

---

## 🚀 CATEGORY 4: UPGRADED ALGORITHM NOVELTY (GARDA+)

### Q13: What specific limitations did you identify in the Base Paper that prompted your upgrade?
**Answer**:
> *"We identified three critical limitations in the Base Paper:*
> 1. *Static Flat Midpoint Reconstruction: Replacing an interval $[a, b]$ with its midpoint $\frac{a+b}{2}$ discards intra-granule trends, introducing a reconstruction variance of $\frac{\lambda^2}{12}$.*
> 2. *Rigid Cluster Counts: Base GARDA required an exhaustive manual grid search across arbitrary standard deviation fractions ($\sigma/2, \dots, \sigma/16$).*
> 3. *Weak Rule Error Propagation: In the iterative relaxation pass, accepting low-confidence rules injects noise that corrupts the data before the hybrid regression phase can act."*

### Q14: How does GARDA+'s Adaptive Granulation work?
**Answer**:
> *"GARDA+ eliminates manual grid searching by using the Freedman-Diaconis rule to determine bin width:*
> $$h = 2 \cdot \text{IQR} \cdot n^{-1/3}$$
> *From $h$, the optimal cluster count $k$ is calculated: $k = \max(4, \min(25, \lceil \frac{\text{Range}}{h} \rceil))$. We then seed K-Means cluster centers at empirical quantiles of the data rather than random points, ensuring fast convergence and preventing empty clusters."*

### Q15: How does Confidence-Weighted Localized Density Reconstruction (LDR) improve accuracy?
**Answer**:
> *"In Base GARDA, if rule $X \to Y$ fires, $Y$ is reconstructed as the static midpoint regardless of where $X$ fell inside its own interval. In GARDA+, we compute the relative percentile position $\alpha$ of the antecedent:*
> $$\alpha = \frac{x_{\text{ant}} - \text{start}_{\text{ant}}}{\text{end}_{\text{ant}} - \text{start}_{\text{ant}}} \in [0, 1]$$
> *We then interpolate the consequent value proportionally: $\hat{y} = \text{start}_{\text{con}} + \alpha \cdot (\text{end}_{\text{con}} - \text{start}_{\text{con}})$. This continuous coupling preserves natural physical gradients and slashes RMSE by up to 27%."*

### Q16: What is Information Entropy Gating (EG-Defense)?
**Answer**:
> *"To prevent weak, ambiguous rules from corrupting the dataset during threshold relaxation, GARDA+ evaluates the posterior Shannon conditional entropy:*
> $$H(Y \mid X) = -\sum_{y \in \mathcal{Y}} P(y \mid X) \log_2 P(y \mid X)$$
> *If $H(Y \mid X) > 1.25$ bits, the rule represents high uncertainty and is suppressed. Instead of making an inaccurate granular guess, the cell is safely delegated directly to the Bayesian Ridge Regressor."*

### Q17: What were the experimental findings comparing GARDA and GARDA+?
**Answer**:
> *"Under a consecutive sensor outage model with 20% missingness on multivariate environmental data:*
> - *Baseline Mean: RMSE = 3.842 | $R^2 = 0.00$*
> - *KNN Imputer: RMSE = 1.845 | $R^2 = 0.77$*
> - *Bayesian Ridge: RMSE = 1.109 | $R^2 = 0.92$*
> - *Base GARDA (Paper): RMSE = 0.529 | $R^2 = 0.98$*
> - *Proposed GARDA+ (Upgraded): RMSE = **0.384** | $R^2 = **0.99$***
> *GARDA+ achieved a **27.4% reduction in RMSE** over the Base Paper and executed in under 0.15 seconds."*

---

## 🛠️ CATEGORY 5: CLEANING WORKFLOW, SAFETY & TESTING

### Q18: How do you ensure that cleaning operations do not permanently destroy user data?
**Answer**:
> *"Through the `CleaningWorkflowManager` class in `src/cleaning.py`. When a dataset is loaded, we maintain an immutable raw copy in `_raw_df`. Every time an operation is confirmed, a deep copy of the DataFrame is pushed onto `_history_stack`. Clicking **Undo** pops the top state, immediately restoring the prior snapshot. Clicking **Reset** restores the original raw dataset."*

### Q19: What happens if a user attempts an operation that would delete substantial data?
**Answer**:
> *"DataSentry implements proactive safety warnings. For example, if dropping rows with missing values would eliminate more than 15% of the total dataset, a high-severity warning banner is displayed showing the exact row count and percentage loss before the user can execute it."*

### Q20: What outlier remediation techniques did you implement?
**Answer**:
> *"We implemented three strategies:*
> 1. *Winsorization (Capping): Values $< Q_1 - 1.5 \cdot \text{IQR}$ are clamped to the lower bound, and values $> Q_3 + 1.5 \cdot \text{IQR}$ are clamped to the upper bound. This preserves sample size while mitigating extreme variance.*
> 2. *Dropping Rows: Drops only rows containing detected outliers.*
> 3. *Set to Null: Converts outliers into NaNs so they can be handled by the missing value imputation engine."*

### Q21: How does DataSentry handle Excel files with multiple sheets?
**Answer**:
> *"Using `get_excel_sheet_names()` in `src/data_loader.py` via openpyxl. It inspects the workbook structure without loading the full data. If multiple sheets exist, Streamlit dynamically renders a dropdown allowing the user to select which worksheet to ingest."*

### Q22: How did you test your codebase?
**Answer**:
> *"We built an automated test suite with Pytest consisting of 31 unit tests across 9 test files covering data ingestion, empty files, duplicate headers, missing imputation, deduplication, IQR outliers, text normalization, quality scoring, undo behavior, and both GARDA algorithms. All 31 tests pass in 3.24 seconds with a 100% pass rate."*

---

## 🔮 CATEGORY 6: EDGE CASES & FUTURE SCOPE

### Q23: How does DataSentry handle edge cases like completely empty columns or constant columns?
**Answer**:
> *"Our modules feature defensive safety guards:*
> - *If a column is 100% NaN, statistical imputation skips it with an informative warning rather than crashing with a division by zero.*
> - *If a column has identical values (IQR = 0), `detect_outliers_iqr` returns a `constant_or_zero_iqr` status rather than calculating invalid bounds.*
> - *If duplicate headers exist in a CSV (e.g. two columns named `col`), `data_loader.py` automatically detects and renames them to `col_1`, `col_2` to avoid silent overwriting."*

### Q24: What are the current limitations of your implementation?
**Answer**:
> *"DataSentry is designed for single-node in-memory processing and works best on datasets up to approximately 1,000,000 rows. Extremely large datasets exceeding available physical RAM would require distributed frameworks like Dask or Apache Spark. Furthermore, GARDA operates over batch time-windows rather than infinite streaming topologies."*

### Q25: How would you extend this project in the future?
**Answer**:
> *"Future extensions include:*
> 1. *Streaming Data Ingestion: Integrating sliding-window adaptive updating for live MQTT/Kafka IoT data streams.*
> 2. *Fuzzy & Rough Granulation: Exploring fuzzy membership functions to soften hard interval boundaries.*
> 3. *Automated Contract Generation: Using statistical tests to automatically suggest and export Great Expectations or JSON-schema validation contracts."*
