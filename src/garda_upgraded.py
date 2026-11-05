"""
garda_upgraded.py - Upgraded Granular Association-Rule Imputation: GARDA+.

Enhancements over Base Paper GARDA (D'Aniello et al., 2026):
1. Adaptive Granulation (FD-KMeans):
   - Computes optimal per-feature granule count using Freedman-Diaconis rule.
   - Quantile-seeded K-Means centroids to handle skewed/multimodal distributions.
2. Confidence-Weighted Localized Density Reconstruction (LDR):
   - Replaces static flat midpoints/centroids with intra-granule antecedent relative position interpolation:
     alpha = (x_ant - start_ant) / (end_ant - start_ant)
     y_hat = start_con + alpha * (end_con - start_con)
   - Preserves continuous trends and slashes RMSE.
3. Information Entropy Gating (EG-Fallback):
   - Calculates conditional rule entropy H(Y | X) = - sum(P * log2(P)).
   - Suppresses ambiguous, high-entropy rules to prevent error cascade into the hybrid regressor.
4. Robust Benchmark Engine:
   - Synthetic error injection (MCAR and Consecutive Sensor Outage).
   - Direct side-by-side comparison with Mean, Median, KNN, Bayesian Ridge, and Base GARDA.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import BayesianRidge
from sklearn.impute import KNNImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


class UpgradedGranule:
    """Enhanced interval granule with local density tracking and finite domain awareness."""
    def __init__(self, attribute: str, index: int, start: float, end: float, centroid: float, values_sample: np.ndarray, vmin: float, vmax: float):
        self.attribute = attribute
        self.index = index
        self.start = float(start)
        self.end = float(end)
        self.centroid = float(centroid)
        self.label = f"{attribute}_G{index}"
        self.sample_size = len(values_sample)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.eff_start = vmin if np.isinf(start) else float(start)
        self.eff_end = vmax if np.isinf(end) else float(end)
        self.density_median = float(np.median(values_sample)) if len(values_sample) > 0 else self.centroid

    def contains(self, value: float) -> bool:
        if np.isnan(value):
            return False
        return self.start <= value <= self.end

    def get_relative_position(self, value: float) -> float:
        """Returns relative position alpha in [0, 1] within granule interval using effective boundaries."""
        width = self.eff_end - self.eff_start
        if width <= 1e-9:
            return 0.5
        val_clamped = max(self.eff_start, min(self.eff_end, value))
        return max(0.01, min(0.99, (val_clamped - self.eff_start) / width))

    def reconstruct_from_alpha(self, alpha: float, sign: float = 1.0) -> float:
        """Interpolate continuous value using antecedent relative position alpha and inter-feature correlation sign."""
        eff_alpha = alpha if sign >= 0 else (1.0 - alpha)
        eff_alpha = max(0.01, min(0.99, eff_alpha))
        width = self.eff_end - self.eff_start
        return self.eff_start + (eff_alpha * width)

    def __repr__(self) -> str:
        return f"{self.label}[{self.start:.2f}, {self.end:.2f}]"


class UpgradedGardaImputer:
    """
    GARDA+ (Adaptive Granular Association Rule Imputer with Entropy Gating).
    """

    def __init__(
        self,
        entropy_threshold: float = 1.25,
        init_sup: float = 0.40,
        init_conf: float = 0.75,
        num_max_iter: int = 8,
        r_sup: float = 10.0,
        r_conf: float = 1.05
    ):
        self.entropy_threshold = entropy_threshold
        self.init_sup = init_sup
        self.init_conf = init_conf
        self.num_max_iter = num_max_iter
        self.r_sup = r_sup
        self.r_conf = r_conf
        self.granules_per_attr: Dict[str, List[UpgradedGranule]] = {}
        self.optimal_k_per_attr: Dict[str, int] = {}
        self.corr_matrix: Dict[str, Dict[str, float]] = {}

    def _determine_optimal_k(self, values: np.ndarray) -> int:
        """Freedman-Diaconis optimal bin estimator bounded for IoT sensor clusters."""
        n = len(values)
        if n < 8:
            return 4
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            h = 2.0 * iqr * (n ** (-1.0 / 3.0))
            data_range = np.max(values) - np.min(values)
            if h > 0:
                k_opt = int(np.ceil(data_range / h))
                return max(6, min(16, k_opt * 2))
        return 8

    def _build_adaptive_granules(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Create adaptive quantile-seeded granules for each continuous feature."""
        self.granules_per_attr.clear()
        self.optimal_k_per_attr.clear()
        self.corr_matrix = df[numeric_cols].corr().fillna(0).to_dict()

        for col in numeric_cols:
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue

            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            k = self._determine_optimal_k(vals)
            k = min(k, len(np.unique(vals)))
            self.optimal_k_per_attr[col] = k

            # Seed K-Means using quantiles for distribution adaptation
            quantiles = np.linspace(0.05, 0.95, k)
            init_centers = np.percentile(vals, quantiles * 100).reshape(-1, 1)

            km = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=42)
            labels = km.fit_predict(vals.reshape(-1, 1))
            centroids = km.cluster_centers_.flatten()

            raw_granules = []
            for idx in range(k):
                cluster_vals = vals[labels == idx]
                c_val = float(centroids[idx])
                if len(cluster_vals) > 0:
                    start = float(np.min(cluster_vals))
                    end = float(np.max(cluster_vals))
                else:
                    start = c_val - 0.05
                    end = c_val + 0.05
                raw_granules.append({
                    "idx": idx,
                    "start": start,
                    "end": end,
                    "centroid": c_val,
                    "vals": cluster_vals
                })

            raw_granules.sort(key=lambda g: g["centroid"])

            # Establish continuous boundaries without gaps
            granules = []
            for i in range(len(raw_granules)):
                cur = raw_granules[i]
                if i == 0:
                    g_start = -float("inf")
                else:
                    prev = raw_granules[i - 1]
                    g_start = (prev["end"] + cur["start"]) / 2.0

                if i == len(raw_granules) - 1:
                    g_end = float("inf")
                else:
                    nxt = raw_granules[i + 1]
                    g_end = (cur["end"] + nxt["start"]) / 2.0

                granules.append(
                    UpgradedGranule(col, cur["idx"], g_start, g_end, cur["centroid"], cur["vals"], vmin, vmax)
                )

            self.granules_per_attr[col] = granules

    def _discretize_row(self, row: pd.Series, numeric_cols: List[str]) -> Dict[str, Optional[str]]:
        """Map row continuous values to granule labels."""
        discrete_row = {}
        for col in numeric_cols:
            val = row[col]
            if pd.isna(val):
                discrete_row[col] = None
            else:
                assigned = None
                for g in self.granules_per_attr.get(col, []):
                    if g.contains(float(val)):
                        assigned = g.label
                        break
                discrete_row[col] = assigned
        return discrete_row

    def _compute_rar_support(
        self,
        itemset: Set[str],
        discrete_dataset: List[Dict[str, Optional[str]]],
        item_to_attr: Dict[str, str]
    ) -> float:
        """RAR Support metric."""
        sigma_count = 0
        disabled_count = 0
        total_s = len(discrete_dataset)
        attrs = {item_to_attr[item] for item in itemset}

        for row in discrete_dataset:
            if any(row.get(attr) is None for attr in attrs):
                disabled_count += 1
                continue
            if all(row.get(item_to_attr[item]) == item for item in itemset):
                sigma_count += 1

        denom = total_s - disabled_count
        return (sigma_count / denom) if denom > 0 else 0.0

    def _compute_rar_confidence_and_entropy(
        self,
        ant_item: str,
        con_item: str,
        target_granules: List[UpgradedGranule],
        discrete_dataset: List[Dict[str, Optional[str]]],
        item_to_attr: Dict[str, str]
    ) -> Tuple[float, float]:
        """
        Compute RAR Confidence and Posterior Conditional Entropy H(Y | X)
        to prevent accepting ambiguous weak rules.
        """
        ant_attr = item_to_attr[ant_item]
        con_attr = item_to_attr[con_item]

        sigma_ant = 0
        consequent_distribution: Dict[str, int] = {g.label: 0 for g in target_granules}
        dis_con_and_ant = 0

        for row in discrete_dataset:
            if row.get(ant_attr) == ant_item:
                sigma_ant += 1
                assigned_con = row.get(con_attr)
                if assigned_con is None:
                    dis_con_and_ant += 1
                elif assigned_con in consequent_distribution:
                    consequent_distribution[assigned_con] += 1

        denom = sigma_ant - dis_con_and_ant
        if denom <= 0:
            return 0.0, 999.0

        con_count = consequent_distribution.get(con_item, 0)
        conf = con_count / denom

        # Calculate Shannon Entropy of consequent given antecedent
        probs = [c / denom for c in consequent_distribution.values() if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs) if probs else 0.0

        return conf, entropy

    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute GARDA+ Imputation with LDR reconstruction and Entropy-gated Fallback.
        """
        start_time = time.time()
        result_df = df.copy()

        if numeric_cols is None:
            numeric_cols = list(result_df.select_dtypes(include=[np.number]).columns)
        else:
            numeric_cols = [c for c in numeric_cols if c in result_df.columns and pd.api.types.is_numeric_dtype(result_df[c])]

        if not numeric_cols or len(result_df) == 0:
            return result_df, {"imputed_cells": 0, "status": "skipped"}

        initial_missing = int(result_df[numeric_cols].isna().sum().sum())
        if initial_missing == 0:
            return result_df, {"imputed_cells": 0, "status": "no_missing"}

        # 1. Adaptive Granulation
        self._build_adaptive_granules(result_df, numeric_cols)

        item_to_attr: Dict[str, str] = {}
        item_to_granule: Dict[str, UpgradedGranule] = {}
        for attr, glist in self.granules_per_attr.items():
            for g in glist:
                item_to_attr[g.label] = attr
                item_to_granule[g.label] = g

        discrete_dataset = [self._discretize_row(result_df.iloc[i], numeric_cols) for i in range(len(result_df))]

        # 2. Mine High Confidence Rules with Entropy Gating
        all_items = list(item_to_attr.keys())
        L1 = [
            item for item in all_items
            if self._compute_rar_support({item}, discrete_dataset, item_to_attr) >= self.init_sup
        ]

        garda_plus_rules: List[Dict[str, Any]] = []
        for i in range(len(L1)):
            for j in range(len(L1)):
                if i != j and item_to_attr[L1[i]] != item_to_attr[L1[j]]:
                    ant = L1[i]
                    con = L1[j]
                    con_attr = item_to_attr[con]
                    target_granules = self.granules_per_attr[con_attr]

                    conf, entropy = self._compute_rar_confidence_and_entropy(
                        ant, con, target_granules, discrete_dataset, item_to_attr
                    )

                    # Entropy Gating Filter
                    if conf >= self.init_conf and entropy <= self.entropy_threshold:
                        sup = self._compute_rar_support({ant, con}, discrete_dataset, item_to_attr)
                        garda_plus_rules.append({
                            "ant": ant,
                            "con": con,
                            "conf": conf,
                            "sup": sup,
                            "entropy": entropy
                        })

        rule_imputed_count = 0
        imputed_cell_alphas: Dict[Tuple[int, str], float] = {}

        # 3. Apply High-Confidence Rules with Local Density Reconstruction Tracking
        for row_idx in range(len(discrete_dataset)):
            row = discrete_dataset[row_idx]
            for target_col in numeric_cols:
                if row[target_col] is None:
                    matching = []
                    for r in garda_plus_rules:
                        if item_to_attr[r["con"]] == target_col:
                            ant_attr = item_to_attr[r["ant"]]
                            if row.get(ant_attr) == r["ant"]:
                                matching.append(r)

                    if matching:
                        matching.sort(key=lambda x: (x["conf"], -x["entropy"], x["sup"]), reverse=True)
                        best = matching[0]
                        row[target_col] = best["con"]
                        rule_imputed_count += 1

                        # Calculate relative alpha from antecedent continuous value
                        ant_attr = item_to_attr[best["ant"]]
                        ant_val = result_df.iloc[row_idx][ant_attr]
                        if not pd.isna(ant_val):
                            ant_granule = item_to_granule[best["ant"]]
                            alpha = ant_granule.get_relative_position(float(ant_val))
                            imputed_cell_alphas[(row_idx, target_col)] = (alpha, ant_attr)

        # 4. Iterative Relaxation with Entropy Safeguards
        y = 1
        relaxed_rules_count = 0
        while y <= self.num_max_iter:
            has_missing = any(row[c] is None for row in discrete_dataset for c in numeric_cols)
            if not has_missing:
                break

            min_sup = self.init_sup / (self.r_sup * y)
            min_conf = self.init_conf / (self.r_conf ** y)
            if min_sup <= 0.02 or min_conf < 0.45:
                break

            for row_idx in range(len(discrete_dataset)):
                row = discrete_dataset[row_idx]
                non_missing_items = [row[c] for c in numeric_cols if row[c] is not None]

                for target_col in numeric_cols:
                    if row[target_col] is None:
                        candidates = []
                        target_granules = self.granules_per_attr.get(target_col, [])

                        for ant_item in non_missing_items:
                            for g in target_granules:
                                con_item = g.label
                                sup = self._compute_rar_support({ant_item, con_item}, discrete_dataset, item_to_attr)
                                if sup >= min_sup:
                                    conf, entropy = self._compute_rar_confidence_and_entropy(
                                        ant_item, con_item, target_granules, discrete_dataset, item_to_attr
                                    )
                                    if conf >= min_conf and entropy <= (self.entropy_threshold * 1.25):
                                        candidates.append({
                                            "ant": ant_item,
                                            "con": con_item,
                                            "conf": conf,
                                            "sup": sup,
                                            "entropy": entropy
                                        })

                        if candidates:
                            candidates.sort(key=lambda x: (x["conf"], -x["entropy"], x["sup"]), reverse=True)
                            best = candidates[0]
                            row[target_col] = best["con"]
                            rule_imputed_count += 1
                            relaxed_rules_count += 1

                            ant_attr = item_to_attr[best["ant"]]
                            ant_val = result_df.iloc[row_idx][ant_attr]
                            if not pd.isna(ant_val):
                                ant_granule = item_to_granule[best["ant"]]
                                alpha = ant_granule.get_relative_position(float(ant_val))
                                imputed_cell_alphas[(row_idx, target_col)] = (alpha, ant_attr)
            y += 1

        # 5. Localized Density & Continuous Antecedent Reconstruction (LDR)
        for row_idx in range(len(result_df)):
            for col in numeric_cols:
                if pd.isna(result_df.iat[row_idx, result_df.columns.get_loc(col)]):
                    assigned_label = discrete_dataset[row_idx].get(col)
                    if assigned_label and assigned_label in item_to_granule:
                        con_granule = item_to_granule[assigned_label]
                        alpha_info = imputed_cell_alphas.get((row_idx, col))
                        if alpha_info is not None:
                            alpha_val, ant_name = alpha_info
                            corr_sign = self.corr_matrix.get(ant_name, {}).get(col, 1.0)
                            recon_val = con_granule.reconstruct_from_alpha(alpha_val, sign=corr_sign)
                        else:
                            recon_val = con_granule.centroid
                        result_df.iat[row_idx, result_df.columns.get_loc(col)] = recon_val

        # 6. Hybrid Fallback (Bayesian Ridge Regression)
        remaining_missing = int(result_df[numeric_cols].isna().sum().sum())
        fallback_imputed_count = 0

        if remaining_missing > 0:
            for col in numeric_cols:
                if result_df[col].isna().sum() > 0:
                    other_cols = [c for c in numeric_cols if c != col]
                    train_mask = result_df[col].notna()
                    test_mask = result_df[col].isna()

                    if train_mask.sum() >= 3 and other_cols:
                        X_train = result_df.loc[train_mask, other_cols].fillna(result_df[other_cols].mean())
                        y_train = result_df.loc[train_mask, col]
                        X_test = result_df.loc[test_mask, other_cols].fillna(result_df[other_cols].mean())

                        model = BayesianRidge()
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        result_df.loc[test_mask, col] = preds
                        fallback_imputed_count += len(preds)
                    else:
                        col_median = result_df[col].median()
                        result_df[col] = result_df[col].fillna(col_median)
                        fallback_imputed_count += int(test_mask.sum())

        elapsed_time = round(time.time() - start_time, 3)

        summary = {
            "initial_missing": initial_missing,
            "final_missing": int(result_df[numeric_cols].isna().sum().sum()),
            "rule_imputed_count": rule_imputed_count,
            "relaxed_rules_count": relaxed_rules_count,
            "fallback_imputed_count": fallback_imputed_count,
            "rules_mined_count": len(garda_plus_rules),
            "optimal_k_per_attr": self.optimal_k_per_attr,
            "elapsed_time_sec": elapsed_time
        }

        return result_df, summary


def benchmark_imputation_methods(
    df_complete: pd.DataFrame,
    numeric_cols: List[str],
    error_model: str = "mcar",
    missing_rate: float = 0.20,
    consecutive_gap_size: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Experimental Benchmarking Engine comparing:
    - Baseline Mean
    - Baseline Median
    - KNN Imputation
    - Bayesian Ridge Regression
    - Base Paper GARDA
    - Proposed Upgraded GARDA+

    Evaluates across: RMSE, MAE, NRMSE, R², and Execution Time.
    """
    np.random.seed(random_seed)
    df_clean = df_complete[numeric_cols].dropna().copy()
    if len(df_clean) < 10:
        raise ValueError("Insufficient rows without missing values for benchmarking (minimum 10 required).")

    df_corrupted = df_clean.copy()
    total_elements = df_clean.shape[0] * df_clean.shape[1]
    mask = np.zeros(df_clean.shape, dtype=bool)

    if error_model.lower() == "mcar":
        # Missing Completely at Random
        n_missing = int(total_elements * missing_rate)
        flat_indices = np.random.choice(total_elements, size=n_missing, replace=False)
        for idx in flat_indices:
            r = idx // df_clean.shape[1]
            c = idx % df_clean.shape[1]
            mask[r, c] = True
            df_corrupted.iat[r, c] = np.nan
    else:
        # Consecutive Error Model (Sensor Outage)
        gap = min(consecutive_gap_size, max(3, len(df_clean) // 3))
        for col_idx in range(df_clean.shape[1]):
            start_row = np.random.randint(0, len(df_clean) - gap)
            for r in range(start_row, start_row + gap):
                mask[r, col_idx] = True
                df_corrupted.iat[r, col_idx] = np.nan

    ground_truth = df_clean.values[mask]

    def evaluate_predictions(preds_matrix: np.ndarray, model_name: str, duration: float) -> Dict[str, Any]:
        preds_extracted = preds_matrix[mask]
        rmse = float(root_mean_squared_error(ground_truth, preds_extracted))
        mae = float(mean_absolute_error(ground_truth, preds_extracted))
        val_range = float(np.max(ground_truth) - np.min(ground_truth))
        nrmse = (rmse / val_range) if val_range > 0 else 0.0
        r2 = float(r2_score(ground_truth, preds_extracted))

        return {
            "model": model_name,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "nrmse": round(nrmse, 4),
            "r2_score": round(r2, 4),
            "time_sec": round(duration, 3)
        }

    results = []

    # 1. Mean Baseline
    t0 = time.time()
    df_mean = df_corrupted.fillna(df_corrupted.mean())
    results.append(evaluate_predictions(df_mean.values, "Baseline (Mean)", time.time() - t0))

    # 2. Median Baseline
    t0 = time.time()
    df_median = df_corrupted.fillna(df_corrupted.median())
    results.append(evaluate_predictions(df_median.values, "Baseline (Median)", time.time() - t0))

    # 3. KNN Imputation
    t0 = time.time()
    knn = KNNImputer(n_neighbors=min(5, len(df_clean) - 1))
    df_knn = pd.DataFrame(knn.fit_transform(df_corrupted), columns=numeric_cols)
    results.append(evaluate_predictions(df_knn.values, "KNN Imputation", time.time() - t0))

    # 4. Bayesian Ridge
    t0 = time.time()
    df_ridge = df_corrupted.copy()
    for col in numeric_cols:
        others = [c for c in numeric_cols if c != col]
        tr_m = df_ridge[col].notna()
        te_m = df_ridge[col].isna()
        if te_m.sum() > 0 and tr_m.sum() >= 2:
            X_tr = df_ridge.loc[tr_m, others].fillna(df_corrupted[others].mean())
            y_tr = df_ridge.loc[tr_m, col]
            X_te = df_ridge.loc[te_m, others].fillna(df_corrupted[others].mean())
            br = BayesianRidge()
            br.fit(X_tr, y_tr)
            df_ridge.loc[te_m, col] = br.predict(X_te)
    df_ridge = df_ridge.fillna(df_corrupted.mean())
    results.append(evaluate_predictions(df_ridge.values, "Bayesian Ridge", time.time() - t0))

    # 5. Base GARDA (from Base Paper)
    t0 = time.time()
    from .garda_base import GardaImputer
    base_garda = GardaImputer(granulation_method="kmeans", k_granules=8, init_sup=0.35, init_conf=0.70)
    df_base_garda, _ = base_garda.fit_transform(df_corrupted, numeric_cols)
    results.append(evaluate_predictions(df_base_garda.values, "Base GARDA (Paper)", time.time() - t0))

    # 6. Upgraded GARDA+ (Our Proposed Upgrade)
    t0 = time.time()
    garda_plus = UpgradedGardaImputer(entropy_threshold=1.3, init_sup=0.35, init_conf=0.70)
    df_garda_plus, plus_summary = garda_plus.fit_transform(df_corrupted, numeric_cols)
    results.append(evaluate_predictions(df_garda_plus.values, "Proposed GARDA+ (Upgraded)", time.time() - t0))

    # Identify best model by RMSE
    results_sorted = sorted(results, key=lambda x: x["rmse"])
    best_model = results_sorted[0]["model"]

    # Calculate improvement of GARDA+ over Base GARDA
    base_rmse = next(r["rmse"] for r in results if r["model"] == "Base GARDA (Paper)")
    plus_rmse = next(r["rmse"] for r in results if r["model"] == "Proposed GARDA+ (Upgraded)")
    pct_improvement = round(((base_rmse - plus_rmse) / base_rmse) * 100, 2) if base_rmse > 0 else 0.0

    return {
        "error_model": error_model.upper(),
        "missing_rate": missing_rate,
        "injected_missing_count": int(np.sum(mask)),
        "evaluated_cells": int(total_elements),
        "results_table": results,
        "best_model": best_model,
        "garda_plus_improvement_pct": pct_improvement,
        "sample_ground_truth": ground_truth[:10].tolist(),
        "sample_garda_plus_predictions": df_garda_plus.values[mask][:10].tolist()
    }
