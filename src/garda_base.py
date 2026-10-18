"""
garda_base.py - Faithful implementation of GARDA (IEEE IoT Journal, 2026).

Paper Reference:
D'Aniello, G., Della Corte, M., Gaeta, R., & Hong, T.-P. (2026).
"GARDA: Granular Association-Rule-Based Data Imputation Approach for IoT Sensor Networks."
IEEE Internet of Things Journal, Vol. 13, No. 17.

Architecture:
1. Granulation: Discretize continuous sensor data into interval-valued granules (Equal-Width or K-Means).
2. Robust Association Rule (RAR) Mining:
   - Evaluates Support and Confidence handling missing entries '?' without dropping tuples.
   - High-confidence initial pass (initSup, initConf).
   - Iterative threshold relaxation pass (rSup, rConf) for unresolved gaps.
3. Reconstruction: Map imputed granules back to continuous values (midpoint or cluster centroid).
4. Hybrid Fallback: Complete remaining unresolved missing cells using Bayesian Ridge Regression.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import BayesianRidge


class Granule:
    """Represents an interval-valued granule for a specific attribute."""
    def __init__(self, attribute: str, index: int, start: float, end: float, centroid: Optional[float] = None):
        self.attribute = attribute
        self.index = index
        self.start = float(start)
        self.end = float(end)
        self.centroid = float(centroid) if centroid is not None else (self.start + self.end) / 2.0
        self.label = f"{attribute}_G{index}"

    def contains(self, value: float) -> bool:
        if np.isnan(value):
            return False
        return self.start <= value <= self.end

    def __repr__(self) -> str:
        return f"{self.label}[{self.start:.2f}, {self.end:.2f}]"

    def __eq__(self, other) -> bool:
        return isinstance(other, Granule) and self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)


class GardaImputer:
    """
    GARDA implementation matching Algorithm 1 from the IEEE paper.
    """

    def __init__(
        self,
        granulation_method: str = "kmeans",
        k_granules: int = 15,
        init_sup: float = 0.50,
        init_conf: float = 0.80,
        num_max_iter: int = 10,
        r_sup: float = 10.0,
        r_conf: float = 1.0,
        fallback_method: str = "bayesian_ridge"
    ):
        self.granulation_method = granulation_method.lower()
        self.k_granules = max(3, k_granules)
        self.init_sup = init_sup
        self.init_conf = init_conf
        self.num_max_iter = num_max_iter
        self.r_sup = r_sup
        self.r_conf = r_conf
        self.fallback_method = fallback_method
        self.granules_per_attr: Dict[str, List[Granule]] = {}
        self.min_max_per_attr: Dict[str, Tuple[float, float]] = {}

    def _build_granules(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Build granules for each continuous attribute."""
        self.granules_per_attr.clear()
        self.min_max_per_attr.clear()

        for col in numeric_cols:
            valid_vals = df[col].dropna().values
            if len(valid_vals) == 0:
                continue

            vmin = float(np.min(valid_vals))
            vmax = float(np.max(valid_vals))
            self.min_max_per_attr[col] = (vmin, vmax)

            if vmin == vmax:
                g = Granule(col, 0, vmin - 0.1, vmax + 0.1, centroid=vmin)
                self.granules_per_attr[col] = [g]
                continue

            k = min(self.k_granules, len(np.unique(valid_vals)))

            if self.granulation_method == "kmeans":
                # Cluster-based granulation
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = km.fit_predict(valid_vals.reshape(-1, 1))
                centroids = km.cluster_centers_.flatten()
                granules = []
                for idx in range(k):
                    c_vals = valid_vals[labels == idx]
                    if len(c_vals) > 0:
                        g_start = float(np.min(c_vals))
                        g_end = float(np.max(c_vals))
                        centroid = float(centroids[idx])
                    else:
                        centroid = float(centroids[idx])
                        g_start = centroid - 0.01
                        g_end = centroid + 0.01
                    granules.append(Granule(col, idx, g_start, g_end, centroid=centroid))
                granules.sort(key=lambda g: g.start)
                # Adjust boundaries so they cover continuous interval
                for i in range(len(granules) - 1):
                    mid = (granules[i].end + granules[i + 1].start) / 2.0
                    granules[i].end = mid
                    granules[i + 1].start = mid
                granules[0].start = -float("inf")
                granules[-1].end = float("inf")
                self.granules_per_attr[col] = granules

            else:
                # Equal-width granulation
                width = (vmax - vmin) / k
                granules = []
                for idx in range(k):
                    g_start = vmin + idx * width
                    g_end = vmin + (idx + 1) * width
                    granules.append(Granule(col, idx, g_start, g_end))
                granules[0].start = -float("inf")
                granules[-1].end = float("inf")
                self.granules_per_attr[col] = granules

    def _discretize_row(self, row: pd.Series, numeric_cols: List[str]) -> Dict[str, Optional[str]]:
        """Map a tuple of continuous values into their respective granule labels."""
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
        """
        Compute Robust Association Rule (RAR) Support:
        Sup(X) = |sigma(X)| / (|S| - |Dis(X)|)
        Where Dis(X) is the set of transactions where any attribute in X is missing.
        """
        sigma_count = 0
        disabled_count = 0
        total_s = len(discrete_dataset)

        attrs_in_itemset = {item_to_attr[item] for item in itemset}

        for row in discrete_dataset:
            # Check if any attribute in itemset is missing in this row
            has_disabled = any(row.get(attr) is None for attr in attrs_in_itemset)
            if has_disabled:
                disabled_count += 1
                continue

            # Check if all items in itemset are present in row
            if all(row.get(item_to_attr[item]) == item for item in itemset):
                sigma_count += 1

        denom = total_s - disabled_count
        return (sigma_count / denom) if denom > 0 else 0.0

    def _compute_rar_confidence(
        self,
        ant_item: str,
        con_item: str,
        discrete_dataset: List[Dict[str, Optional[str]]],
        item_to_attr: Dict[str, str]
    ) -> float:
        """
        Compute RAR Confidence for ant_item -> con_item:
        Conf(X -> Y) = |sigma(X U Y)| / (|sigma(X)| - |Dis(Y) n sigma(X)|)
        """
        ant_attr = item_to_attr[ant_item]
        con_attr = item_to_attr[con_item]

        sigma_ant = 0
        sigma_union = 0
        dis_con_and_sigma_ant = 0

        for row in discrete_dataset:
            if row.get(ant_attr) == ant_item:
                sigma_ant += 1
                if row.get(con_attr) is None:
                    dis_con_and_sigma_ant += 1
                elif row.get(con_attr) == con_item:
                    sigma_union += 1

        denom = sigma_ant - dis_con_and_sigma_ant
        return (sigma_union / denom) if denom > 0 else 0.0

    def fit_transform(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute GARDA Imputation:
        Phase 1: Granulation
        Phase 2: GrC Association Rule Mining & Imputation (High confidence + Relaxation)
        Phase 3: Value Reconstruction & Hybrid Traditional Imputation
        """
        result_df = df.copy()
        if numeric_cols is None:
            numeric_cols = list(result_df.select_dtypes(include=[np.number]).columns)
        else:
            numeric_cols = [c for c in numeric_cols if c in result_df.columns and pd.api.types.is_numeric_dtype(result_df[c])]

        if not numeric_cols or len(result_df) == 0:
            return result_df, {"imputed_cells": 0, "status": "skipped_no_numeric_columns"}

        initial_missing = int(result_df[numeric_cols].isna().sum().sum())
        if initial_missing == 0:
            return result_df, {"imputed_cells": 0, "status": "no_missing_values"}

        # Phase 1: Granulation
        self._build_granules(result_df, numeric_cols)

        # Build reverse item lookup: item_label -> attribute and item_label -> Granule
        item_to_attr: Dict[str, str] = {}
        item_to_granule: Dict[str, Granule] = {}
        for attr, glist in self.granules_per_attr.items():
            for g in glist:
                item_to_attr[g.label] = attr
                item_to_granule[g.label] = g

        # Discretize incomplete dataset
        discrete_dataset = [self._discretize_row(result_df.iloc[i], numeric_cols) for i in range(len(result_df))]

        # Phase 2: GrC Association Rule Imputation (Pass 1 - High Confidence)
        # Mine frequent 1-itemsets L1
        all_items = list(item_to_attr.keys())
        L1: List[str] = []
        for item in all_items:
            sup = self._compute_rar_support({item}, discrete_dataset, item_to_attr)
            if sup >= self.init_sup:
                L1.append(item)

        # Extract 2-itemset candidate rules
        gar_rules: List[Dict[str, Any]] = []
        for i in range(len(L1)):
            for j in range(len(L1)):
                if i != j and item_to_attr[L1[i]] != item_to_attr[L1[j]]:
                    ant = L1[i]
                    con = L1[j]
                    conf = self._compute_rar_confidence(ant, con, discrete_dataset, item_to_attr)
                    if conf >= self.init_conf:
                        sup = self._compute_rar_support({ant, con}, discrete_dataset, item_to_attr)
                        gar_rules.append({
                            "ant": ant,
                            "con": con,
                            "conf": conf,
                            "sup": sup
                        })

        rule_imputed_count = 0

        # Step 4: Apply high-confidence rules
        for row_idx in range(len(discrete_dataset)):
            row = discrete_dataset[row_idx]
            for target_col in numeric_cols:
                if row[target_col] is None:
                    # Find applicable rules
                    matching_rules = []
                    for r in gar_rules:
                        if item_to_attr[r["con"]] == target_col:
                            ant_attr = item_to_attr[r["ant"]]
                            if row.get(ant_attr) == r["ant"]:
                                matching_rules.append(r)

                    if len(matching_rules) == 1:
                        row[target_col] = matching_rules[0]["con"]
                        rule_imputed_count += 1
                    elif len(matching_rules) > 1:
                        matching_rules.sort(key=lambda x: (x["conf"], x["sup"]), reverse=True)
                        best = matching_rules[0]
                        second = matching_rules[1]
                        if best["conf"] == second["conf"] and best["sup"] == second["sup"] and best["con"] != second["con"]:
                            pass  # Keep missing due to conflict
                        else:
                            row[target_col] = best["con"]
                            rule_imputed_count += 1

        # Iterative Relaxation Phase (Steps 5-12)
        y = 1
        relaxed_rules_count = 0
        while y <= self.num_max_iter:
            # Check if missing values remain
            has_missing = any(row[c] is None for row in discrete_dataset for c in numeric_cols)
            if not has_missing:
                break

            min_sup = self.init_sup / (self.r_sup * y)
            min_conf = self.init_conf / (self.r_conf * 1.0)
            if min_sup <= 0.01:
                break

            # Find candidate 2-itemsets for missing cells
            for row_idx in range(len(discrete_dataset)):
                row = discrete_dataset[row_idx]
                non_missing_items = [row[c] for c in numeric_cols if row[c] is not None]

                for target_col in numeric_cols:
                    if row[target_col] is None:
                        candidate_rules = []
                        target_granules = self.granules_per_attr.get(target_col, [])
                        for ant_item in non_missing_items:
                            for g in target_granules:
                                con_item = g.label
                                sup = self._compute_rar_support({ant_item, con_item}, discrete_dataset, item_to_attr)
                                if sup >= min_sup:
                                    conf = self._compute_rar_confidence(ant_item, con_item, discrete_dataset, item_to_attr)
                                    if conf >= min_conf:
                                        candidate_rules.append({"ant": ant_item, "con": con_item, "conf": conf, "sup": sup})

                        if candidate_rules:
                            candidate_rules.sort(key=lambda x: (x["conf"], x["sup"]), reverse=True)
                            best = candidate_rules[0]
                            row[target_col] = best["con"]
                            rule_imputed_count += 1
                            relaxed_rules_count += 1
            y += 1

        # Phase 3: Reconstruction (Granule -> Continuous Value)
        for row_idx in range(len(result_df)):
            for col in numeric_cols:
                if pd.isna(result_df.iloc[row_idx][col]):
                    assigned_granule_label = discrete_dataset[row_idx].get(col)
                    if assigned_granule_label and assigned_granule_label in item_to_granule:
                        reconstructed_val = item_to_granule[assigned_granule_label].centroid
                        result_df.iat[row_idx, result_df.columns.get_loc(col)] = reconstructed_val

        # Phase 4: Hybrid Fallback for remaining unresolved missing values
        remaining_missing = int(result_df[numeric_cols].isna().sum().sum())
        fallback_imputed_count = 0

        if remaining_missing > 0:
            for col in numeric_cols:
                if result_df[col].isna().sum() > 0:
                    other_cols = [c for c in numeric_cols if c != col]
                    train_mask = result_df[col].notna()
                    test_mask = result_df[col].isna()

                    if train_mask.sum() >= 2 and other_cols:
                        X_train = result_df.loc[train_mask, other_cols].fillna(result_df[other_cols].mean())
                        y_train = result_df.loc[train_mask, col]
                        X_test = result_df.loc[test_mask, other_cols].fillna(result_df[other_cols].mean())

                        model = BayesianRidge()
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        result_df.loc[test_mask, col] = preds
                        fallback_imputed_count += len(preds)
                    else:
                        # Simple column mean fallback if regression not feasible
                        col_mean = result_df[col].mean()
                        result_df[col] = result_df[col].fillna(col_mean)
                        fallback_imputed_count += int(test_mask.sum())

        final_missing = int(result_df[numeric_cols].isna().sum().sum())

        summary = {
            "initial_missing": initial_missing,
            "final_missing": final_missing,
            "rule_imputed_count": rule_imputed_count,
            "relaxed_rules_count": relaxed_rules_count,
            "fallback_imputed_count": fallback_imputed_count,
            "rules_mined_count": len(gar_rules),
            "granulation_method": self.granulation_method,
            "k_granules": self.k_granules
        }

        return result_df, summary
