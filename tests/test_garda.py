"""
test_garda.py - Unit tests for Base Paper GARDA and Proposed Upgraded GARDA+.
"""

import pandas as pd
import numpy as np
from src.garda_base import GardaImputer
from src.garda_upgraded import UpgradedGardaImputer, benchmark_imputation_methods


def test_base_garda_imputation():
    # Correlated sensor readings
    np.random.seed(42)
    t1 = np.linspace(20.0, 30.0, 25)
    t2 = t1 * 1.05 + np.random.normal(0, 0.2, 25)
    h = 80.0 - (t1 * 1.5) + np.random.normal(0, 0.5, 25)

    df = pd.DataFrame({"temp1": t1, "temp2": t2, "humidity": h})
    # Inject missing values
    df.loc[3, "temp1"] = np.nan
    df.loc[7, "temp2"] = np.nan
    df.loc[12, "humidity"] = np.nan

    imputer = GardaImputer(granulation_method="kmeans", k_granules=5, init_sup=0.3, init_conf=0.6)
    imputed_df, summary = imputer.fit_transform(df, ["temp1", "temp2", "humidity"])

    assert imputed_df.isna().sum().sum() == 0
    assert summary["initial_missing"] == 3
    assert summary["final_missing"] == 0


def test_upgraded_garda_plus_imputation():
    np.random.seed(42)
    t1 = np.linspace(20.0, 30.0, 30)
    t2 = t1 * 1.02 + np.random.normal(0, 0.1, 30)
    p = 1013.0 - (t1 * 0.2) + np.random.normal(0, 0.1, 30)

    df = pd.DataFrame({"temp1": t1, "temp2": t2, "pressure": p})
    # Inject missing values
    df.loc[2, "temp1"] = np.nan
    df.loc[5, "pressure"] = np.nan

    garda_plus = UpgradedGardaImputer(entropy_threshold=1.4, init_sup=0.3, init_conf=0.6)
    imputed_df, summary = garda_plus.fit_transform(df, ["temp1", "temp2", "pressure"])

    assert imputed_df.isna().sum().sum() == 0
    assert summary["initial_missing"] == 2
    assert summary["final_missing"] == 0
    assert "elapsed_time_sec" in summary


def test_benchmark_imputation_methods():
    np.random.seed(42)
    x = np.linspace(10, 50, 40)
    y = x * 2.0 + np.random.normal(0, 1.0, 40)
    z = 100 - x + np.random.normal(0, 1.0, 40)
    df = pd.DataFrame({"s1": x, "s2": y, "s3": z})

    bench_res = benchmark_imputation_methods(
        df_complete=df,
        numeric_cols=["s1", "s2", "s3"],
        error_model="mcar",
        missing_rate=0.15
    )

    assert bench_res["error_model"] == "MCAR"
    assert len(bench_res["results_table"]) == 6  # Mean, Median, KNN, Ridge, Base GARDA, GARDA+
    assert bench_res["injected_missing_count"] > 0
