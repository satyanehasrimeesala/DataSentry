"""
test_cleaning.py - Unit tests for CleaningWorkflowManager (undo, reset, history stack).
"""

import pandas as pd
import numpy as np
from src.cleaning import CleaningWorkflowManager


def test_workflow_manager_undo_and_reset():
    raw_df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": [10.0, 20.0, 20.0, np.nan]
    })
    mgr = CleaningWorkflowManager(raw_df)

    assert mgr.history_length == 0
    assert len(mgr.current_data) == 4

    # Step 1: Remove duplicates
    mgr.apply_duplicate_removal()
    assert len(mgr.current_data) == 3
    assert mgr.history_length == 1

    # Step 2: Mean imputation
    mgr.apply_missing_value_action(strategy="mean")
    assert mgr.current_data["val"].isna().sum() == 0
    assert mgr.history_length == 2

    # Test Undo (should revert mean imputation)
    undo_res = mgr.undo()
    assert undo_res is not None
    assert mgr.history_length == 1
    assert mgr.current_data["val"].isna().sum() == 1  # Missing value restored

    # Test Reset (should restore raw dataset completely)
    mgr.reset_to_original()
    assert len(mgr.current_data) == 4
    assert mgr.current_data["id"].tolist() == [1, 2, 2, 3]
