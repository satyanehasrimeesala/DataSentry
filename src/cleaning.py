"""
cleaning.py - State-preserving, reversible Data Cleaning Workflow Manager for DataSentry.

Features:
- Maintains untouched raw original dataset
- Linear history stack with full Undo capability
- Reset-to-original dataset functionality
- Operation audit log with timestamps, parameters, and affected cell counts
- Pre-cleaning diff previews and impact validation
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from .missing_values import impute_or_drop_missing
from .duplicates import remove_duplicates
from .outliers import remediate_outliers
from .consistency import normalize_text, convert_column_type


class CleaningWorkflowManager:
    """
    Manages active cleaning sessions with audit logging and rollback capabilities.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self._raw_df: pd.DataFrame = raw_df.copy()
        self._current_df: pd.DataFrame = raw_df.copy()
        self._history_stack: List[pd.DataFrame] = [raw_df.copy()]
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def raw_data(self) -> pd.DataFrame:
        """Immutable view of the raw dataset."""
        return self._raw_df.copy()

    @property
    def current_data(self) -> pd.DataFrame:
        """Current modified dataset."""
        return self._current_df.copy()

    @property
    def history_length(self) -> int:
        """Number of undoable operations performed."""
        return len(self._audit_log)

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """List of all applied cleaning actions."""
        return list(self._audit_log)

    def reset_to_original(self) -> Dict[str, Any]:
        """
        Revert all changes and restore original dataset.
        """
        previous_rows = len(self._current_df)
        self._current_df = self._raw_df.copy()
        self._history_stack = [self._raw_df.copy()]
        entry = {
            "step": len(self._audit_log) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": "RESET_TO_ORIGINAL",
            "description": "Reset dataset to raw uploaded state.",
            "rows_before": previous_rows,
            "rows_after": len(self._current_df),
            "columns_after": len(self._current_df.columns),
            "details": {}
        }
        self._audit_log.append(entry)
        return entry

    def undo(self) -> Optional[Dict[str, Any]]:
        """
        Revert the most recent operation.
        """
        if len(self._history_stack) <= 1:
            return None

        # Pop current state
        self._history_stack.pop()
        # Restore prior state
        self._current_df = self._history_stack[-1].copy()
        reverted_action = self._audit_log.pop() if self._audit_log else None

        return {
            "reverted_action": reverted_action,
            "current_rows": len(self._current_df),
            "current_columns": len(self._current_df.columns),
            "remaining_history_steps": len(self._audit_log)
        }

    def apply_missing_value_action(
        self,
        strategy: str,
        target_columns: Optional[List[str]] = None,
        custom_value: Optional[Any] = None,
        drop_threshold_pct: float = 50.0,
        knn_neighbors: int = 5
    ) -> Dict[str, Any]:
        """Apply missing value imputation or dropping."""
        new_df, summary = impute_or_drop_missing(
            self._current_df,
            strategy=strategy,
            target_columns=target_columns,
            custom_value=custom_value,
            drop_threshold_pct=drop_threshold_pct,
            knn_neighbors=knn_neighbors
        )
        return self._commit_change(
            new_df=new_df,
            action=f"MISSING_VALUES_{strategy.upper()}",
            description=f"Applied {strategy} imputation/dropping on {target_columns or 'all columns'}",
            details=summary
        )

    def apply_duplicate_removal(
        self,
        subset_columns: Optional[List[str]] = None,
        keep_strategy: str = "first"
    ) -> Dict[str, Any]:
        """Apply duplicate removal."""
        new_df, summary = remove_duplicates(
            self._current_df,
            subset_columns=subset_columns,
            keep_strategy=keep_strategy
        )
        return self._commit_change(
            new_df=new_df,
            action="REMOVE_DUPLICATES",
            description=f"Removed duplicate rows keeping {keep_strategy} occurrence",
            details=summary
        )

    def apply_outlier_remediation(
        self,
        strategy: str,
        target_columns: List[str],
        multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """Apply outlier clipping, dropping, or nullification."""
        new_df, summary = remediate_outliers(
            self._current_df,
            strategy=strategy,
            target_columns=target_columns,
            multiplier=multiplier
        )
        return self._commit_change(
            new_df=new_df,
            action=f"OUTLIER_{strategy.upper()}",
            description=f"Outlier {strategy} applied on columns {target_columns} (k={multiplier})",
            details=summary
        )

    def apply_text_normalization(
        self,
        target_columns: List[str],
        strip_whitespace: bool = True,
        case_style: Optional[str] = None,
        collapse_spaces: bool = False
    ) -> Dict[str, Any]:
        """Apply string cleaning."""
        new_df, summary = normalize_text(
            self._current_df,
            target_columns=target_columns,
            strip_whitespace=strip_whitespace,
            case_style=case_style,
            collapse_spaces=collapse_spaces
        )
        return self._commit_change(
            new_df=new_df,
            action="TEXT_NORMALIZATION",
            description=f"Normalized text for {target_columns} (strip={strip_whitespace}, case={case_style})",
            details=summary
        )

    def apply_type_conversion(
        self,
        column: str,
        target_type: str,
        date_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply explicit type conversion."""
        new_df, summary = convert_column_type(
            self._current_df,
            column=column,
            target_type=target_type,
            date_format=date_format
        )
        return self._commit_change(
            new_df=new_df,
            action="TYPE_CONVERSION",
            description=f"Converted column '{column}' to {target_type}",
            details=summary
        )

    def apply_custom_dataframe(
        self,
        new_df: pd.DataFrame,
        action_name: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply custom modified DataFrame (e.g. from GARDA imputation)."""
        return self._commit_change(
            new_df=new_df,
            action=action_name,
            description=description,
            details=details or {}
        )

    def _commit_change(
        self,
        new_df: pd.DataFrame,
        action: str,
        description: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal helper to push state to stack and log audit record."""
        rows_before = len(self._current_df)
        cols_before = len(self._current_df.columns)

        self._current_df = new_df.copy()
        self._history_stack.append(self._current_df.copy())

        entry = {
            "step": len(self._audit_log) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "description": description,
            "rows_before": rows_before,
            "rows_after": len(self._current_df),
            "columns_before": cols_before,
            "columns_after": len(self._current_df.columns),
            "details": details
        }
        self._audit_log.append(entry)
        return entry
