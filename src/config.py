"""
config.py - Centralized configuration, thresholds, and mathematical weights for DataSentry.

Maintains all system constants in one auditable location to guarantee
consistency across analytical modules, scoring engines, and UI layers.
"""

from typing import Dict, Tuple

# ==============================================================================
# 1. QUALITY SCORE DIMENSIONS & MATHEMATICAL WEIGHTS
# Formula: Score = W_comp*Comp + W_uniq*Uniq + W_val*Val + W_cons*Cons
# ==============================================================================
QUALITY_WEIGHTS: Dict[str, float] = {
    "completeness": 0.35,
    "uniqueness": 0.25,
    "validity": 0.20,
    "consistency": 0.20,
}

# Grade boundaries: (min_score, grade_letter, color_hex, description)
QUALITY_GRADES: Dict[str, Tuple[float, str, str]] = {
    "A": (90.0, "#10B981", "Excellent dataset quality. Ready for production and modeling."),
    "B": (80.0, "#3B82F6", "Good quality with minor defects. Safe for exploratory data analysis."),
    "C": (70.0, "#F59E0B", "Moderate quality with notable missingness or outliers. Remediation recommended."),
    "D": (60.0, "#F97316", "Poor quality. High risk of predictive distortion and calculation bias."),
    "F": (0.0, "#EF4444", "Critical condition. Severe defects across multiple dimensions. Immediate remediation required."),
}

# ==============================================================================
# 2. STATISTICAL OUTLIER DETECTION DEFAULTS
# ==============================================================================
OUTLIER_CONFIG = {
    "iqr_standard_multiplier": 1.5,
    "iqr_extreme_multiplier": 3.0,
    "zscore_standard_threshold": 3.0,
    "modified_zscore_mad_threshold": 3.5,  # Iglewicz and Hoaglin (1993) standard
}

# ==============================================================================
# 3. GARDA & GARDA+ ALGORITHMIC HYPERPARAMETERS
# (IEEE Internet of Things Journal 2026 Reference Implementation)
# ==============================================================================
GARDA_CONFIG = {
    "default_k_granules": 8,
    "default_init_support": 0.35,
    "default_init_confidence": 0.70,
    "default_entropy_threshold": 1.25,  # Shannon entropy threshold θ_H for EG-defense
    "relaxation_r_sup": 10.0,
    "relaxation_r_conf": 1.05,
    "max_iterations": 8,
}

# ==============================================================================
# 4. APPLICATION UI THEME CONFIGURATION
# ==============================================================================
UI_CONFIG = {
    "app_name": "DataSentry",
    "app_tagline": "Intelligent Dataset Quality Investigator & Research Suite",
    "app_version": "2.2.0-Production",
    "primary_color": "#2563EB",
    "secondary_color": "#0F172A",
    "background_color": "#FFFFFF",
    "card_background": "#F8FAFC",
    "border_color": "#E2E8F0",
    "max_display_rows": 1000,
}
