from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndividualChartLimits:
    center_line: float
    moving_range_bar: float
    sigma: float
    ucl: float
    lcl: float
    warning_upper: float
    warning_lower: float


def calculate_imr_limits(values: pd.Series, baseline_points: int = 120) -> IndividualChartLimits:
    """Calculate I-chart limits from an initial stable baseline window."""
    baseline = values.iloc[:baseline_points].astype(float)
    moving_range = baseline.diff().abs().dropna()
    moving_range_bar = moving_range.mean()
    sigma = moving_range_bar / 1.128
    center_line = baseline.mean()

    return IndividualChartLimits(
        center_line=center_line,
        moving_range_bar=moving_range_bar,
        sigma=sigma,
        ucl=center_line + 3 * sigma,
        lcl=center_line - 3 * sigma,
        warning_upper=center_line + 2 * sigma,
        warning_lower=center_line - 2 * sigma,
    )


def apply_spc_rules(df: pd.DataFrame, value_col: str, limits: IndividualChartLimits) -> pd.DataFrame:
    """Apply common I-chart rules and return a scored copy of the input data."""
    scored = df.copy()
    values = scored[value_col].astype(float)
    above_center = values > limits.center_line
    below_center = values < limits.center_line

    scored["spc_rule_1_beyond_3sigma"] = (values > limits.ucl) | (values < limits.lcl)
    scored["spc_rule_2_beyond_2sigma"] = (values > limits.warning_upper) | (values < limits.warning_lower)
    scored["spc_rule_3_run_8_same_side"] = _rolling_all(above_center, 8) | _rolling_all(below_center, 8)
    scored["spc_rule_4_trend_6"] = _trend(values, 6)
    rule_cols = [col for col in scored.columns if col.startswith("spc_rule_")]
    scored["spc_signal"] = scored[rule_cols].any(axis=1)
    return scored


def _rolling_all(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).sum().eq(window).fillna(False)


def _trend(values: pd.Series, window: int) -> pd.Series:
    diffs = values.diff()
    increasing = diffs.gt(0).rolling(window=window - 1, min_periods=window - 1).sum().eq(window - 1)
    decreasing = diffs.lt(0).rolling(window=window - 1, min_periods=window - 1).sum().eq(window - 1)
    return (increasing | decreasing).replace(np.nan, False)
