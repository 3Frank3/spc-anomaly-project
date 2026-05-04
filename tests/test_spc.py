import pandas as pd

from src.spc_anomaly.spc import apply_spc_rules, calculate_imr_limits


def test_calculate_imr_limits_returns_expected_ordering():
    values = pd.Series([10, 10.2, 9.8, 10.1, 10.0, 10.3, 9.9, 10.1])

    limits = calculate_imr_limits(values, baseline_points=len(values))

    assert limits.lcl < limits.center_line < limits.ucl
    assert limits.warning_lower < limits.center_line < limits.warning_upper


def test_apply_spc_rules_flags_beyond_3sigma():
    values = pd.Series([10, 10.2, 9.8, 10.1, 10.0, 10.3, 9.9, 10.1, 25.0])
    df = pd.DataFrame({"measurement": values})
    limits = calculate_imr_limits(values.iloc[:8], baseline_points=8)

    scored = apply_spc_rules(df, "measurement", limits)

    assert bool(scored.loc[8, "spc_rule_1_beyond_3sigma"])
    assert bool(scored.loc[8, "spc_signal"])
