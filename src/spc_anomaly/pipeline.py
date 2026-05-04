from __future__ import annotations

import argparse
from pathlib import Path

from src.spc_anomaly.anomaly import score_anomalies
from src.spc_anomaly.data import generate_process_data, load_process_data
from src.spc_anomaly.plotting import plot_anomaly_scores, plot_spc_chart
from src.spc_anomaly.spc import apply_spc_rules, calculate_imr_limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an SPC + anomaly detection pipeline.")
    parser.add_argument("--raw-data", type=Path, default=Path("data/raw/process_data.csv"))
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed/scored_process_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--baseline-points", type=int, default=120)
    parser.add_argument("--contamination", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic raw data even if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.regenerate or not args.raw_data.exists():
        df = generate_process_data(args.raw_data, n_points=args.n_points, seed=args.seed)
    else:
        df = load_process_data(args.raw_data)

    limits = calculate_imr_limits(df["measurement"], baseline_points=args.baseline_points)
    scored = apply_spc_rules(df, "measurement", limits)
    scored = score_anomalies(scored, contamination=args.contamination, seed=args.seed)
    scored["combined_signal"] = scored["spc_signal"] | scored["ml_anomaly"]

    args.processed_data.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.processed_data, index=False)

    figure_dir = args.output_dir / "figures"
    plot_spc_chart(scored, limits, figure_dir / "spc_chart.png")
    plot_anomaly_scores(scored, figure_dir / "anomaly_scores.png")
    write_report(scored, limits, args.output_dir / "report.md")

    print(f"Raw data: {args.raw_data}")
    print(f"Scored data: {args.processed_data}")
    print(f"Report: {args.output_dir / 'report.md'}")
    print(f"SPC signals: {int(scored['spc_signal'].sum())}")
    print(f"ML anomalies: {int(scored['ml_anomaly'].sum())}")
    print(f"Combined signals: {int(scored['combined_signal'].sum())}")


def write_report(scored, limits, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rule_counts = {
        "Rule 1, beyond 3 sigma": int(scored["spc_rule_1_beyond_3sigma"].sum()),
        "Rule 2, beyond 2 sigma": int(scored["spc_rule_2_beyond_2sigma"].sum()),
        "Rule 3, 8 points same side": int(scored["spc_rule_3_run_8_same_side"].sum()),
        "Rule 4, 6-point trend": int(scored["spc_rule_4_trend_6"].sum()),
    }
    lines = [
        "# SPC + Anomaly Detection Report",
        "",
        "## Control Limits",
        "",
        f"- Center line: {limits.center_line:.3f}",
        f"- Sigma estimate: {limits.sigma:.3f}",
        f"- UCL: {limits.ucl:.3f}",
        f"- LCL: {limits.lcl:.3f}",
        "",
        "## Signal Counts",
        "",
        f"- SPC signals: {int(scored['spc_signal'].sum())}",
        f"- ML anomalies: {int(scored['ml_anomaly'].sum())}",
        f"- Combined signals: {int(scored['combined_signal'].sum())}",
        "",
        "## SPC Rule Counts",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in rule_counts.items())
    lines.extend(
        [
            "",
            "## Suggested Review",
            "",
            "Review rows where `combined_signal` is true, then classify each signal as special-cause, expected transition, or false positive.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
