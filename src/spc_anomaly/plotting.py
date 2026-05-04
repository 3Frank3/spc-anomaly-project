from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.spc_anomaly.spc import IndividualChartLimits


def plot_spc_chart(df: pd.DataFrame, limits: IndividualChartLimits, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["timestamp"], df["measurement"], color="#334155", linewidth=1.2, label="Measurement")

    spc_points = df[df["spc_signal"]]
    ax.scatter(
        spc_points["timestamp"],
        spc_points["measurement"],
        color="#dc2626",
        s=28,
        label="SPC signal",
        zorder=3,
    )

    ax.axhline(limits.center_line, color="#2563eb", linestyle="-", linewidth=1, label="Center line")
    ax.axhline(limits.ucl, color="#dc2626", linestyle="--", linewidth=1, label="UCL/LCL")
    ax.axhline(limits.lcl, color="#dc2626", linestyle="--", linewidth=1)
    ax.axhline(limits.warning_upper, color="#f59e0b", linestyle=":", linewidth=1, label="2 sigma")
    ax.axhline(limits.warning_lower, color="#f59e0b", linestyle=":", linewidth=1)

    ax.set_title("Individual Control Chart")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Measurement")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_anomaly_scores(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["timestamp"], df["anomaly_score"], color="#475569", linewidth=1.2, label="Anomaly score")

    anomaly_points = df[df["ml_anomaly"]]
    ax.scatter(
        anomaly_points["timestamp"],
        anomaly_points["anomaly_score"],
        color="#7c3aed",
        s=28,
        label="ML anomaly",
        zorder=3,
    )

    ax.axhline(0, color="#0f172a", linestyle="--", linewidth=1)
    ax.set_title("Isolation Forest Anomaly Scores")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
