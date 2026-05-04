from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_process_data(
    output_path: Path,
    n_points: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic process data with drift, shifts, and injected anomalies."""
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2026-01-01", periods=n_points, freq="h")

    baseline = 100.0
    noise = rng.normal(0, 1.1, n_points)
    slow_drift = np.linspace(0, 2.0, n_points)
    value = baseline + noise + slow_drift

    # Process shift and isolated anomalies.
    value[260:] += 3.0
    anomaly_indices = np.array([80, 165, 210, 335, 410, 455])
    value[anomaly_indices] += rng.choice([-8.0, 8.0], size=len(anomaly_indices))

    temperature = 70 + rng.normal(0, 0.8, n_points) + slow_drift * 0.4
    pressure = 30 + rng.normal(0, 0.35, n_points)
    pressure[260:] += 0.9

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "measurement": value,
            "temperature": temperature,
            "pressure": pressure,
            "known_anomaly": 0,
        }
    )
    df.loc[anomaly_indices, "known_anomaly"] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_process_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
