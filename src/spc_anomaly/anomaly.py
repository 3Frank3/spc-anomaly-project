from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["measurement", "temperature", "pressure"]


def score_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """Score process rows with Isolation Forest."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    features = df[feature_cols].astype(float)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
    )
    predictions = model.fit_predict(scaled_features)
    scores = model.decision_function(scaled_features)

    scored = df.copy()
    scored["anomaly_score"] = scores
    scored["ml_anomaly"] = predictions == -1
    return scored
