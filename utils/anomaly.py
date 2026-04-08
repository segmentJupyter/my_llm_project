from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    model: IsolationForest
    is_anomaly: np.ndarray  # bool
    scores: np.ndarray      # higher => more anomalous (after sign flip)


def detect_anomalies(values: np.ndarray, contamination: float = 0.02, seed: int = 42) -> AnomalyResult:
    x = np.asarray(values, dtype=float).reshape(-1, 1)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
    )
    model.fit(x)
    pred = model.predict(x)  # -1 anomaly, 1 normal
    decision = model.decision_function(x)  # higher => more normal
    scores = -decision
    return AnomalyResult(model=model, is_anomaly=(pred == -1), scores=scores)

