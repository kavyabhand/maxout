"""Anomaly defend family, unsupervised IsolationForest, catching
genuinely novel, unlabeled patterns none of the four supervised families
were trained to recognize. Routed to review, not auto-decline, in the
meta-learner's decision tiers (meta.py): an unsupervised flag on its own
is too weak a signal to hard-decline a legitimate transaction on, but
exactly the right signal to route to a human for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from janus.common.metrics import EvalReport, evaluate


@dataclass
class AnomalyModelResult:
    model: IsolationForest
    eval_report: EvalReport


def train_anomaly_detector(x_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.Series, *, contamination: float = 0.01, random_state: int = 42) -> AnomalyModelResult:
    """Trained UNSUPERVISED on x_train (label not used at fit time, by
    design, that's what makes this family able to catch patterns the
    other four, all label-supervised, structurally cannot). y_test is only
    used afterward to measure how well the resulting anomaly score
    happens to line up with known fraud, for reporting purposes."""

    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(x_train)

    # score_samples: higher = more normal. Flip and min-max scale to [0, 1]
    # so it's directly comparable to the other families' fraud-probability outputs.
    raw = -model.score_samples(x_test)
    score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

    report = evaluate(y_test.to_numpy(), score)
    return AnomalyModelResult(model=model, eval_report=report)


def anomaly_score(model: IsolationForest, x: pd.DataFrame) -> np.ndarray:
    raw = -model.score_samples(x)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
