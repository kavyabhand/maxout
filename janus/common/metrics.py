"""Shared evaluation metrics for every JANUS detector, deliberately NOT
centered on raw accuracy or ROC-AUC, on a ~99:1+ imbalanced fraud problem
both inflate perceived performance (a classifier that predicts "legit" for
everything scores ~99% accuracy and a deceptively high ROC-AUC). Precision,
Recall, F1, and PR-AUC are the headline numbers everywhere in this codebase.

Lives in common/ (not defend/) because generate/ modules (mule-ring and
voice-scam detectors) need it too; a prior version of this had generate/
importing from defend/, an inverted dependency this module exists to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float  # reported for completeness, NOT the headline metric
    threshold: float
    n_positive: int
    n_total: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "threshold": round(self.threshold, 4),
            "n_positive": self.n_positive,
            "n_total": self.n_total,
        }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None) -> EvalReport:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if threshold is None:
        threshold = best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    return EvalReport(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        pr_auc=average_precision_score(y_true, y_score),
        roc_auc=roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        threshold=threshold,
        n_positive=int(y_true.sum()),
        n_total=len(y_true),
    )
