"""SHAP explainability for the GBM fast path's decision (dev/RESEARCH.md
§5, "Explainability"): EU AI Act / debanking-law-era regulators expect
human-readable reasons for a decline, not a bare score.

NOTE: not currently wired into any pipeline or persisted artifact, see
the audit note in BUILDLOG.md. `explain()` works and is importable, but
no defend/orchestrate module calls it yet, so no SHAP output reaches the
UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


@dataclass
class ExplanationResult:
    base_value: float
    shap_values: np.ndarray  # (n_samples, n_features)
    feature_names: list[str]

    def top_features_for_row(self, row_idx: int, k: int = 5) -> list[tuple[str, float]]:
        contributions = list(zip(self.feature_names, self.shap_values[row_idx]))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        return contributions[:k]


def explain(model: xgb.XGBClassifier, X: pd.DataFrame) -> ExplanationResult:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).flatten()[0])
    return ExplanationResult(base_value=float(base), shap_values=np.asarray(sv), feature_names=list(X.columns))


