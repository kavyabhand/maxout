"""Per-feature statistical fidelity metrics: the competition's "fidelity
of attacks in simulation" judging criterion requires quantitative proof
that synthetic data resembles real distributions, not a visual/narrative
claim. Three metrics, used together because they catch different failure
modes. See scorer.py for the full Fidelity Scorecard this feeds into
(correlation delta, distinguisher AUC, graph-topology similarity).

- Wasserstein distance (earth-mover's): sensitive to how FAR apart two
  distributions are, not just whether they're statistically distinguishable; a small Wasserstein distance is the strongest "these look the same"
  signal of the three.
- Kolmogorov-Smirnov test: standard, well-understood significance test;
  reported for completeness and because reviewers will expect it by name.
- Jensen-Shannon divergence: symmetric, bounded [0, ln 2], robust to
  distributions with different supports, useful specifically because
  Wasserstein distance can be misleadingly small for distributions with
  very different shapes but similar central tendency, and JS divergence
  catches shape mismatches Wasserstein alone can miss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


@dataclass
class FidelityReport:
    feature_name: str
    wasserstein: float
    ks_statistic: float
    ks_pvalue: float
    js_divergence: float
    real_n: int
    synthetic_n: int

    def as_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "wasserstein": round(self.wasserstein, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "js_divergence": round(self.js_divergence, 4),
            "real_n": self.real_n,
            "synthetic_n": self.synthetic_n,
        }


def _histogram_probs(a: np.ndarray, b: np.ndarray, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    return pa, pb


def compare_feature(real: np.ndarray, synthetic: np.ndarray, feature_name: str) -> FidelityReport:
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    real = real[~np.isnan(real)]
    synthetic = synthetic[~np.isnan(synthetic)]

    w_dist = wasserstein_distance(real, synthetic)
    ks_stat, ks_p = ks_2samp(real, synthetic)
    pa, pb = _histogram_probs(real, synthetic)
    js_div = float(jensenshannon(pa, pb) ** 2)  # jensenshannon returns sqrt(JSD); square back to the divergence itself

    return FidelityReport(
        feature_name=feature_name,
        wasserstein=float(w_dist),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        js_divergence=js_div,
        real_n=len(real),
        synthetic_n=len(synthetic),
    )
