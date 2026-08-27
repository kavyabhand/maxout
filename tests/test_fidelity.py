"""Tests for janus.generate.fidelity, both the per-feature distance
metrics (ported near-unchanged) and the new Fidelity Scorecard additions
(correlation delta, distinguisher AUC) that make JANUS's fidelity claim
measured rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.fidelity.distributional import compare_feature
from janus.generate.fidelity.scorer import correlation_delta, distinguisher_auc, score_batch


class TestCompareFeature:
    def test_identical_distributions_score_near_zero(self):
        rng = np.random.RandomState(0)
        real = rng.normal(0, 1, 2000)
        synthetic = real.copy()
        report = compare_feature(real, synthetic, "x")
        assert report.wasserstein < 1e-6
        assert report.js_divergence < 1e-6

    def test_completely_disjoint_distributions_score_high(self):
        rng = np.random.RandomState(0)
        real = rng.normal(0, 1, 2000)
        synthetic = rng.normal(1000, 1, 2000)
        report = compare_feature(real, synthetic, "x")
        assert report.wasserstein > 100
        assert report.ks_statistic > 0.9


class TestCorrelationDelta:
    def test_identical_correlation_structure_scores_zero(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 1000)
        y = x * 2 + rng.normal(0, 0.01, 1000)
        df = pd.DataFrame({"a": x, "b": y})
        result = correlation_delta(df, df.copy(), ["a", "b"])
        assert result.mean_abs_delta < 0.01

    def test_destroyed_correlation_structure_scores_high(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 1000)
        y_correlated = x * 2 + rng.normal(0, 0.01, 1000)
        real = pd.DataFrame({"a": x, "b": y_correlated})
        synthetic = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
        result = correlation_delta(real, synthetic, ["a", "b"])
        assert result.mean_abs_delta > 0.5


class TestDistinguisherAuc:
    def test_identical_populations_score_near_half(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
        result = distinguisher_auc(df, df.copy(), ["a", "b"], cv_folds=3)
        assert 0.35 <= result.auc <= 0.65

    def test_trivially_separable_populations_score_high(self):
        rng = np.random.RandomState(0)
        real = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
        synthetic = pd.DataFrame({"a": rng.normal(50, 1, 500), "b": rng.normal(-50, 1, 500)})
        result = distinguisher_auc(real, synthetic, ["a", "b"], cv_folds=3)
        assert result.auc > 0.9


class TestScoreBatch:
    def test_produces_full_scorecard(self):
        rng = np.random.RandomState(0)
        real = pd.DataFrame({"amount": rng.lognormal(3, 1, 300)})
        synthetic = pd.DataFrame({"amount": rng.lognormal(3, 1, 300)})
        card = score_batch("test_batch", real, synthetic, ["amount"])
        assert card.batch_name == "test_batch"
        assert len(card.feature_reports) == 1
        assert card.distinguisher is not None


# --- distinguisher robustness ----------------------------------------
# Two bugs that both make the headline fidelity number stop measuring the
# generator while still looking like a plausible score.

import numpy as np
import pandas as pd

from janus.generate.fidelity.scorer import distinguisher_auc
from janus.generate.tabular.synthesizer import GaussianCopulaSynthesizer


def _grouped_frame(seed: int = 0) -> pd.DataFrame:
    """Rows arriving grouped by segment, the way real data arrives when it
    is sorted by time or concatenated per merchant category."""

    rng = np.random.RandomState(seed)
    cols = [f"f{i}" for i in range(4)]
    return pd.concat(
        [pd.DataFrame(rng.normal(mean, 1.0, size=(400, 4)), columns=cols) for mean in (-3.0, 0.0, 3.0)],
        ignore_index=True,
    )


def test_distinguisher_does_not_score_below_chance_on_grouped_input():
    """An unshuffled CV split trains on some segments and tests on others,
    so the classifier scores the held-out segment backwards and returns an
    AUC below 0.5, which reads as better-than-perfect fidelity and
    actually means the metric has stopped working. Identical frames must
    land at chance, not under it."""

    frame = _grouped_frame()
    result = distinguisher_auc(frame, frame.copy(), list(frame.columns))
    assert 0.4 <= result.auc <= 0.6


def test_distinguisher_still_separates_obviously_different_data():
    real = _grouped_frame()
    shifted = real + 8.0
    assert distinguisher_auc(real, shifted, list(real.columns)).auc > 0.9


def test_copula_mixture_recovers_multimodal_joint_structure():
    """One Gaussian copula has one correlation matrix, so a population with
    several differently-correlated modes gets averaged into a smeared
    middle, marginals and pooled correlation both still look right, which
    is why only the distinguisher catches it."""

    rng = np.random.RandomState(1)
    cols = [f"f{i}" for i in range(5)]
    parts = []
    for mean, rho in ((-3.0, 0.85), (3.0, -0.6)):
        cov = np.full((5, 5), rho)
        np.fill_diagonal(cov, 1.0)
        cov = cov @ cov.T / 5 + np.eye(5) * 0.4
        parts.append(pd.DataFrame(rng.multivariate_normal(np.full(5, mean), cov, size=1200), columns=cols))
    real = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=2).reset_index(drop=True)

    def corr_delta(synth: pd.DataFrame) -> float:
        delta = np.abs(real[cols].corr().to_numpy() - synth[cols].corr().to_numpy())
        return float(delta[~np.eye(len(cols), dtype=bool)].mean())

    single = GaussianCopulaSynthesizer(n_components=1).fit(real, cols).sample(len(real), random_state=3)
    mixture = GaussianCopulaSynthesizer(n_components=4).fit(real, cols).sample(len(real), random_state=3)

    assert corr_delta(mixture) < corr_delta(single)
    assert distinguisher_auc(real, mixture, cols).auc < distinguisher_auc(real, single, cols).auc


def test_mixture_falls_back_to_a_single_component_on_small_input():
    rng = np.random.RandomState(4)
    cols = ["a", "b"]
    tiny = pd.DataFrame(rng.normal(size=(30, 2)), columns=cols)
    synth = GaussianCopulaSynthesizer(n_components=8).fit(tiny, cols).sample(50, random_state=5)
    assert len(synth) == 50
    assert list(synth.columns) == cols
