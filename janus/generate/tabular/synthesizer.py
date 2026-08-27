"""Conditional tabular synthesis (Categories A-C): a Gaussian-copula
generator fit directly against real ULB/IEEE marginals and correlation
structure, rather than an off-the-shelf GAN package. This is a real,
well-understood technique (the same modeling idea SDV's GaussianCopula
synthesizer implements): transform each feature to a standard-normal
marginal via its empirical CDF, fit a multivariate-normal correlation
structure on the transformed data, sample from that joint normal, then
invert each dimension back through its own empirical quantile function.
The result preserves both per-feature marginals (by construction, since
sampling always inverts through the real empirical distribution) and
pairwise correlation (via the fitted Gaussian copula) far better than
independent per-feature sampling would.

Fit separately per class (legit / fraud) and mixed at a configurable
ratio, so the generator is genuinely conditional on the label rather than
producing one undifferentiated population, and fraud can be injected in
temporal bursts rather than spread uniformly (see inject_bursts), real
card-testing and structuring fraud clusters in time, and a uniformly
spread synthetic fraud population would make a downstream detector's job
artificially easy in a way that overstates real-world performance.

One Gaussian copula is fitted per latent cluster rather than one over the
whole class, because a single copula has exactly one correlation matrix
and real payment populations do not. Cardholder segments, merchant
categories and channel mixes each carry their own dependence structure,
and averaging them produces a joint distribution that matches the
population's overall correlation while describing none of its actual
modes, a smeared middle. Per-feature marginals and pairwise correlations
both look right afterwards, which is why this failure survives the first
two fidelity metrics and is caught only by the distinguisher: a
gradient-boosted classifier finds the region between two real modes,
which is densely populated in the synthetic data and nearly empty in the
real data, immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


#: Minimum rows a cluster needs before it gets its own copula. Below this
#: the per-cluster correlation estimate is noise, and a noisy correlation
#: matrix produces worse samples than a pooled one.
MIN_CLUSTER_ROWS = 60


@dataclass
class _FittedMarginal:
    sorted_values: np.ndarray


@dataclass
class _Component:
    weight: float
    corr: np.ndarray
    marginals: dict[str, _FittedMarginal]


def _psd_correlation(z: np.ndarray) -> np.ndarray:
    """Correlation of the latent normals, projected back onto the positive
    semi-definite cone. Near-duplicate or constant columns can produce an
    estimate with small negative eigenvalues, which makes sampling fail."""

    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.ndim == 0:
        corr = corr.reshape(1, 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


class GaussianCopulaSynthesizer:
    """`n_components=1` is the plain single-copula generator; higher values
    fit one copula per k-means cluster and sample from the mixture in the
    real clusters' proportions."""

    def __init__(self, n_components: int = 1) -> None:
        self._components: list[_Component] = []
        self._feature_cols: list[str] = []
        self._n_components = max(1, n_components)

    def _fit_component(self, frame: pd.DataFrame, weight: float) -> _Component:
        n = len(frame)
        u = np.empty((n, len(self._feature_cols)))
        marginals: dict[str, _FittedMarginal] = {}
        for i, col in enumerate(self._feature_cols):
            values = frame[col].to_numpy(dtype=float)
            marginals[col] = _FittedMarginal(sorted_values=np.sort(values))
            # empirical CDF rank -> uniform(0,1), clipped away from the
            # boundary so norm.ppf doesn't blow up to +/-inf
            ranks = pd.Series(values).rank(method="average").to_numpy()
            u[:, i] = np.clip(ranks / (n + 1), 1e-6, 1 - 1e-6)
        z = np.nan_to_num(norm.ppf(u), nan=0.0, posinf=0.0, neginf=0.0)
        return _Component(weight=weight, corr=_psd_correlation(z), marginals=marginals)

    def fit(self, df: pd.DataFrame, feature_cols: list[str], *, random_state: int = 42) -> "GaussianCopulaSynthesizer":
        self._feature_cols = list(feature_cols)
        self._components = []

        k = min(self._n_components, max(1, len(df) // MIN_CLUSTER_ROWS))
        if k <= 1:
            self._components = [self._fit_component(df, 1.0)]
            return self

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(df[feature_cols].to_numpy(dtype=float))
        labels = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit_predict(scaled)

        for label in range(k):
            member = df[labels == label]
            if len(member) < MIN_CLUSTER_ROWS:
                # Fold an undersized cluster into the pooled fit rather than
                # fitting a correlation matrix on too few rows.
                continue
            self._components.append(self._fit_component(member, len(member) / len(df)))

        if not self._components:
            self._components = [self._fit_component(df, 1.0)]
        total = sum(c.weight for c in self._components)
        for c in self._components:
            c.weight /= total
        return self

    def sample(self, n: int, *, random_state: int = 42) -> pd.DataFrame:
        if not self._components:
            raise RuntimeError("call fit() before sample()")
        rng = np.random.RandomState(random_state)

        weights = np.array([c.weight for c in self._components])
        counts = rng.multinomial(n, weights)
        frames = [
            self._sample_component(component, int(count), rng)
            for component, count in zip(self._components, counts)
            if count > 0
        ]
        combined = pd.concat(frames, ignore_index=True)
        return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    def _sample_component(self, component: _Component, n: int, rng: np.random.RandomState) -> pd.DataFrame:
        z = rng.multivariate_normal(mean=np.zeros(len(self._feature_cols)), cov=component.corr, size=n)
        u = norm.cdf(z)

        out = {}
        for i, col in enumerate(self._feature_cols):
            sorted_vals = component.marginals[col].sorted_values
            # Linear interpolation between order statistics rather than a
            # nearest-index lookup. The index version could only ever emit
            # values that appear verbatim in the training data, which makes
            # a synthetic row detectable by exact-value membership alone --
            # a tell that has nothing to do with whether the distribution is
            # right, and one a distinguisher will happily find.
            positions = u[:, i] * (len(sorted_vals) - 1)
            lower = np.floor(positions).astype(int)
            upper = np.clip(lower + 1, 0, len(sorted_vals) - 1)
            frac = positions - lower
            out[col] = sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
        return pd.DataFrame(out)


@dataclass
class SynthesisResult:
    data: pd.DataFrame
    n_legit: int
    n_fraud: int
    fraud_ratio: float


def synthesize_conditional(
    real_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    n_rows: int,
    fraud_ratio: float,
    n_components: int | dict[int, int] = 8,
    random_state: int = 42,
) -> SynthesisResult:
    """Fits one GaussianCopulaSynthesizer per class and samples at the
    requested fraud_ratio, genuinely conditional generation, not a single
    unconditional population with labels attached after the fact.

    `n_components` is the mixture size within each class, either one value
    for both or a {label: k} mapping. Per-class is usually what you want:
    ULB's two classes differ by three orders of magnitude in row count, so
    the size that resolves the legitimate population's modes estimates the
    fraud class's correlation matrices from far too few rows."""

    legit_df = real_df[real_df[label_col] == 0]
    fraud_df = real_df[real_df[label_col] == 1]

    n_fraud = max(1, int(round(n_rows * fraud_ratio)))
    n_legit = n_rows - n_fraud

    components = n_components if isinstance(n_components, dict) else {0: n_components, 1: n_components}

    legit_synth = (
        GaussianCopulaSynthesizer(n_components=components.get(0, 1))
        .fit(legit_df, feature_cols, random_state=random_state)
        .sample(n_legit, random_state=random_state)
    )
    fraud_synth = (
        GaussianCopulaSynthesizer(n_components=components.get(1, 1))
        .fit(fraud_df, feature_cols, random_state=random_state)
        .sample(n_fraud, random_state=random_state + 1)
    )

    legit_synth[label_col] = 0
    fraud_synth[label_col] = 1
    combined = pd.concat([legit_synth, fraud_synth], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return SynthesisResult(data=combined, n_legit=n_legit, n_fraud=n_fraud, fraud_ratio=fraud_ratio)


def inject_bursts(df: pd.DataFrame, label_col: str, time_col: str, *, n_bursts: int, burst_window: int, max_step: int, random_state: int = 42) -> pd.DataFrame:
    """Reassigns the fraud-labeled rows' time_col into a handful of tight
    windows instead of leaving them uniformly spread, real card-testing
    and structuring fraud clusters in time; a uniformly-spread synthetic
    fraud population understates how bursty real fraud looks, which would
    make a downstream detector's job artificially easy."""

    rng = np.random.RandomState(random_state)
    df = df.copy()
    fraud_idx = df.index[df[label_col] == 1]
    if len(fraud_idx) == 0:
        return df

    burst_starts = rng.randint(0, max(1, max_step - burst_window), size=n_bursts)
    assigned_burst = rng.randint(0, n_bursts, size=len(fraud_idx))
    offsets = rng.randint(0, burst_window, size=len(fraud_idx))
    new_times = burst_starts[assigned_burst] + offsets
    df.loc[fraud_idx, time_col] = new_times
    return df


@dataclass
class ComponentSelection:
    n_components: int
    trials: list[dict]

    def as_dict(self) -> dict:
        return {"selected_n_components": self.n_components, "trials": self.trials}


def select_n_components(
    real_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    random_state: int = 0,
) -> ComponentSelection:
    """Chooses the mixture size by measuring it rather than assuming it.

    More components is not monotonically better. Each additional component
    buys a correlation matrix that describes one mode more precisely and
    pays for it by estimating that matrix from fewer rows, and past the
    point where a cluster has enough rows per feature the estimate is
    noise. Where that crossover sits depends on how many rows the class
    has and how many features are being modelled, ULB's fraud class is
    492 rows, its legitimate class is 284,315, so a single hard-coded
    number would be right for at most one of them.

    IMPORTANT; the selection is deliberately kept off the reported score.
    Picking the k that minimises the distinguisher AUC and then publishing
    that same AUC is selection bias: the reported number would be the best
    of five draws rather than an unbiased estimate. So selection fits on
    one half of the real data and scores against the other half with its
    own seed; the caller then refits on everything and re-scores with fresh
    seeds, and it is that independent number that gets reported.
    """

    from janus.generate.fidelity.scorer import distinguisher_auc

    rng = np.random.RandomState(random_state)
    shuffled = real_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    half = len(shuffled) // 2
    fit_half, score_half = shuffled.iloc[:half], shuffled.iloc[half:]
    n_draw = min(len(score_half), 4000)

    trials: list[dict] = []
    best_k, best_gap = candidates[0], float("inf")
    for k in candidates:
        synth = (
            GaussianCopulaSynthesizer(n_components=k)
            .fit(fit_half, feature_cols, random_state=random_state)
            .sample(n_draw, random_state=int(rng.randint(1_000_000)))
        )
        auc = distinguisher_auc(score_half.sample(n_draw, random_state=random_state), synth, feature_cols).auc
        gap = abs(auc - 0.5)
        trials.append({"n_components": k, "selection_auc": round(auc, 4), "distance_from_ideal": round(gap, 4)})
        if gap < best_gap:
            best_k, best_gap = k, gap

    return ComponentSelection(n_components=best_k, trials=trials)
