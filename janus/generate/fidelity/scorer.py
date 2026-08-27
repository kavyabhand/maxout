"""The Fidelity Scorecard, JANUS treats "fidelity of simulation" as a
measured, displayed number, not an asserted claim (RESEARCH.md §4.2).

Four independent signals, each catching a different failure mode a
generator can have:

- Per-feature distributional distance (distributional.compare_feature), are individual feature marginals close to real data?
- Correlation-matrix delta, do features move together the way real ones
  do? A generator can nail every marginal while destroying the joint
  structure (e.g. amount and merchant-category independence that isn't
  real); this is the check that would catch that.
- Real-vs-synthetic distinguisher AUC, train a classifier to tell real
  and synthetic rows apart. AUC -> 0.5 means it can't, which is the
  strongest single fidelity signal because it's sensitive to whatever the
  first two metrics didn't think to check.
- Graph-topology similarity (for the mule-ring generator), degree
  distribution and clustering-coefficient distance between synthetic ring
  topology and literature-documented real mule-ring shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from janus.generate.fidelity.distributional import FidelityReport, compare_feature


@dataclass
class CorrelationDelta:
    mean_abs_delta: float
    max_abs_delta: float
    feature_names: list[str]

    def as_dict(self) -> dict:
        return {
            "mean_abs_delta": round(self.mean_abs_delta, 4),
            "max_abs_delta": round(self.max_abs_delta, 4),
            "n_features": len(self.feature_names),
        }


@dataclass
class DistinguisherResult:
    auc: float
    n_real: int
    n_synthetic: int
    #: Rows whose exact feature vector appeared on both sides and were
    #: therefore excluded from the test. See distinguisher_auc.
    n_shared_rows_excluded: int = 0

    def as_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            # distance from the ideal (indistinguishable) AUC of 0.5, so
            # "closer to zero is better" reads consistently with the other
            # fidelity numbers on this scorecard.
            "distance_from_ideal": round(abs(self.auc - 0.5), 4),
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "n_shared_rows_excluded": self.n_shared_rows_excluded,
        }


@dataclass
class GraphTopologyResult:
    real_degree_ks: float
    real_clustering_delta: float

    def as_dict(self) -> dict:
        return {
            "degree_distribution_ks": round(self.real_degree_ks, 4),
            "clustering_coefficient_delta": round(self.real_clustering_delta, 4),
        }


@dataclass
class FidelityScorecard:
    batch_name: str
    feature_reports: list[FidelityReport]
    correlation: CorrelationDelta | None
    distinguisher: DistinguisherResult | None
    graph_topology: GraphTopologyResult | None = None

    def as_dict(self) -> dict:
        return {
            "batch_name": self.batch_name,
            "features": [r.as_dict() for r in self.feature_reports],
            "correlation": self.correlation.as_dict() if self.correlation else None,
            "distinguisher": self.distinguisher.as_dict() if self.distinguisher else None,
            "graph_topology": self.graph_topology.as_dict() if self.graph_topology else None,
        }


def correlation_delta(real: pd.DataFrame, synthetic: pd.DataFrame, feature_names: list[str]) -> CorrelationDelta:
    real_corr = real[feature_names].corr().to_numpy()
    synth_corr = synthetic[feature_names].corr().to_numpy()
    delta = np.abs(real_corr - synth_corr)
    # exclude the diagonal (always 0 delta, would deflate the mean)
    off_diag = delta[~np.eye(delta.shape[0], dtype=bool)]
    return CorrelationDelta(
        mean_abs_delta=float(np.nanmean(off_diag)) if off_diag.size else 0.0,
        max_abs_delta=float(np.nanmax(off_diag)) if off_diag.size else 0.0,
        feature_names=feature_names,
    )


def distinguisher_auc(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_names: list[str],
    *,
    random_state: int = 42,
    cv_folds: int = 3,
) -> DistinguisherResult:
    """A classifier that can't tell real from synthetic apart (AUC -> 0.5)
    is the strongest evidence of high fidelity, because unlike a fixed set
    of hand-chosen statistics, the classifier is free to find whatever
    signal actually separates the two distributions.

    Two things this has to get right, both of which silently produce a
    number that looks plausible and measures nothing:

    Exact duplicates across the two sets. A generator that resamples real
    values, the mule-ring and voice-scam amount generators bootstrap from
    real distributions, emits feature vectors that also appear verbatim in
    the real set. The classifier is then asked to give one identical vector
    two different labels; it memorizes whichever it saw in training and
    scores the held-out twin backwards. Every such pair contributes
    systematically WRONG ordering, which drags the AUC BELOW 0.5, and an AUC
    below 0.5 reads as better-than-indistinguishable when it actually means
    the metric has broken. Those rows carry no information about
    separability (they are shared mass, identical under both distributions) so they are excluded and counted rather than left to corrupt the
    score.

    Fold shuffling. `cv=3` defaults to StratifiedKFold WITHOUT shuffling,
    which balances the labels but preserves row order within each side. If
    the real frame arrives grouped, sorted by time, or concatenated per
    segment; each fold then trains on some segments and tests on others,
    and the classifier learns "this segment is real" rather than "real data
    looks like this".
    """

    real_x = real[feature_names].reset_index(drop=True)
    synth_x = synthetic[feature_names].reset_index(drop=True)

    real_keys = pd.MultiIndex.from_frame(real_x.fillna(0.0))
    synth_keys = pd.MultiIndex.from_frame(synth_x.fillna(0.0))
    shared = real_keys.intersection(synth_keys)

    n_excluded = 0
    if len(shared):
        real_mask = ~real_keys.isin(shared)
        synth_mask = ~synth_keys.isin(shared)
        n_excluded = int((~real_mask).sum() + (~synth_mask).sum())
        real_x = real_x[real_mask]
        synth_x = synth_x[synth_mask]

    # Two sets that overlap entirely are indistinguishable by definition,
    # and there is nothing left to fit, report chance rather than a
    # degenerate fit on a handful of survivors.
    if len(real_x) < cv_folds * 2 or len(synth_x) < cv_folds * 2:
        return DistinguisherResult(
            auc=0.5, n_real=len(real), n_synthetic=len(synthetic), n_shared_rows_excluded=n_excluded
        )

    real_x = real_x.copy()
    synth_x = synth_x.copy()
    real_x["__label__"] = 0
    synth_x["__label__"] = 1
    combined = pd.concat([real_x, synth_x], ignore_index=True).fillna(0.0)
    y = combined.pop("__label__").to_numpy()
    x = combined.to_numpy()

    clf = GradientBoostingClassifier(random_state=random_state, n_estimators=100, max_depth=3)
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, x, y, cv=folds, scoring="roc_auc")
    return DistinguisherResult(
        auc=float(scores.mean()),
        n_real=len(real),
        n_synthetic=len(synthetic),
        n_shared_rows_excluded=n_excluded,
    )


def graph_topology_similarity(real_graph: nx.Graph, synthetic_graph: nx.Graph) -> GraphTopologyResult:
    from scipy.stats import ks_2samp

    real_degrees = np.array([d for _, d in real_graph.degree()], dtype=float)
    synth_degrees = np.array([d for _, d in synthetic_graph.degree()], dtype=float)
    ks_stat, _ = ks_2samp(real_degrees, synth_degrees) if len(real_degrees) and len(synth_degrees) else (float("nan"), 1.0)

    real_clustering = nx.average_clustering(real_graph) if real_graph.number_of_nodes() else 0.0
    synth_clustering = nx.average_clustering(synthetic_graph) if synthetic_graph.number_of_nodes() else 0.0

    return GraphTopologyResult(
        real_degree_ks=float(ks_stat),
        real_clustering_delta=abs(real_clustering - synth_clustering),
    )


def score_batch(
    batch_name: str,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_names: list[str],
    *,
    real_graph: nx.Graph | None = None,
    synthetic_graph: nx.Graph | None = None,
) -> FidelityScorecard:
    feature_reports = [
        compare_feature(real[f].to_numpy(), synthetic[f].to_numpy(), f) for f in feature_names
    ]
    corr = correlation_delta(real, synthetic, feature_names) if len(feature_names) > 1 else None
    dist = distinguisher_auc(real, synthetic, feature_names)
    topo = (
        graph_topology_similarity(real_graph, synthetic_graph)
        if real_graph is not None and synthetic_graph is not None
        else None
    )
    return FidelityScorecard(
        batch_name=batch_name,
        feature_reports=feature_reports,
        correlation=corr,
        distinguisher=dist,
        graph_topology=topo,
    )
