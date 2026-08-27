"""The meta-learner, stacks per-family risk scores into one calibrated
decision, routed into four tiers mirroring real issuer logic:
auto-approve -> step-up (3-D-Secure-style challenge) -> hold for review ->
decline. Family-agnostic by design (fits on an arbitrary N-column score
matrix), because the five defend families run on genuinely different data
substrates in this pipeline (GBM/anomaly on ULB/IEEE tabular rows, the
sequence transformer on simulated entity histories, the NLP classifier on
catalog text); there is no single real dataset here where all five are
simultaneously available per-transaction.

This is run end-to-end by janus/orchestrate/ensemble.py, which trains the
members and this stack on three disjoint splits (members on train, this
stack on a held-out meta split, both evaluated on a test split neither has
seen) so the reported stacked numbers are not member-in-sample. The
resulting scored tier distribution is persisted to
data/processed/defend_meta_ensemble.json.

Latency framing: fits the fast path (GBM + deterministic hard rules) to
comfortably clear a realistic sub-300ms authorization SLA; the heavier
members (GNN, sequence transformer) run async, with the fast path's tier
as the interim decision and the heavier score attached for case management
/ step-up-auth follow-up rather than blocking the authorization itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from janus.common.metrics import EvalReport, evaluate

DECISION_TIERS = ["auto_approve", "step_up", "review", "decline"]


@dataclass
class TierThresholds:
    """Score cuts between the four tiers.

    The defaults are fixed probability cuts, which is the principled
    version: a calibrated score of 0.85 means an 85% chance this is fraud,
    and declining on that is defensible to a regulator in a way that
    declining "the top 0.2% of today's volume" is not.

    They are not, however, how capacity gets planned. A review queue has a
    fixed number of analysts, and an issuer that routes 3% of volume to it
    on a busy day has a broken control regardless of how well-calibrated
    the score was. `from_volume_targets` produces the other view: cuts
    placed at the score quantiles that hit a target share of traffic per
    tier. Both are reported, because they answer different questions and
    each one alone is misleading, fixed cuts can leave a tier empty (if
    the model is never that confident, nothing is ever auto-declined), and
    volume cuts always fill every tier whether or not the model has any
    confidence to justify it."""

    step_up: float = 0.3
    review: float = 0.6
    decline: float = 0.85

    @classmethod
    def from_volume_targets(
        cls,
        risk_scores: np.ndarray,
        *,
        decline_share: float = 0.002,
        review_share: float = 0.01,
        step_up_share: float = 0.03,
    ) -> "TierThresholds":
        """Cuts placed so the top `decline_share` of scored volume declines,
        the next `review_share` goes to review, and the next
        `step_up_share` is challenged."""

        scores = np.asarray(risk_scores, dtype=float)
        if scores.size == 0:
            return cls()
        cumulative_review = decline_share + review_share
        cumulative_step_up = cumulative_review + step_up_share
        return cls(
            step_up=float(np.quantile(scores, max(0.0, 1 - cumulative_step_up))),
            review=float(np.quantile(scores, max(0.0, 1 - cumulative_review))),
            decline=float(np.quantile(scores, max(0.0, 1 - decline_share))),
        )


def decision_tier(risk_score: float, thresholds: TierThresholds = TierThresholds()) -> str:
    if risk_score >= thresholds.decline:
        return "decline"
    if risk_score >= thresholds.review:
        return "review"
    if risk_score >= thresholds.step_up:
        return "step_up"
    return "auto_approve"


@dataclass
class MetaLearnerResult:
    model: object
    eval_report: EvalReport
    member_names: list[str]
    member_weights: list[float]


def fit_meta_learner(score_matrix: np.ndarray, y: np.ndarray, member_names: list[str], *, random_state: int = 42) -> MetaLearnerResult:
    """score_matrix: (n_samples, n_members); each column a calibrated-ish
    [0,1] risk score from one defend family. Fits a calibrated logistic
    regression stack (Platt-scaled) so the combined output is itself a
    genuine probability, not just an arbitrary linear score."""

    base = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(score_matrix, y)

    combined_score = model.predict_proba(score_matrix)[:, 1]
    report = evaluate(y, combined_score)

    # member_weights: the underlying logistic regression's coefficients,
    # refit once more without calibration purely to surface interpretable
    # per-family weights for the UI (calibration itself is a monotonic
    # wrapper and doesn't change relative feature importance).
    plain = LogisticRegression(max_iter=1000).fit(score_matrix, y)
    weights = plain.coef_[0].tolist()

    return MetaLearnerResult(model=model, eval_report=report, member_names=member_names, member_weights=weights)


def latency_budget_report() -> dict:
    return {
        "fast_path_ms_budget": 300,
        "fast_path_members": ["hard_rules", "gbm"],
        "async_members": ["gnn", "sequence_transformer", "nlp_classifier", "anomaly"],
        "note": (
            "Fast path (deterministic hard rules + GBM) is designed to clear a realistic "
            "sub-300ms authorization SLA; heavier members run async and feed case management "
            "and step-up-auth follow-up rather than gating the authorization itself, mirroring "
            "how a real issuer risk stack layers an inline scorer with deeper offline models."
        ),
    }


@dataclass
class TierDistribution:
    """Where a scored population actually lands across the four tiers, and
    what fraction of the fraud each tier caught. This is the number an
    issuer cares about that a bare PR-AUC does not answer: a model can rank
    well and still route so much volume to `review` that no operations team
    could staff it."""

    counts: dict[str, int]
    fraud_counts: dict[str, int]
    n_total: int
    n_fraud: int
    thresholds: "TierThresholds | None" = None

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_fraud": self.n_fraud,
            "thresholds": (
                {
                    "step_up": round(self.thresholds.step_up, 6),
                    "review": round(self.thresholds.review, 6),
                    "decline": round(self.thresholds.decline, 6),
                }
                if self.thresholds
                else None
            ),
            "tiers": [
                {
                    "tier": tier,
                    "n": self.counts.get(tier, 0),
                    "share_of_volume": round(self.counts.get(tier, 0) / max(self.n_total, 1), 6),
                    "n_fraud": self.fraud_counts.get(tier, 0),
                    "share_of_fraud_caught": round(self.fraud_counts.get(tier, 0) / max(self.n_fraud, 1), 6),
                    # Precision within the tier, how much of what lands here is actually fraud.
                    "fraud_rate_within_tier": round(
                        self.fraud_counts.get(tier, 0) / max(self.counts.get(tier, 0), 1), 6
                    ),
                }
                for tier in DECISION_TIERS
            ],
        }


def tier_distribution(risk_scores: np.ndarray, y: np.ndarray, thresholds: TierThresholds = TierThresholds()) -> TierDistribution:
    counts = {t: 0 for t in DECISION_TIERS}
    fraud_counts = {t: 0 for t in DECISION_TIERS}
    for score_value, label in zip(risk_scores, y):
        tier = decision_tier(float(score_value), thresholds)
        counts[tier] += 1
        if int(label) == 1:
            fraud_counts[tier] += 1
    return TierDistribution(
        counts=counts, fraud_counts=fraud_counts,
        n_total=int(len(risk_scores)), n_fraud=int(np.asarray(y).sum()),
        thresholds=thresholds,
    )
