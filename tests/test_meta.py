"""Tests for janus.defend.meta; the decision-tier mapping (a fixed,
non-learned function of the calibrated risk score) and the meta-learner
stacking fit, which must actually improve on (or at least match) its best
individual member on a case where one member is clearly informative and
the other is noise.
"""

from __future__ import annotations

import numpy as np

from janus.defend.meta import TierThresholds, decision_tier, fit_meta_learner


class TestDecisionTier:
    def test_low_score_auto_approves(self):
        assert decision_tier(0.05) == "auto_approve"

    def test_boundaries_are_inclusive_on_the_upper_tier(self):
        t = TierThresholds()
        assert decision_tier(t.step_up) == "step_up"
        assert decision_tier(t.review) == "review"
        assert decision_tier(t.decline) == "decline"

    def test_high_score_declines(self):
        assert decision_tier(0.99) == "decline"

    def test_tiers_are_monotonic_in_score(self):
        order = ["auto_approve", "step_up", "review", "decline"]
        scores = [0.0, 0.31, 0.61, 0.9]
        tiers = [decision_tier(s) for s in scores]
        assert [order.index(t) for t in tiers] == sorted(order.index(t) for t in tiers)


class TestFitMetaLearner:
    def test_stacking_an_informative_and_a_noise_member_beats_chance(self):
        rng = np.random.RandomState(0)
        n = 1000
        y = rng.randint(0, 2, n)
        informative_score = np.clip(y + rng.normal(0, 0.2, n), 0, 1)
        noise_score = rng.uniform(0, 1, n)
        matrix = np.stack([informative_score, noise_score], axis=1)

        result = fit_meta_learner(matrix, y, member_names=["informative", "noise"])
        assert result.eval_report.pr_auc > 0.7
        assert len(result.member_weights) == 2


# --- tier distribution -----------------------------------------------
# The four decision tiers were a design explainer with nothing measured
# behind them until janus/orchestrate/ensemble.py ran the stack on data.
# These cover the reporting side of that: an issuer's question is not
# "what is the PR-AUC" but "how much volume lands in review, and how much
# of the fraud does declining actually catch".

import numpy as np

from janus.defend.meta import TierThresholds, tier_distribution


def test_tier_distribution_partitions_the_population():
    scores = np.array([0.05, 0.35, 0.65, 0.95, 0.99])
    y = np.array([0, 0, 1, 1, 1])
    dist = tier_distribution(scores, y)

    assert dist.n_total == 5
    assert sum(dist.counts.values()) == 5
    assert dist.counts["auto_approve"] == 1
    assert dist.counts["step_up"] == 1
    assert dist.counts["review"] == 1
    assert dist.counts["decline"] == 2


def test_tier_distribution_tracks_fraud_per_tier():
    scores = np.array([0.05, 0.95, 0.99])
    y = np.array([0, 1, 1])
    payload = tier_distribution(scores, y).as_dict()

    decline = next(row for row in payload["tiers"] if row["tier"] == "decline")
    assert decline["n_fraud"] == 2
    assert decline["share_of_fraud_caught"] == 1.0
    assert decline["fraud_rate_within_tier"] == 1.0


def test_shares_are_zero_safe_on_an_empty_tier():
    scores = np.array([0.01, 0.02])
    payload = tier_distribution(scores, np.array([0, 0])).as_dict()

    decline = next(row for row in payload["tiers"] if row["tier"] == "decline")
    assert decline["n"] == 0
    assert decline["share_of_volume"] == 0.0
    assert decline["fraud_rate_within_tier"] == 0.0


def test_volume_targets_place_cuts_at_the_requested_quantiles():
    """Capacity planning, not probability: the top 0.2% of scored volume
    should decline regardless of what the calibrated probability there
    happens to be."""

    scores = np.linspace(0.0, 1.0, 10_000)
    thresholds = TierThresholds.from_volume_targets(
        scores, decline_share=0.01, review_share=0.04, step_up_share=0.05
    )
    dist = tier_distribution(scores, np.zeros(len(scores), dtype=int), thresholds)

    assert abs(dist.counts["decline"] / len(scores) - 0.01) < 0.005
    assert abs(dist.counts["review"] / len(scores) - 0.04) < 0.005
    assert abs(dist.counts["step_up"] / len(scores) - 0.05) < 0.005


def test_fixed_thresholds_can_leave_a_tier_empty_and_that_is_reported():
    """A model whose calibrated score never reaches the decline cut should
    show an empty decline tier rather than have the cut quietly moved --
    that emptiness is a real property of the model."""

    scores = np.linspace(0.0, 0.5, 500)
    dist = tier_distribution(scores, np.zeros(500, dtype=int))
    assert dist.counts["decline"] == 0
    assert dist.counts["review"] == 0


def test_thresholds_are_reported_alongside_the_distribution():
    payload = tier_distribution(np.array([0.1, 0.9]), np.array([0, 1])).as_dict()
    assert payload["thresholds"]["decline"] == 0.85


def test_volume_targets_on_empty_input_fall_back_to_defaults():
    assert TierThresholds.from_volume_targets(np.array([])).decline == TierThresholds().decline
