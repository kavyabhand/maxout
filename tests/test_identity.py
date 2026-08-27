"""Category A onboarding generator and detector.

The property worth guarding here is not that the detector scores well --
it is that the generator does not make the problem trivial. An earlier
version of this generator produced a clean 1.000 across every metric,
which looked like a strong result and was actually the signature of two
populations separable on a single near-disjoint feature. The sophistication
gradient and the thin-file control group are what fixed that, so both are
asserted rather than assumed.
"""

import pytest

from janus.generate.identity.synthetic_identity import (
    FEATURE_COLS,
    RING_SOPHISTICATION,
    generate_onboarding_population,
    train_onboarding_detector,
)


@pytest.fixture(scope="module")
def population():
    return generate_onboarding_population(n_established=6_000, n_thin_file=1_500, n_synthetic=400, n_rings=30, seed=5)


@pytest.fixture(scope="module")
def detector(population):
    return train_onboarding_detector(population.applications)


def test_population_contains_the_hard_negative_segment(population):
    segments = set(population.applications["segment"])
    assert segments == {"established", "thin_file_legit", "synthetic_identity"}
    assert population.n_thin_file_legit > 0


def test_every_sophistication_tier_is_represented(population):
    synthetic = population.applications[population.applications["is_synthetic"] == 1]
    assert set(synthetic["ring_sophistication"]) == set(RING_SOPHISTICATION)


def test_all_feature_columns_present_and_finite(population):
    frame = population.applications[FEATURE_COLS]
    assert not frame.isna().any().any()
    assert frame.shape[1] == len(FEATURE_COLS)


def test_detection_degrades_with_ring_sophistication(detector):
    """The generator's central claim: a ring that rotates its
    infrastructure is harder to catch than one that does not. If this
    inverts or flattens, the tiers have stopped meaning anything."""

    by_tier = detector.recall_by_ring_sophistication
    assert by_tier["cheap"]["recall"] > by_tier["advanced"]["recall"]


def test_thin_file_applicants_are_not_swept_up(detector):
    """Genuine thin-file applicants share the fraud population's headline
    signature, so they are the population a lazy detector debanks."""

    assert detector.thin_file_fpr < 0.05


def test_result_is_not_degenerate(detector):
    # A perfect score across the board means the benchmark separated on
    # one disjoint feature, not that the detector is good.
    assert detector.eval_report.pr_auc < 1.0
