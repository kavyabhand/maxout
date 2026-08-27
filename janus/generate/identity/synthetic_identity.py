"""Category A, attack A1 (Synthetic Identity Fraud 2.0); the onboarding
surface, which until now was the one taxonomy category with no generator
and no detector behind it at all.

What is actually being simulated, and what is not: this does NOT generate
face images or ID documents. Defeating a liveness check is a computer-
vision problem that would eat the whole build and produce a result no
payment risk team could act on. What a payment network *can* act on, and
what this models, is the APPLICATION-LEVEL fingerprint a GenAI-assembled
identity leaves at account opening; the part that reaches a risk engine
as structured data regardless of how convincing the selfie was:

  - an implausibly complete, internally consistent profile attached to a
    credit file that barely exists (the defining synthetic-identity tell:
    the persona is well-formed, the history is not),
  - device fingerprint and IP subnet reuse across applications, because
    scale is the entire economic point of generating identities with a
    model,
  - machine-speed form completion and low input-correction counts,
  - disposable/VoIP contact rails and freshly-registered email domains.

THE HARD-NEGATIVE POPULATION IS THE WHOLE POINT. A generator that emits
"fraud = thin credit file" against a control group of established
customers produces a detector with a near-perfect score and no meaning:
it has learned to detect being new. Real thin-file applicants, young
adults opening a first account, recent immigrants with no domestic credit
history, people rebuilding after a life event, look identical on exactly
the axis that separates the naive version, and they are precisely the
population a bad model debanks. So the legitimate population here
deliberately includes them, plus genuine shared-device households and
shared public IPs, and `evaluate_thin_file_fairness` reports the false-
positive rate on that subgroup separately from the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from janus.common.metrics import EvalReport, evaluate

FEATURE_COLS = [
    "profile_completeness",
    "credit_file_age_months",
    "n_tradelines",
    "applications_per_device_30d",
    "applications_per_ip_subnet_24h",
    "address_reuse_count",
    "email_domain_age_days",
    "is_voip_phone",
    "form_dwell_seconds",
    "form_corrections",
    "identity_element_consistency",
    "declared_income_to_segment_median",
]


@dataclass
class OnboardingPopulation:
    applications: pd.DataFrame
    n_legit: int
    n_thin_file_legit: int
    n_synthetic: int
    n_rings: int

    def as_dict(self) -> dict:
        return {
            "n_applications": len(self.applications),
            "n_legit": self.n_legit,
            "n_thin_file_legit": self.n_thin_file_legit,
            "n_synthetic": self.n_synthetic,
            "n_rings": self.n_rings,
            "synthetic_rate": round(self.n_synthetic / max(len(self.applications), 1), 6),
        }


def _established_legit(rng: np.random.RandomState, n: int) -> pd.DataFrame:
    """Ordinary applicants with real history behind them."""

    return pd.DataFrame({
        "profile_completeness": np.clip(rng.beta(5, 3, n), 0, 1),
        "credit_file_age_months": rng.gamma(shape=3.0, scale=40.0, size=n).clip(6, 600),
        "n_tradelines": rng.poisson(5.0, n).clip(1, 40),
        "applications_per_device_30d": rng.choice([1, 1, 1, 1, 2, 3], size=n),
        "applications_per_ip_subnet_24h": rng.choice([1, 1, 1, 2, 2, 4], size=n),
        "address_reuse_count": rng.choice([1, 1, 1, 2, 3], size=n),
        "email_domain_age_days": rng.gamma(shape=4.0, scale=1400.0, size=n).clip(30, 12000),
        "is_voip_phone": (rng.uniform(size=n) < 0.04).astype(int),
        "form_dwell_seconds": rng.gamma(shape=6.0, scale=35.0, size=n).clip(25, 1800),
        "form_corrections": rng.poisson(3.5, n),
        "identity_element_consistency": np.clip(rng.beta(7, 2, n), 0, 1),
        "declared_income_to_segment_median": rng.lognormal(0.0, 0.35, n).clip(0.2, 6.0),
        "segment": "established",
        "is_synthetic": 0,
    })


def _thin_file_legit(rng: np.random.RandomState, n: int) -> pd.DataFrame:
    """The hard negatives. Genuine applicants who share the synthetic
    population's headline signature (little or no credit history) and
    differ only in the behavioural and infrastructure signals. A detector
    that cannot separate these two groups is a detector that declines
    first-time applicants for being first-time applicants."""

    return pd.DataFrame({
        # A real first-time applicant often leaves optional fields blank;
        # a generated persona rarely does, because a model has no reason
        # to withhold anything. This is the separating axis that survives.
        "profile_completeness": np.clip(rng.beta(3, 4, n), 0, 1),
        "credit_file_age_months": rng.gamma(shape=1.4, scale=5.0, size=n).clip(0, 30),
        "n_tradelines": rng.poisson(0.7, n).clip(0, 4),
        "applications_per_device_30d": rng.choice([1, 1, 1, 2, 3, 5], size=n),  # shared household devices
        "applications_per_ip_subnet_24h": rng.choice([1, 1, 2, 3, 6, 12], size=n),  # campus / public wifi / CGNAT
        "address_reuse_count": rng.choice([1, 1, 2, 4, 7], size=n),  # dorms, shared housing
        "email_domain_age_days": rng.gamma(shape=3.0, scale=900.0, size=n).clip(20, 9000),
        "is_voip_phone": (rng.uniform(size=n) < 0.11).astype(int),  # prepaid/VoIP is common here
        "form_dwell_seconds": rng.gamma(shape=5.0, scale=45.0, size=n).clip(30, 2400),
        "form_corrections": rng.poisson(4.5, n),
        "identity_element_consistency": np.clip(rng.beta(5, 3, n), 0, 1),
        "declared_income_to_segment_median": rng.lognormal(-0.35, 0.4, n).clip(0.1, 4.0),
        "segment": "thin_file_legit",
        "is_synthetic": 0,
    })


#: How much infrastructure a ring is willing to pay for. This is the axis
#: that actually decides whether the ring is catchable, and collapsing it
#: into one "fraud" distribution is what makes a synthetic-identity
#: benchmark trivially separable and therefore worthless.
RING_SOPHISTICATION = ("cheap", "moderate", "advanced")


def _synthetic_ring(rng: np.random.RandomState, n: int, n_rings: int) -> pd.DataFrame:
    """GenAI-assembled identities, submitted in rings of varying
    sophistication.

    A first version of this generator drew every ring from one distribution
    with fresh throwaway email domains, heavy device/IP reuse and
    machine-fast form fills. The detector scored a clean 1.000 on every
    metric, which is not a good result, it is the sign of a benchmark that
    separates on a single near-disjoint feature. Real rings buy their way
    out of exactly those tells, and cheaply: aged mailboxes on mainstream
    domains are a commodity purchase, residential proxy pools drop IP reuse
    to ~1, anti-detect browsers mint a fresh fingerprint per session, and
    replaying recorded human keystroke timings defeats a dwell-time
    threshold.

    So sophistication is an explicit per-ring property here:

      cheap     one device, one subnet, throwaway domains, bot-speed fill.
                Caught by infrastructure features alone.
      moderate  partial rotation, mixed domain age, some pacing.
      advanced  fresh device and residential IP per application, aged
                mainstream mailboxes, replayed human timing. Every
                infrastructure tell is gone by construction; what remains
                is only the statistical signature of a fabricated persona: a profile too complete and too internally consistent for a
                credit file that thin, with a declared income the rest of
                the application does not support.

    Reporting one aggregate number over this mix would hide the finding
    that matters, so `train_onboarding_detector` reports recall per tier.
    """

    tier_weights = np.array([0.45, 0.35, 0.20])
    ring_tier = rng.choice(len(RING_SOPHISTICATION), size=max(n_rings, 1), p=tier_weights)
    ring_id = rng.randint(0, max(n_rings, 1), size=n)
    tier = ring_tier[ring_id]

    # Per-tier infrastructure reuse. Advanced rings sit inside the legitimate
    # shared-wifi/household range and are not separable on these at all.
    device_load = np.where(tier == 0, rng.gamma(5.0, 4.0, n), np.where(tier == 1, rng.gamma(2.0, 2.0, n), rng.gamma(1.1, 1.1, n)))
    ip_load = np.where(tier == 0, rng.gamma(6.0, 5.0, n), np.where(tier == 1, rng.gamma(2.5, 2.5, n), rng.gamma(1.2, 1.4, n)))
    addr_reuse = np.where(tier == 0, rng.gamma(4.0, 2.5, n), np.where(tier == 1, rng.gamma(2.0, 1.5, n), rng.gamma(1.2, 1.2, n)))

    # Mailbox age: cheap rings burn fresh domains, advanced rings buy aged
    # mainstream accounts whose age distribution matches ordinary users'.
    domain_age = np.where(
        tier == 0, rng.gamma(1.5, 60.0, n),
        np.where(tier == 1, rng.gamma(2.5, 500.0, n), rng.gamma(4.0, 1300.0, n)),
    )
    voip_rate = np.where(tier == 0, 0.70, np.where(tier == 1, 0.35, 0.07))

    # Form pacing: replayed human timings for advanced rings, including the
    # correction noise a naive automation script never produces.
    dwell = np.where(
        tier == 0, rng.gamma(2.0, 18.0, n),
        np.where(tier == 1, rng.gamma(4.0, 30.0, n), rng.gamma(6.0, 38.0, n)),
    )
    corrections = np.where(tier == 0, rng.poisson(0.6, n), np.where(tier == 1, rng.poisson(2.2, n), rng.poisson(3.9, n)))

    # The residual signal that survives every tier: a fabricated persona is
    # coherent and complete by construction. Advanced rings deliberately
    # blank some optional fields to blend in, which narrows but does not
    # close the gap against genuine thin-file applicants.
    completeness = np.where(
        tier == 0, rng.beta(14, 1.5, n),
        np.where(tier == 1, rng.beta(8, 2.0, n), rng.beta(5.0, 2.6, n)),
    )
    consistency = np.where(
        tier == 0, rng.beta(16, 1.4, n),
        np.where(tier == 1, rng.beta(9, 2.0, n), rng.beta(6.0, 2.4, n)),
    )

    frame = pd.DataFrame({
        "profile_completeness": np.clip(completeness, 0, 1),
        "credit_file_age_months": rng.gamma(shape=1.3, scale=5.0, size=n).clip(0, 34),
        "n_tradelines": rng.poisson(0.8, n).clip(0, 6),
        "applications_per_device_30d": np.round(device_load).clip(1, None),
        "applications_per_ip_subnet_24h": np.round(ip_load).clip(1, None),
        "address_reuse_count": np.round(addr_reuse).clip(1, None),
        "email_domain_age_days": domain_age.clip(1, 12000),
        "is_voip_phone": (rng.uniform(size=n) < voip_rate).astype(int),
        "form_dwell_seconds": dwell.clip(8, 2400),
        "form_corrections": corrections,
        "identity_element_consistency": np.clip(consistency, 0, 1),
        "declared_income_to_segment_median": rng.lognormal(0.18, 0.34, n).clip(0.2, 6.0),
        "segment": "synthetic_identity",
        "is_synthetic": 1,
    })
    frame["ring_id"] = ring_id
    frame["ring_sophistication"] = [RING_SOPHISTICATION[t] for t in tier]
    return frame


def generate_onboarding_population(
    n_established: int = 40_000,
    n_thin_file: int = 8_000,
    n_synthetic: int = 900,
    n_rings: int = 45,
    seed: int = 17,
) -> OnboardingPopulation:
    """Default mix puts synthetic identities at ~1.8% of applications and
    makes genuine thin-file applicants a 16% minority, deliberately large
    enough that a detector cannot reach a good score by declining the whole
    thin-file segment."""

    rng = np.random.RandomState(seed)
    frames = [
        _established_legit(rng, n_established),
        _thin_file_legit(rng, n_thin_file),
        _synthetic_ring(rng, n_synthetic, n_rings),
    ]
    applications = pd.concat(frames, ignore_index=True)
    for col in ("ring_id", "ring_sophistication"):
        if col not in applications.columns:
            applications[col] = np.nan
    applications = applications.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return OnboardingPopulation(
        applications=applications,
        n_legit=n_established + n_thin_file,
        n_thin_file_legit=n_thin_file,
        n_synthetic=n_synthetic,
        n_rings=n_rings,
    )


@dataclass
class OnboardingDetectorResult:
    model: xgb.XGBClassifier
    eval_report: EvalReport
    thin_file_fpr: float
    established_fpr: float
    recall_by_ring_sophistication: dict[str, dict]
    feature_importance: dict[str, float]

    def as_dict(self) -> dict:
        return {
            **self.eval_report.as_dict(),
            "false_positive_rate_thin_file_legit": round(self.thin_file_fpr, 6),
            "false_positive_rate_established_legit": round(self.established_fpr, 6),
            "recall_by_ring_sophistication": self.recall_by_ring_sophistication,
            "feature_importance": self.feature_importance,
            "fairness_note": (
                "false_positive_rate_thin_file_legit is the rate at which genuine applicants with "
                "little or no credit history are flagged. It is reported separately because that "
                "subgroup shares the synthetic population's headline signature; a detector that "
                "scores well on the aggregate while failing here is declining people for being new."
            ),
            "sophistication_note": (
                "recall_by_ring_sophistication splits detection by how much infrastructure the ring "
                "rotates. Cheap rings are caught on device/IP/domain reuse alone; advanced rings have "
                "no infrastructure tell left and are only reachable through the statistical signature "
                "of a fabricated persona. The aggregate number alone would hide that gap."
            ),
        }


def train_onboarding_detector(applications: pd.DataFrame, *, test_size: float = 0.3, random_state: int = 42) -> OnboardingDetectorResult:
    x = applications[FEATURE_COLS]
    y = applications["is_synthetic"]
    segment = applications["segment"]

    sophistication = applications["ring_sophistication"]

    x_train, x_test, y_train, y_test, _, seg_test, _, soph_test = train_test_split(
        x, y, segment, sophistication, test_size=test_size, random_state=random_state, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_score = model.predict_proba(x_test)[:, 1]
    report = evaluate(y_test.to_numpy(), y_score)

    flagged = y_score >= report.threshold
    thin_mask = (seg_test == "thin_file_legit").to_numpy()
    est_mask = (seg_test == "established").to_numpy()

    recall_by_tier = {}
    for tier in RING_SOPHISTICATION:
        tier_mask = (soph_test == tier).to_numpy() & (y_test == 1).to_numpy()
        n_tier = int(tier_mask.sum())
        recall_by_tier[tier] = {
            "n": n_tier,
            "recall": round(float(flagged[tier_mask].mean()), 6) if n_tier else None,
        }

    return OnboardingDetectorResult(
        model=model,
        eval_report=report,
        recall_by_ring_sophistication=recall_by_tier,
        thin_file_fpr=float(flagged[thin_mask].mean()) if thin_mask.any() else 0.0,
        established_fpr=float(flagged[est_mask].mean()) if est_mask.any() else 0.0,
        feature_importance={
            name: round(float(score), 5)
            for name, score in sorted(
                zip(FEATURE_COLS, model.feature_importances_), key=lambda t: t[1], reverse=True
            )
        },
    )
