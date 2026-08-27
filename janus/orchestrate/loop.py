"""The closed self-play loop; the competition brief's explicit
requirement made concrete rather than asserted in a slide. Two arms, both
real, both feeding the same headline curve (Attacker Win Rate declining /
Defense strength rising round over round):

- Agentic arm (run_agentic_closed_loop): Round 0 = naive shopping agent,
  no Mandate Firewall. Round 1+ = firewall present, `.learn()`-ed on the
  previous round's successful payloads before the next round runs, so the
  defense's own gaps become its next training signal, not just the
  attacker's.
- Tabular arm (run_tabular_adversarial_loop): wraps
  janus.generate.adversarial.evasion.iterative_harden; a black-box greedy
  evader attacks the GBM, evasive examples get folded into retraining,
  repeated for N rounds, each round's attacker adapting to the PREVIOUS
  round's hardened model.

combined_curve() reduces both arms to one comparable
(attacker_win_rate, defense_strength) pair per round so the UI's Red vs
Blue Arena can plot one curve even though the two arms measure genuinely
different things (an LLM/scripted red-team bypass rate vs. a black-box
evasion rate): documented as a normalized comparison, not literally the
same metric.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from janus.common.llm import ChatBackend, get_backend
from janus.defend.firewall import MandateFirewall
from janus.generate.agentic.red_team import TECHNIQUES, AttackAttempt, GenerationBlocked, run_attack_round


@dataclass
class TechniqueRoundStats:
    technique: str
    attempts: int
    successes: int
    incompletes: int
    bypass_rate: float
    successful_payloads: list[str] = field(default_factory=list)
    #: Attempts where the LLM provider's own safety layer refused to
    #: generate a payload, so the scripted template stood in.
    template_fallbacks: int = 0

    def as_dict(self) -> dict:
        return {
            "technique": self.technique, "attempts": self.attempts, "successes": self.successes,
            "incompletes": self.incompletes,
            "bypass_rate": None if self.bypass_rate != self.bypass_rate else round(self.bypass_rate, 4),
            "template_fallbacks": self.template_fallbacks,
            "successful_payloads": self.successful_payloads,
        }


@dataclass
class ClosedLoopRound:
    round_num: int
    firewall_present: bool
    detector_kind: str | None
    technique_stats: dict[str, TechniqueRoundStats]
    overall_bypass_rate: float

    def as_dict(self) -> dict:
        return {
            "round_num": self.round_num, "firewall_present": self.firewall_present, "detector_kind": self.detector_kind,
            "overall_bypass_rate": None if self.overall_bypass_rate != self.overall_bypass_rate else round(self.overall_bypass_rate, 4),
            "technique_stats": {k: v.as_dict() for k, v in self.technique_stats.items()},
        }


def run_agentic_closed_loop(
    n_rounds: int = 5,
    attempts_per_round: int = 4,
    techniques: tuple[str, ...] = ("branded_whisper", "vault_whisper", "cart_inflation", "currency_locale_confusion"),
    backend: ChatBackend | None = None,
) -> list[ClosedLoopRound]:
    backend = backend or get_backend()
    learned_texts: list[str] = []
    technique_history: dict[str, list[AttackAttempt]] = {t: [] for t in techniques}
    rounds: list[ClosedLoopRound] = []

    for round_num in range(n_rounds):
        firewall = None if round_num == 0 else MandateFirewall(learned_texts=learned_texts)
        round_stats: dict[str, TechniqueRoundStats] = {}
        new_successful_payloads: list[str] = []

        for technique in techniques:
            successes = incompletes = fallbacks = 0
            successful_payloads: list[str] = []
            for _ in range(attempts_per_round):
                try:
                    attempt = run_attack_round(
                        technique, len(technique_history[technique]), technique_history[technique], firewall=firewall, backend=backend
                    )
                except GenerationBlocked as exc:
                    # The provider's own safety classifier refused to write
                    # this payload. That is a real operating condition for
                    # an automated red team against a hosted model, not an
                    # error to retry around, so the attempt falls back to
                    # the deterministic template library and is counted, so
                    # the reported bypass rate says how much of the campaign
                    # was model-authored and how much was templated.
                    fallbacks += 1
                    print(f"  round {round_num} [{technique}]: generation refused by provider policy, using template ({exc})")
                    attempt = run_attack_round(
                        technique,
                        len(technique_history[technique]),
                        technique_history[technique],
                        firewall=firewall,
                        backend=get_backend("scripted"),
                    )
                technique_history[technique].append(attempt)
                if attempt.divergence.incomplete:
                    incompletes += 1
                elif attempt.divergence.attack_succeeded:
                    successes += 1
                    successful_payloads.append(attempt.payload)
                print(f"  round {round_num} [{technique}] attempt {len(technique_history[technique])-1}: succeeded={attempt.divergence.attack_succeeded} incomplete={attempt.divergence.incomplete}")

            completed = attempts_per_round - incompletes
            bypass_rate = successes / completed if completed else float("nan")
            round_stats[technique] = TechniqueRoundStats(
                technique=technique, attempts=attempts_per_round, successes=successes,
                incompletes=incompletes, bypass_rate=bypass_rate, successful_payloads=successful_payloads,
                template_fallbacks=fallbacks,
            )
            new_successful_payloads.extend(successful_payloads)

        overall_successes = sum(s.successes for s in round_stats.values())
        overall_completed = sum(s.attempts - s.incompletes for s in round_stats.values())
        overall_bypass = overall_successes / overall_completed if overall_completed else float("nan")

        detector_kind = firewall.detector_kind if firewall else None
        rounds.append(ClosedLoopRound(round_num, firewall is not None, detector_kind, round_stats, overall_bypass))
        print(f"round {round_num} DONE: overall_bypass_rate={overall_bypass:.2%} detector={detector_kind}")

        learned_texts.extend(new_successful_payloads)

    return rounds


def run_tabular_adversarial_loop(n_rounds: int = 3) -> dict:
    """Wraps the GBM adversarial-hardening arm. Trains the ULB baseline
    fresh, then runs iterative_harden, see janus/generate/adversarial/evasion.py."""

    from sklearn.model_selection import train_test_split

    from janus.data.load import load_ulb
    from janus.defend.gbm import train_ulb_baseline
    from janus.generate.adversarial.evasion import iterative_harden

    df = load_ulb()
    trained = train_ulb_baseline(df)
    model, threshold = trained.model, trained.eval_report.threshold

    feature_cols = trained.feature_names
    fraud_rows = df[df["Class"] == 1][feature_cols]
    _, fraud_eval = train_test_split(fraud_rows, test_size=0.5, random_state=42)

    x_full, y_full = df[feature_cols], df["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    hardened_model, rounds = iterative_harden(model, x_train, y_train, x_test, y_test, fraud_eval, threshold, n_rounds=n_rounds)

    return {
        "baseline_eval": trained.eval_report.as_dict(),
        "rounds": [
            {"round": r.round, "evasion_rate": r.evasion_rate, "n_targeted": r.n_targeted, "n_evaded": r.n_evaded,
             "mean_features_perturbed": r.mean_features_perturbed, "mean_l2_perturbation": r.mean_l2_perturbation,
             "mean_perturbation_std_units": r.mean_perturbation_std_units,
             "clean_eval": r.clean_eval}
            for r in rounds
        ],
    }


def combined_curve(agentic_rounds: list[ClosedLoopRound], tabular_result: dict) -> list[dict]:
    """One (attacker_win_rate, defense_strength) pair per round index,
    normalized across the two arms so the UI can plot a single curve.
    defense_strength = 1 - attacker_win_rate for the agentic arm (bypass
    rate IS the attacker's win rate there); for the tabular arm,
    defense_strength is the held-out clean PR-AUC (a genuinely different
    axis (accuracy, not resistance) documented as such in the point's
    own `note` field rather than silently conflated)."""

    points = []
    for r in agentic_rounds:
        points.append({
            "round": r.round_num, "arm": "agentic",
            "attacker_win_rate": r.overall_bypass_rate,
            "defense_strength": None if r.overall_bypass_rate != r.overall_bypass_rate else round(1 - r.overall_bypass_rate, 4),
            "note": "defense_strength = 1 - overall bypass rate",
        })
    for r in tabular_result.get("rounds", []):
        points.append({
            "round": r["round"], "arm": "tabular_adversarial",
            "attacker_win_rate": round(r["evasion_rate"], 4),
            "defense_strength": round(r["clean_eval"]["pr_auc"], 4),
            "note": "defense_strength = held-out clean PR-AUC (a different axis than evasion rate, not directly comparable to bypass rate)",
        })
    return points
