"""Discrete-event behavioral simulator, produces per-entity event
SEQUENCES (auth -> decline -> retry -> dispute; or repeated near-threshold
structuring payments) rather than flat i.i.d. rows, which is what the
sequence transformer in janus/defend/sequence.py needs to have anything
temporal to learn from.

Real transaction datasets with rich per-account history and a genuine
"was this card being tested" label are not available credential-free (and
PaySim's own accounts are structurally ~always single-transaction, so it
can't supply this either, see janus/generate/graph/mule_ring.py's
docstring for the same structural finding). This simulator is therefore
the PRIMARY source for this category, not a stand-in for a withheld real
dataset, documented as such rather than implied to be calibrated against
real per-account behavioral data, which does not exist in this pipeline.

Three entity behavior classes:
- "legit": a handful of ordinary approved purchases, occasional benign
  decline+retry (wrong CVV, insufficient funds resolved next try).
- "card_testing_bot" (Category C, attack #10): a rapid burst of small,
  varied-amount auth attempts against a static card, high decline rate,
  amounts stepping through a narrow range (BIN/amount iteration), no
  human-plausible pacing.
- "structuring" (Category C, attack #12): repeated payments clustered
  just under a reporting/velocity threshold, evenly paced to look
  routine individually, only suspicious in aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVENT_TYPES = ["auth_approved", "auth_declined", "retry", "dispute"]
EVENT_TO_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}


@dataclass
class EntitySequence:
    entity_id: str
    label: int  # 0 = legit, 1 = fraud (card_testing_bot or structuring)
    behavior: str
    steps: list[int] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)


def _simulate_legit(rng: np.random.RandomState, entity_id: str, max_step: int) -> EntitySequence:
    n_events = rng.randint(2, 8)
    steps = sorted(rng.randint(0, max_step, size=n_events).tolist())
    amounts, events = [], []
    for _ in range(n_events):
        amounts.append(float(rng.lognormal(mean=3.5, sigma=1.0)))
        events.append("auth_declined" if rng.random() < 0.05 else "auth_approved")
        if events[-1] == "auth_declined" and rng.random() < 0.6:
            events[-1] = "retry"
    return EntitySequence(entity_id=entity_id, label=0, behavior="legit", steps=steps, amounts=amounts, event_types=events)


def _simulate_card_testing(rng: np.random.RandomState, entity_id: str, max_step: int) -> EntitySequence:
    n_events = rng.randint(15, 60)
    burst_start = rng.randint(0, max(1, max_step - 20))
    steps = sorted((burst_start + rng.randint(0, 20, size=n_events)).tolist())
    base_amount = rng.uniform(1, 5)
    amounts, events = [], []
    for i in range(n_events):
        amounts.append(round(base_amount + i * rng.uniform(0.1, 1.0), 2))  # stepping through amounts
        events.append("auth_approved" if rng.random() < 0.15 else "auth_declined")
    return EntitySequence(entity_id=entity_id, label=1, behavior="card_testing_bot", steps=steps, amounts=amounts, event_types=events)


def _simulate_structuring(rng: np.random.RandomState, entity_id: str, max_step: int, threshold: float = 10_000.0) -> EntitySequence:
    n_events = rng.randint(6, 14)
    spacing = max(1, max_step // (n_events + 1))
    steps = [i * spacing + rng.randint(0, spacing) for i in range(1, n_events + 1)]
    amounts, events = [], []
    for _ in range(n_events):
        amounts.append(round(threshold * rng.uniform(0.85, 0.98), 2))  # just under threshold
        events.append("auth_approved")
    return EntitySequence(entity_id=entity_id, label=1, behavior="structuring", steps=sorted(steps), amounts=amounts, event_types=events)


def simulate_population(n_legit: int = 2000, n_card_testing: int = 150, n_structuring: int = 150, max_step: int = 744, seed: int = 42) -> list[EntitySequence]:
    rng = np.random.RandomState(seed)
    sequences: list[EntitySequence] = []
    for i in range(n_legit):
        sequences.append(_simulate_legit(rng, f"ENT_LEGIT_{i:05d}", max_step))
    for i in range(n_card_testing):
        sequences.append(_simulate_card_testing(rng, f"ENT_CARDTEST_{i:05d}", max_step))
    for i in range(n_structuring):
        sequences.append(_simulate_structuring(rng, f"ENT_STRUCT_{i:05d}", max_step))
    rng.shuffle(sequences)
    return sequences
