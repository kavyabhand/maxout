"""Family B, GenAI-recruited mule-ring injection on top of PaySim.

PaySim (`ealaxi/paysim1`, 6.36M rows) provides the realistic "normal"
background population: the baseline graph of legitimate transactions
serves as the host network into which the simulator injects its synthetic
fraud paths (dev/RESEARCH.md §4.1.3, "Graph-Based Mule-Ring Generator"). PaySim's own
built-in fraud labels are a narrow TRANSFER-then-CASH_OUT pattern; they do
NOT model organized mule-ring topology (many-source fan-in -> one mule ->
many-sink fan-out, clustered in time) which is the actual signature GenAI-
recruited mule networks produce (taxonomy attack C12, "Synthetic
Mule-Network Orchestration", recruitment bots
onboard many mules who then move funds in coordinated bursts to evade
single-hop velocity rules). This module injects that topology explicitly,
on top of (not instead of) PaySim's real accounts and balances, so the
resulting graph has both a realistic legitimate backbone and a genuinely
novel-relative-to-PaySim fraud pattern to detect.

Temporal burstiness is deliberate (dev/RESEARCH.md §4 on realistic
distributions, behaviours and edge cases): each ring's fan-in and fan-out both
happen within a tight step window, mimicking real synchronized cash-out
runs rather than uniformly-scattered fraud that would make the problem
artificially easy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_mule_rings(
    df: pd.DataFrame,
    n_rings: int = 40,
    fan_in_range: tuple[int, int] = (6, 18),
    fan_out_range: tuple[int, int] = (2, 5),
    fan_in_window_steps: int = 2,
    fan_out_delay_range: tuple[int, int] = (1, 4),
    skim_rate_range: tuple[float, float] = (0.05, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (augmented_df, ring_metadata). augmented_df has the original
    rows plus injected rows; injected rows carry `isFraud=1` and a
    `ring_id` (NaN for all original rows) so ring-level structure is
    recoverable for graph construction / evaluation.
    """

    rng = np.random.RandomState(seed)
    df = df.copy()
    if "ring_id" not in df.columns:
        df["ring_id"] = np.nan

    # Pool of real, sufficiently-funded accounts to act as unwitting fan-in
    # sources (their own outgoing transfer just looks like one more
    # transaction from their perspective; the fraud signal is in the
    # mule's aggregate in/out pattern, not any single source's behavior).
    source_pool = df.loc[df["oldbalanceOrg"] > 5000, "nameOrig"].drop_duplicates().to_numpy()
    max_step = int(df["step"].max())

    # Each individual fan-in transaction is sized by BOOTSTRAP SAMPLING from
    # PaySim's own real legitimate TRANSFER amounts (not a guessed constant
    # range); the entire point of a structuring/smurfing pattern is that
    # each single transaction looks like an unremarkable, real, ordinary
    # transfer; the fraud signal is in the aggregate fan-in/fan-out
    # structure, not in any individual amount looking suspicious. An
    # earlier version used `uniform(200, 2500)`, which was ~1000x too small
    # relative to PaySim's actual amount scale (real TRANSFER median is
    # ~450k-490k, fraud and legit alike) and produced a near-total
    # distributional non-overlap on the fidelity check, see BUILDLOG.md
    # 2026-08-15 for the diagnosis. Bootstrapping from the real empirical
    # distribution guarantees a realistic marginal by construction.
    real_legit_transfer_pool = df.loc[(df["isFraud"] == 0) & (df["type"] == "TRANSFER"), "amount"].to_numpy()

    injected_rows = []
    ring_records = []

    for ring_idx in range(n_rings):
        mule_account = f"C_MULE_{ring_idx:05d}"
        fan_in_n = rng.randint(*fan_in_range)
        fan_out_n = rng.randint(*fan_out_range)
        t0 = rng.randint(1, max(2, max_step - fan_out_delay_range[1] - fan_in_window_steps - 1))

        sources = rng.choice(source_pool, size=fan_in_n, replace=False)
        received_total = 0.0
        for src in sources:
            step = t0 + rng.randint(0, fan_in_window_steps + 1)
            # Bootstrap an amount from the real legit distribution, then
            # discount it toward the lower end (structuring/smurfing keeps
            # individual amounts modest relative to what's typical, while
            # staying within the real range rather than off-scale small).
            amount = float(rng.choice(real_legit_transfer_pool)) * rng.uniform(0.05, 0.35)
            received_total += amount
            injected_rows.append(
                {
                    "step": step,
                    "type": "TRANSFER",
                    "amount": round(amount, 2),
                    "nameOrig": src,
                    "oldbalanceOrg": 0.0,  # unknown/not recomputed, structural signal is what matters here
                    "newbalanceOrig": 0.0,
                    "nameDest": mule_account,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "isFraud": 1,
                    "isFlaggedFraud": 0,
                    "ring_id": ring_idx,
                }
            )

        skim = rng.uniform(*skim_rate_range)
        payout_total = received_total * (1 - skim)
        sinks = [f"C_SINK_{ring_idx:05d}_{k}" for k in range(fan_out_n)]
        remaining = payout_total
        for i, sink in enumerate(sinks):
            step = t0 + fan_in_window_steps + rng.randint(*fan_out_delay_range)
            share = payout_total / fan_out_n if i < fan_out_n - 1 else remaining
            remaining -= share
            injected_rows.append(
                {
                    "step": min(step, max_step),
                    "type": "CASH_OUT",
                    "amount": round(max(share, 1.0), 2),
                    "nameOrig": mule_account,
                    "oldbalanceOrg": 0.0,
                    "newbalanceOrig": 0.0,
                    "nameDest": sink,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "isFraud": 1,
                    "isFlaggedFraud": 0,
                    "ring_id": ring_idx,
                }
            )

        ring_records.append(
            {
                "ring_id": ring_idx,
                "mule_account": mule_account,
                "fan_in_n": fan_in_n,
                "fan_out_n": fan_out_n,
                "t0": t0,
                "received_total": round(received_total, 2),
                "payout_total": round(payout_total, 2),
                "skim_rate": round(skim, 4),
            }
        )

    injected_df = pd.DataFrame(injected_rows)
    augmented = pd.concat([df, injected_df], ignore_index=True)
    ring_meta = pd.DataFrame(ring_records)
    return augmented, ring_meta


def inject_benign_bidirectional_accounts(
    df: pd.DataFrame,
    n_accounts: int = 300,
    seed: int = 43,
) -> pd.DataFrame:
    """Hard negatives: PaySim's real destination accounts essentially never
    send anything back out (verified empirically, see BUILDLOG.md
    2026-08-15, `time_to_first_outflow` was the sentinel "never" value for
    ~all sampled legit accounts), which made the mule-ring detector's first
    pass suspiciously perfect (1.0 PR-AUC); it was plausibly learning "did
    this account send ANYTHING back out," not real fan-in/burst/flow-ratio
    topology. These synthetic-but-benign accounts DO have both inbound and
    outbound activity, but scattered (single sender, spread-out timing, no
    tight burst, no consistent skim-style flow ratio): a legitimate
    consumer paying bills after receiving a salary transfer, not a mule.
    Forces the detector to learn the actual structural signature instead of
    a degenerate "has outflow" shortcut.
    """

    rng = np.random.RandomState(seed)
    source_pool = df.loc[df["oldbalanceOrg"] > 5000, "nameOrig"].drop_duplicates().to_numpy()
    dest_pool = df["nameDest"].drop_duplicates().to_numpy()
    max_step = int(df["step"].max())

    # Bootstrap amounts from PaySim's own real legit distributions per type
    # (same fix, same reason as the fan-in amounts above, see BUILDLOG.md
    # 2026-08-15) rather than an off-scale guessed constant range.
    real_transfer_pool = df.loc[(df["isFraud"] == 0) & (df["type"] == "TRANSFER"), "amount"].to_numpy()
    real_payment_pool = df.loc[(df["isFraud"] == 0) & (df["type"] == "PAYMENT"), "amount"].to_numpy()
    real_cashout_pool = df.loc[(df["isFraud"] == 0) & (df["type"] == "CASH_OUT"), "amount"].to_numpy()

    rows = []
    for i in range(n_accounts):
        account = f"C_BENIGN_{i:05d}"
        n_in = rng.randint(1, 4)
        n_out = rng.randint(1, 4)
        for _ in range(n_in):
            step = rng.randint(1, max_step)
            src = rng.choice(source_pool)
            rows.append(
                {
                    "step": step, "type": "TRANSFER", "amount": round(float(rng.choice(real_transfer_pool)), 2),
                    "nameOrig": src, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
                    "nameDest": account, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
                    "isFraud": 0, "isFlaggedFraud": 0, "ring_id": np.nan,
                }
            )
        for _ in range(n_out):
            step = rng.randint(1, max_step)  # NOT clustered right after inflow, that's the point
            dst = rng.choice(dest_pool)
            out_type = rng.choice(["PAYMENT", "CASH_OUT"])
            amount_pool = real_payment_pool if out_type == "PAYMENT" else real_cashout_pool
            rows.append(
                {
                    "step": step, "type": out_type, "amount": round(float(rng.choice(amount_pool)), 2),
                    "nameOrig": account, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
                    "nameDest": dst, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
                    "isFraud": 0, "isFlaggedFraud": 0, "ring_id": np.nan,
                }
            )

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


if __name__ == "__main__":
    from janus.data.load import load_paysim

    df = load_paysim()
    print(f"loaded PaySim: {len(df)} rows, {df['isFraud'].sum()} native-labeled fraud")

    sample = df.sample(n=300_000, random_state=42)  # tractable for local iteration
    augmented, ring_meta = inject_mule_rings(sample, n_rings=40)
    print(f"augmented: {len(augmented)} rows ({len(augmented) - len(sample)} injected), "
          f"{augmented['isFraud'].sum()} total fraud ({augmented['isFraud'].mean():.4%})")
    print(ring_meta.head())


def build_transfer_graph(df, *, max_edges: int = 200_000):
    """The account-to-account transfer graph as a plain networkx graph, for
    the fidelity scorer's topology comparison.

    The generator's own scorecard previously reported `graph_topology: null`
    on every batch because nothing in the pipeline ever built a graph object
    to hand it; the mule-ring code works on account-level feature frames
    throughout. This is the missing piece: it answers whether the injected
    ring topology sits inside the real transfer network's structural
    envelope or sticks out as an obviously synthetic clique, which is a
    fidelity failure a per-feature amount comparison structurally cannot
    see.

    Capped at `max_edges` (sampled, seeded) because average_clustering over
    a multi-million-edge graph is not worth the wall clock for a
    distributional summary.
    """

    import networkx as nx

    edges = df[["nameOrig", "nameDest"]].dropna()
    if len(edges) > max_edges:
        edges = edges.sample(n=max_edges, random_state=13)
    graph = nx.Graph()
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    return graph
