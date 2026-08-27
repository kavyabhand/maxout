"""Runs the five-family defend stack end-to-end on one shared substrate.

Until this module existed, each defend family reported its own eval on its
own data and `janus/defend/meta.py` was never called by any pipeline; the
"stacked into one calibrated decision" claim had no measured number behind
it. This closes that gap on IEEE-CIS, which is the only dataset here where
genuinely different families can score the *same* rows: it carries both
raw tabular features (GBM), entity linkage fields that make a real graph
(GNN), and enough dimensionality for an unsupervised detector to have
something to say (IsolationForest).

Stacking protocol, three disjoint splits, because the obvious two-split
version is silently optimistic:

    train (60%)  members are fit here
    meta  (20%)  members SCORE this split; the stack is fit on those scores
    test  (20%)  members score it, the stack combines, and that is reported

Fitting the stack on member scores from the members' own training rows
would hand it in-sample scores that are far sharper than anything it will
see in production, and the resulting test number would be inflated. The
meta split exists purely so the stack learns from member behaviour on data
the members did not memorize.

The GNN is transductive over the full graph, message passing sees every
node's features, which is exactly the point of a graph model, but its
supervised loss is masked to the train split only, so no test label
reaches it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MEMBER_NAMES = ["gbm_tabular", "gnn_graph", "anomaly_isolation_forest"]


@dataclass
class EnsembleResult:
    member_evals: dict[str, dict]
    stacked_eval: dict
    member_weights: dict[str, float]
    tier_distribution: dict
    tier_distribution_capacity: dict
    n_train: int
    n_meta: int
    n_test: int
    split_note: str
    wall_clock_s: float
    graph: dict = field(default_factory=dict)
    scored_sample: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "members": self.member_evals,
            "stacked": self.stacked_eval,
            "member_weights": self.member_weights,
            "tier_distribution": self.tier_distribution,
            "tier_distribution_capacity": self.tier_distribution_capacity,
            "n_train": self.n_train,
            "n_meta": self.n_meta,
            "n_test": self.n_test,
            "split_note": self.split_note,
            "wall_clock_s": round(self.wall_clock_s, 1),
            "graph": self.graph,
            "scored_sample": self.scored_sample,
        }


def run_ieee_ensemble(df: pd.DataFrame, *, gnn_epochs: int = 60, random_state: int = 42, sample_size: int = 1500) -> EnsembleResult:
    import torch
    import torch.nn.functional as F
    import xgboost as xgb

    from janus.common.metrics import evaluate
    from janus.defend.anomaly import anomaly_score, train_anomaly_detector
    from janus.defend.gnn import JanusSAGE, _select_device, build_transaction_entity_graph, select_numeric_columns
    from janus.defend.meta import TierThresholds, decision_tier, fit_meta_learner, tier_distribution

    started = time.time()
    df = df.reset_index(drop=True)
    numeric_cols = select_numeric_columns(df)
    y_all = df["isFraud"].to_numpy()
    n_txn = len(df)

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_txn)
    n_test = int(n_txn * 0.2)
    n_meta = int(n_txn * 0.2)
    test_idx = perm[:n_test]
    meta_idx = perm[n_test:n_test + n_meta]
    train_idx = perm[n_test + n_meta:]

    x_tab = df[numeric_cols].fillna(-999).to_numpy(dtype=np.float32)

    # --- member 1: GBM on raw tabular features -------------------------
    scale_pos_weight = (y_all[train_idx] == 0).sum() / max((y_all[train_idx] == 1).sum(), 1)
    gbm = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
        random_state=random_state, n_jobs=-1,
    )
    gbm.fit(x_tab[train_idx], y_all[train_idx])
    gbm_meta = gbm.predict_proba(x_tab[meta_idx])[:, 1]
    gbm_test = gbm.predict_proba(x_tab[test_idx])[:, 1]

    # --- member 2: GraphSAGE over the transaction/entity graph ---------
    graph = build_transaction_entity_graph(df, numeric_cols)
    device = _select_device()
    data = graph.data.to(device)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[torch.tensor(train_idx, dtype=torch.long, device=device)] = True

    model = JanusSAGE(in_channels=data.x.shape[1]).to(device)
    class_counts = torch.bincount(data.y[train_mask])
    class_weights = (class_counts.sum() / (2.0 * class_counts.float())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for _ in range(gnn_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        gnn_scores = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    gnn_meta, gnn_test = gnn_scores[meta_idx], gnn_scores[test_idx]

    # --- member 3: unsupervised IsolationForest ------------------------
    # Through janus/defend/anomaly.py rather than a local IsolationForest.
    # An inlined copy here meant that module had no caller anywhere in the
    # repo while the write-up described it as one of the defend families,
    # which is exactly the kind of claim this project is trying not to make.
    # Numerically identical: same estimator, parameters, seed and min-max
    # flip, so the artifact this produced remains valid.
    def _frame(idx: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x_tab[idx], columns=numeric_cols)

    anomaly = train_anomaly_detector(_frame(train_idx), _frame(test_idx), pd.Series(y_all[test_idx]))
    iso_meta = anomaly_score(anomaly.model, _frame(meta_idx))
    iso_test = anomaly_score(anomaly.model, _frame(test_idx))

    # --- the stack -----------------------------------------------------
    meta_matrix = np.column_stack([gbm_meta, gnn_meta, iso_meta])
    test_matrix = np.column_stack([gbm_test, gnn_test, iso_test])

    stack = fit_meta_learner(meta_matrix, y_all[meta_idx], MEMBER_NAMES, random_state=random_state)
    stacked_test_score = stack.model.predict_proba(test_matrix)[:, 1]
    stacked_eval = evaluate(y_all[test_idx], stacked_test_score)

    member_evals = {
        "gbm_tabular": evaluate(y_all[test_idx], gbm_test).as_dict(),
        "gnn_graph": evaluate(y_all[test_idx], gnn_test).as_dict(),
        "anomaly_isolation_forest": evaluate(y_all[test_idx], iso_test).as_dict(),
    }

    # A faithful random slice of the scored test split, carried in the
    # artifact so the prototype's authorization stream can replay decisions
    # this model actually made on data it was not trained on, rather than
    # animating invented numbers. Not stratified or curated: the tier mix
    # and fraud rate a viewer sees are the measured ones.
    sample_rng = np.random.RandomState(random_state + 1)
    take = sample_rng.choice(len(test_idx), size=min(sample_size, len(test_idx)), replace=False)
    amounts = df["TransactionAmt"].to_numpy() if "TransactionAmt" in df.columns else np.zeros(n_txn)
    txn_dt = df["TransactionDT"].to_numpy() if "TransactionDT" in df.columns else np.zeros(n_txn)
    product = df["ProductCD"].astype(str).to_numpy() if "ProductCD" in df.columns else np.full(n_txn, "?")

    scored_sample = [
        {
            "amount": round(float(amounts[test_idx[i]]), 2),
            "hour": int((txn_dt[test_idx[i]] // 3600) % 24),
            "product": str(product[test_idx[i]]),
            "gbm": round(float(gbm_test[i]), 5),
            "gnn": round(float(gnn_test[i]), 5),
            "anomaly": round(float(iso_test[i]), 5),
            "risk": round(float(stacked_test_score[i]), 5),
            "tier": decision_tier(float(stacked_test_score[i])),
            "is_fraud": int(y_all[test_idx[i]]),
        }
        for i in take
    ]

    return EnsembleResult(
        scored_sample=scored_sample,
        member_evals=member_evals,
        stacked_eval=stacked_eval.as_dict(),
        member_weights=dict(zip(MEMBER_NAMES, [round(w, 4) for w in stack.member_weights])),
        tier_distribution=tier_distribution(stacked_test_score, y_all[test_idx]).as_dict(),
        # The same population under capacity-planned cuts. Reported
        # alongside because fixed probability cuts leave the decline tier
        # empty whenever the model is never confident enough to reach them,
        # which is a real property of the model and a misleading table.
        tier_distribution_capacity=tier_distribution(
            stacked_test_score,
            y_all[test_idx],
            TierThresholds.from_volume_targets(stacked_test_score),
        ).as_dict(),
        n_train=len(train_idx), n_meta=len(meta_idx), n_test=len(test_idx),
        split_note=(
            "Members fit on train; the stack fit on member scores over a held-out meta split; "
            "every number reported here measured on a test split neither the members nor the "
            "stack were fit on. The GNN is transductive (message passing sees all node features) "
            "but its supervised loss is masked to train rows only."
        ),
        wall_clock_s=time.time() - started,
        graph={"n_nodes": int(data.num_nodes), "n_edges": int(data.num_edges), "device": str(device), "epochs": gnn_epochs},
    )
