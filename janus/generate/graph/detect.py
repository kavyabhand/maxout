"""Mule-ring detection via account-level velocity/topology features +
XGBoost. Deliberately feature-engineered rather than a full GNN retrain for
this secondary pillar; the account-level fan-in/fan-out/burst-compactness
features below are exactly the graph-topology signal a GraphSAGE encoder
would also have to learn to extract, so computing them directly is the
lightweight-but-real version of reusing the GNN architecture. Swapping in
the actual GraphSAGE encoder from janus/defend/gnn.py on an account-node
graph (nodes=accounts, edges=transactions) is a direct, described
extension of this same pipeline if ring topology gets adversarially
harder to characterize with hand-engineered features alone.

Target: is this DESTINATION account a mule-ring hub? (not per-transaction
fraud, PaySim already has its own narrow fraud label for that; this is a
genuinely different, higher-level question about account ROLE.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from janus.common.metrics import EvalReport, evaluate


def _account_features(df: pd.DataFrame, account_col: str, other_col: str, amount_col: str, step_col: str, prefix: str) -> pd.DataFrame:
    grouped = df.groupby(account_col)
    feats = grouped.agg(
        **{
            f"{prefix}_n_txn": (other_col, "count"),
            f"{prefix}_n_unique_counterparties": (other_col, "nunique"),
            f"{prefix}_total_amount": (amount_col, "sum"),
            f"{prefix}_mean_amount": (amount_col, "mean"),
            f"{prefix}_min_step": (step_col, "min"),
            f"{prefix}_max_step": (step_col, "max"),
        }
    )
    feats[f"{prefix}_span_steps"] = feats[f"{prefix}_max_step"] - feats[f"{prefix}_min_step"]
    return feats.drop(columns=[f"{prefix}_max_step"])


def build_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per account that appears as a destination at least once."""

    inbound = _account_features(df, "nameDest", "nameOrig", "amount", "step", "in")
    outbound = _account_features(df, "nameOrig", "nameDest", "amount", "step", "out")

    features = inbound.join(outbound, how="left")
    for col in ["out_n_txn", "out_n_unique_counterparties", "out_total_amount", "out_mean_amount", "out_span_steps"]:
        features[col] = features[col].fillna(0.0)
    features["out_min_step"] = features["out_min_step"].fillna(np.inf)

    features["flow_ratio"] = features["out_total_amount"] / features["in_total_amount"].clip(lower=1.0)
    features["time_to_first_outflow"] = (features["out_min_step"] - features["in_min_step"]).clip(lower=-1e6)
    features["time_to_first_outflow"] = features["time_to_first_outflow"].replace(np.inf, 9999)
    features = features.drop(columns=["out_min_step", "in_min_step"])
    features = features.fillna(0.0)
    return features


def label_accounts(feature_index: pd.Index, ring_meta: pd.DataFrame) -> pd.Series:
    mule_accounts = set(ring_meta["mule_account"])
    return pd.Series([1 if acct in mule_accounts else 0 for acct in feature_index], index=feature_index)


def train_mule_detector(
    df: pd.DataFrame, ring_meta: pd.DataFrame, negative_sample_n: int = 20_000, random_state: int = 42
) -> tuple[xgb.XGBClassifier, EvalReport]:
    features = build_account_features(df)
    labels = label_accounts(features.index, ring_meta)

    positives = features[labels == 1]
    negative_pool = features[labels == 0]

    # Hard negatives (benign-but-bidirectional accounts, see mule_ring.py's
    # inject_benign_bidirectional_accounts) must always be included, not
    # just randomly diluted into a much larger pool of trivial pure-sink
    # negatives, random sampling would include only a handful of them by
    # chance, defeating the point of adding them.
    is_hard_negative = negative_pool.index.str.startswith("C_BENIGN_")
    hard_negatives = negative_pool[is_hard_negative]
    easy_pool = negative_pool[~is_hard_negative]
    n_easy = max(min(negative_sample_n, len(easy_pool)) - len(hard_negatives), 0)
    easy_negatives = easy_pool.sample(n=n_easy, random_state=random_state)
    negatives = pd.concat([hard_negatives, easy_negatives])

    X = pd.concat([positives, negatives])
    y = pd.concat([labels.loc[positives.index], labels.loc[negatives.index]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    report = evaluate(y_test.to_numpy(), y_score)
    return model, report


if __name__ == "__main__":
    from janus.data.load import load_paysim
    from janus.generate.graph.mule_ring import inject_benign_bidirectional_accounts, inject_mule_rings

    sample = load_paysim(sample_n=300_000)
    augmented, ring_meta = inject_mule_rings(sample, n_rings=60)
    augmented = inject_benign_bidirectional_accounts(augmented, n_accounts=300)
    print(f"augmented: {len(augmented)} rows, {len(ring_meta)} injected mule rings, "
          f"300 hard-negative benign bidirectional accounts")

    model, report = train_mule_detector(augmented, ring_meta)
    print("\n=== mule-ring account detector, WITH hard negatives (held-out test set) ===")
    for k, v in report.as_dict().items():
        print(f"  {k}: {v}")
