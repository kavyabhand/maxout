"""Gradient-boosted trees: the fast, explainable first-pass scoring member
of the 5-family defend ensemble, and the tabular half of the GNN+GBM hybrid
(see gnn.py). Runs standalone here (CPU, local) as its own baseline, then
GNN embeddings get concatenated in as extra features for the hybrid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from janus.common.features import ulb_feature_cols
from janus.common.metrics import EvalReport, evaluate


@dataclass
class TrainedModel:
    model: xgb.XGBClassifier
    eval_report: EvalReport
    train_latency_s: float
    inference_latency_ms_per_1k: float
    feature_names: list[str]


def train_ulb_baseline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainedModel:
    """ULB creditcard.csv: Time, Amount, V1-V28, Class. Stratified split
    preserves the 0.172% fraud rate in both train and test.

    `Time` is excluded (see janus/common/features.py): it is a
    capture-relative clock no live scorer has, and it distorted both the
    fidelity distinguisher and the evasion budget downstream."""

    feature_cols = ulb_feature_cols(df)
    x, y = df[feature_cols], df["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )

    start = time.time()
    model.fit(x_train, y_train)
    train_latency_s = time.time() - start

    y_score = model.predict_proba(x_test)[:, 1]
    report = evaluate(y_test.to_numpy(), y_score)

    infer_start = time.time()
    _ = model.predict_proba(x_test.iloc[:1000])
    infer_latency = (time.time() - infer_start) * 1000

    return TrainedModel(
        model=model, eval_report=report, train_latency_s=train_latency_s,
        inference_latency_ms_per_1k=infer_latency, feature_names=feature_cols,
    )


def train_ieee_baseline(df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.2, random_state: int = 42) -> TrainedModel:
    """Same GBM recipe on IEEE-CIS's richer feature set, taking an explicit
    feature_cols list since IEEE-CIS's raw columns need numeric selection/
    encoding first (see graph.py's _select_numeric_columns for the same
    selection logic the GNN side uses, so both halves of the hybrid train
    on a consistent feature basis)."""

    x, y = df[feature_cols].fillna(-999), df["isFraud"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )

    start = time.time()
    model.fit(x_train, y_train)
    train_latency_s = time.time() - start

    y_score = model.predict_proba(x_test)[:, 1]
    report = evaluate(y_test.to_numpy(), y_score)

    infer_start = time.time()
    _ = model.predict_proba(x_test.iloc[:1000])
    infer_latency = (time.time() - infer_start) * 1000

    return TrainedModel(
        model=model, eval_report=report, train_latency_s=train_latency_s,
        inference_latency_ms_per_1k=infer_latency, feature_names=feature_cols,
    )


if __name__ == "__main__":
    from janus.data.load import load_ulb

    df = load_ulb()
    print(f"loaded ULB: {len(df)} rows, {df['Class'].sum()} fraud ({df['Class'].mean():.4%})")

    trained = train_ulb_baseline(df)
    print("\n=== GBM baseline on ULB (held-out test set) ===")
    for k, v in trained.eval_report.as_dict().items():
        print(f"  {k}: {v}")
    print(f"  train_latency_s: {trained.train_latency_s:.2f}")
    print(f"  inference: {trained.inference_latency_ms_per_1k:.2f} ms / 1000 rows ({trained.inference_latency_ms_per_1k / 1000:.4f} ms/row)")
