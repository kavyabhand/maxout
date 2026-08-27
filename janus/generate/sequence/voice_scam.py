"""Category B, attack #6 (Deepfake "Emergency" Voice/Video Scams), transaction-side detection only. Deliberately does NOT generate or detect
audio; a large, separate ML problem that would cannibalize time better
spent on the transaction/agent problems. What's actually detectable
without touching audio is the downstream TRANSACTION FINGERPRINT a
successful vishing call produces: the victim, now convinced by a cloned
voice claiming to be a family member or executive, authorizes an
out-of-band instant transfer that looks nothing like their normal
behavior: first-time beneficiary, an amount far outside the
population-typical distribution, an off-hours timestamp, and a channel
switch to the instant-rail-equivalent type.

Honest limitation, not a design preference: real APP-fraud systems ideally
profile against each account's OWN history. PaySim's `nameOrig` accounts
are overwhelmingly single-transaction across the whole dataset (mean
1.0014 transactions/account, max 3, across 6.35M accounts), so per-account
baselining isn't meaningful on this substrate. The features below are
POPULATION/SEGMENT-relative instead; a realistic real-world fallback,
since most retail-payment fraud systems fall back to peer-group baselines
for exactly this reason (most consumer accounts don't have enough
individual history for a personalized model either).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from janus.common.metrics import EvalReport, evaluate

OFF_HOURS = (22, 23, 0, 1, 2, 3, 4, 5)


def inject_voice_scam_transactions(df: pd.DataFrame, n_scams: int = 300, off_hours: tuple[int, ...] = OFF_HOURS, seed: int = 44) -> pd.DataFrame:
    """Injects one out-of-character, first-time-beneficiary, off-hours,
    amount-outlier TRANSFER per victim account, sized relative to the
    POPULATION-typical (upper-half of real fraud-labeled) TRANSFER amount
    rather than the account's own history."""

    rng = np.random.RandomState(seed)
    df = df.copy()
    if "scam_type" not in df.columns:
        df["scam_type"] = None

    real_fraud_transfer = df.loc[(df["isFraud"] == 1) & (df["type"] == "TRANSFER"), "amount"]
    upper_half_pool = real_fraud_transfer[real_fraud_transfer >= real_fraud_transfer.median()].to_numpy()
    if len(upper_half_pool) == 0:
        upper_half_pool = df.loc[df["type"] == "TRANSFER", "amount"].to_numpy()

    candidate_accounts = df["nameOrig"].drop_duplicates().to_numpy()
    victims = rng.choice(candidate_accounts, size=min(n_scams, len(candidate_accounts)), replace=False)

    max_step = int(df["step"].max())
    rows = []
    for i, victim in enumerate(victims):
        scam_amount = float(rng.choice(upper_half_pool)) * rng.uniform(0.9, 1.1)
        day = rng.randint(0, max_step // 24)
        hour = rng.choice(off_hours)
        step = min(day * 24 + hour, max_step)
        beneficiary = f"C_SCAM_BENEFICIARY_{i:05d}"
        rows.append({
            "step": step, "type": "TRANSFER", "amount": round(float(scam_amount), 2),
            "nameOrig": victim, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
            "nameDest": beneficiary, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "isFraud": 1, "isFlaggedFraud": 0, "ring_id": np.nan, "scam_type": "voice_clone_app_fraud",
        })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def build_behavioral_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per-transaction features, population/segment-relative rather than
    per-account (see module docstring). Returns (features, labels)."""

    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    # Percentile rank of amount WITHIN transaction type, not a z-score:
    # real PaySim TRANSFER amounts are extremely heavy-tailed (legit max is
    # ~92M vs. median ~490k), which swamps a std-based z-score; percentile
    # rank is robust to that tail shape by construction.
    amount_percentile = df.groupby("type")["amount"].rank(pct=True)

    # channel_switch: TRANSFER is the instant-rail-equivalent channel real
    # APP-fraud payouts favor, over the account's more typical PAYMENT/CASH_OUT
    channel_switch = (df["type"] == "TRANSFER").astype(int)
    hour_of_day = (df["step"] % 24).to_numpy()

    features = pd.DataFrame(
        {
            "amount": df["amount"].to_numpy(),
            "amount_percentile_within_type": amount_percentile.to_numpy(),
            "channel_switch": channel_switch.to_numpy(),
            "hour_of_day": hour_of_day,
            "is_off_hours": np.isin(hour_of_day, list(OFF_HOURS)).astype(int),
        },
        index=df.index,
    )
    return features, df["isFraud"]


def train_voice_scam_detector(df: pd.DataFrame, random_state: int = 42) -> tuple[xgb.XGBClassifier, EvalReport]:
    x, y = build_behavioral_features(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state, stratify=y)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    report = evaluate(y_test.to_numpy(), y_score)
    return model, report
