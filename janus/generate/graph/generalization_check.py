"""Out-of-distribution generalization check for the mule-ring detector.

The original evaluation (`detect.py`) trains and tests on rings drawn from
the SAME generator parameter distribution; a same-distribution held-out
split, which BUILDLOG.md 2026-08-15 already flagged as insufficient to
claim "mule-ring detection is solved" given the small (n~18) test-positive
count. This script is the flagged follow-up: train on one ring "style" and
test on a deliberately DIFFERENT one, to check whether the account-level
features generalize to a structurally different attacker rather than
memorizing this exact generator's fingerprint.

Two styles:
- "obvious": the original defaults, wide fan-in (6-18), tight 2-step
  window, fast fan-out (1-4 step delay). A blatant, easily-clustered burst.
- "subtle": narrower fan-in (3-7, closer to what a legitimate account might
  plausibly show), wider fan-in window (0-5 steps, less tightly clustered),
  slower fan-out delay (3-10 steps); an attacker deliberately spreading
  activity out to look less bursty, informed by the same "structuring"
  logic real launderers use to evade velocity rules.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from janus.common.metrics import evaluate
from janus.generate.graph.detect import build_account_features, label_accounts
from janus.generate.graph.mule_ring import inject_benign_bidirectional_accounts, inject_mule_rings

STYLES = {
    "obvious": dict(fan_in_range=(6, 18), fan_in_window_steps=2, fan_out_delay_range=(1, 4)),
    "subtle": dict(fan_in_range=(3, 7), fan_in_window_steps=5, fan_out_delay_range=(3, 10)),
}


def run_generalization_check(sample_n: int = 300_000, seed: int = 42) -> dict:
    from janus.data.load import load_paysim

    df = load_paysim()
    sample = df.sample(n=sample_n, random_state=seed)

    train_augmented, train_rings = inject_mule_rings(sample, n_rings=60, seed=seed, **STYLES["obvious"])
    train_augmented = inject_benign_bidirectional_accounts(train_augmented, n_accounts=300, seed=seed + 1)

    # A DIFFERENT sample (disjoint rows) for the test-distribution rings, so
    # there's no leakage of the same background transactions into both.
    test_sample = df.drop(sample.index).sample(n=sample_n, random_state=seed + 100)
    test_augmented, test_rings = inject_mule_rings(test_sample, n_rings=40, seed=seed + 2, **STYLES["subtle"])
    test_augmented = inject_benign_bidirectional_accounts(test_augmented, n_accounts=200, seed=seed + 3)

    train_features = build_account_features(train_augmented)
    train_labels = label_accounts(train_features.index, train_rings)
    test_features = build_account_features(test_augmented)
    test_labels = label_accounts(test_features.index, test_rings)

    train_positives = train_features[train_labels == 1]
    train_negatives = train_features[train_labels == 0].sample(
        n=min(20_000, (train_labels == 0).sum()), random_state=seed
    )
    X_train = pd.concat([train_positives, train_negatives])
    y_train = pd.concat([train_labels.loc[train_positives.index], train_labels.loc[train_negatives.index]])

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # In-distribution sanity check (held-out "obvious"-style rows).
    _, X_indist_test, _, y_indist_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=seed, stratify=y_train
    )
    indist_report = evaluate(y_indist_test.to_numpy(), model.predict_proba(X_indist_test)[:, 1])

    # Out-of-distribution: "subtle"-style rings the model never trained on,
    # plus their own fresh hard negatives.
    test_positives = test_features[test_labels == 1]
    test_negatives = test_features[test_labels == 0].sample(
        n=min(10_000, (test_labels == 0).sum()), random_state=seed
    )
    X_ood = pd.concat([test_positives, test_negatives])
    y_ood = pd.concat([test_labels.loc[test_positives.index], test_labels.loc[test_negatives.index]])
    ood_report = evaluate(y_ood.to_numpy(), model.predict_proba(X_ood)[:, 1])

    return {
        "in_distribution_obvious_style": indist_report.as_dict(),
        "out_of_distribution_subtle_style": ood_report.as_dict(),
        "n_train_rings_obvious": len(train_rings),
        "n_test_rings_subtle": len(test_rings),
    }


if __name__ == "__main__":
    import json

    result = run_generalization_check()
    print(json.dumps(result, indent=2))
