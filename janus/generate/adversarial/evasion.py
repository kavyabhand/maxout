"""Adversarial hardening exercise: the attacker arm of the closed loop
(dev/RESEARCH.md §4.1.4 "Adversarial Example Generator" and §6 "The
Closed Loop"), implementing attack C11 "Adversarial Evasion of Scoring
Models" from the taxonomy.

XGBoost (a tree ensemble) has no usable gradient the way a neural net
does, so this implements a practical BLACK-BOX evasion attack instead of a
gradient method: greedy coordinate-descent perturbation, which is the
standard practical approach for tree-ensemble evasion in the literature
(c.f. Kantchelian et al. 2016, "Evasion and Hardening of Tree Ensemble
Classifiers", MILP-exact or heuristic evasion against GBDTs specifically
because they aren't differentiable). For each fraud transaction, greedily
perturb one feature at a time (in the direction and by the step size that
most reduces predicted fraud probability, within a bounded budget) until
either the prediction flips below the decision threshold or the budget is
exhausted. This directly measures: how much feature-level wiggle room does
an attacker who can observe the model's score need to evade it?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class EvasionResult:
    n_targeted: int
    n_evaded: int
    evasion_rate: float
    mean_features_perturbed: float
    mean_l2_perturbation: float
    # Raw L2 is in whatever units the features happen to carry, so it is
    # not comparable across feature sets and not interpretable on its own
    # (on ULB it was dominated by whichever feature had the largest
    # variance). This is the same displacement re-expressed in per-feature
    # standard deviations, scale-free, and directly readable as "the
    # attacker had to move the transaction this far from normal".
    mean_perturbation_std_units: float = 0.0
    max_perturbation_std_units: float = 0.0


def greedy_evade(
    model: xgb.XGBClassifier,
    X_fraud: pd.DataFrame,
    threshold: float,
    max_steps: int = 30,
    step_std_fraction: float = 0.5,
    max_features_touched: int = 8,
) -> tuple[pd.DataFrame, EvasionResult]:
    """Returns (perturbed_X, result). Only rows the model originally scored
    correctly above `threshold` are targeted; perturbing an already-missed
    fraud transaction isn't a meaningful evasion.

    Vectorized across all targeted rows and all candidate perturbations
    within a step: one batched `predict_proba` call per step (covering
    every active row's every candidate feature/direction at once) instead
    of one call per row per candidate. A first version called `predict_proba`
    on a single-row DataFrame inside a triple-nested loop (row x step x
    candidate), correct, but hundreds of thousands of individual model
    calls, which made a single round already slow and 3-round iterative
    hardening (`iterative_harden`) intractable. This produces the identical
    greedy policy: each row still changes exactly one feature per step, the
    single largest-improvement candidate found that step, just batched.
    """

    feature_names = list(X_fraud.columns)
    n_features = len(feature_names)
    feature_std = X_fraud.std().replace(0, 1.0).to_numpy()
    step_size = feature_std * step_std_fraction

    all_scores = model.predict_proba(X_fraud)[:, 1]
    target_mask = all_scores >= threshold
    targets_df = X_fraud[target_mask]
    X = targets_df.to_numpy(dtype=np.float64)
    original = X.copy()
    n_targets = X.shape[0]

    if n_targets == 0:
        return X_fraud.iloc[0:0].copy(), EvasionResult(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    touched_mask = np.zeros((n_targets, n_features), dtype=bool)
    active = np.ones(n_targets, dtype=bool)

    for _ in range(max_steps):
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break

        current_scores = model.predict_proba(pd.DataFrame(X[active_idx], columns=feature_names))[:, 1]
        evaded_now = current_scores < threshold
        active[active_idx[evaded_now]] = False
        still_idx = active_idx[~evaded_now]
        still_scores = current_scores[~evaded_now]
        if len(still_idx) == 0:
            continue

        trial_rows, trial_row_idx, trial_feat_idx, trial_delta = [], [], [], []
        for row_i in still_idx:
            touched_count = int(touched_mask[row_i].sum())
            candidates = np.where(touched_mask[row_i])[0] if touched_count >= max_features_touched else np.arange(n_features)
            for f in candidates:
                for direction in (-1, 1):
                    delta = direction * step_size[f]
                    trial = X[row_i].copy()
                    trial[f] += delta
                    trial_rows.append(trial)
                    trial_row_idx.append(row_i)
                    trial_feat_idx.append(f)
                    trial_delta.append(delta)

        if not trial_rows:
            active[still_idx] = False
            continue

        trial_scores = model.predict_proba(pd.DataFrame(np.vstack(trial_rows), columns=feature_names))[:, 1]
        trial_row_idx = np.array(trial_row_idx)

        score_map = dict(zip(still_idx.tolist(), still_scores.tolist()))
        best_per_row: dict[int, tuple[float, int, float]] = {}
        for k in range(len(trial_row_idx)):
            row_i = int(trial_row_idx[k])
            s = float(trial_scores[k])
            if row_i not in best_per_row or s < best_per_row[row_i][0]:
                best_per_row[row_i] = (s, trial_feat_idx[k], trial_delta[k])

        for row_i in still_idx:
            entry = best_per_row.get(int(row_i))
            if entry is None:
                active[row_i] = False
                continue
            best_score, best_feat, best_delta = entry
            if best_score < score_map[int(row_i)]:
                X[row_i, best_feat] += best_delta
                touched_mask[row_i, best_feat] = True
            else:
                active[row_i] = False

    final_scores = model.predict_proba(pd.DataFrame(X, columns=feature_names))[:, 1]
    evaded = final_scores < threshold
    features_touched_counts = touched_mask.sum(axis=1)
    delta = X - original
    l2_perturbations = np.linalg.norm(delta, axis=1)
    std_units = np.linalg.norm(delta / feature_std, axis=1)

    perturbed_df = pd.DataFrame(X, columns=feature_names, index=targets_df.index)
    result = EvasionResult(
        n_targeted=n_targets,
        n_evaded=int(evaded.sum()),
        evasion_rate=float(evaded.sum()) / max(n_targets, 1),
        mean_features_perturbed=float(features_touched_counts.mean()),
        mean_l2_perturbation=float(l2_perturbations.mean()),
        mean_perturbation_std_units=float(std_units.mean()),
        max_perturbation_std_units=float(std_units.max()) if std_units.size else 0.0,
    )
    return perturbed_df, result


def adversarial_retrain(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    adversarial_X: pd.DataFrame,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Retrains from scratch with the successfully-evasive adversarial
    examples appended to the training set, still labeled fraud=1 (they ARE
    still fraud: only the feature values moved, not the ground truth)."""

    X_aug = pd.concat([X_train, adversarial_X], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1] * len(adversarial_X))], ignore_index=True)

    scale_pos_weight = (y_aug == 0).sum() / max((y_aug == 1).sum(), 1)
    hardened = xgb.XGBClassifier(**{**model.get_params(), "scale_pos_weight": scale_pos_weight})
    hardened.fit(X_aug, y_aug)
    return hardened


@dataclass
class HardeningRound:
    round: int
    evasion_rate: float
    n_targeted: int
    n_evaded: int
    mean_features_perturbed: float
    mean_l2_perturbation: float
    mean_perturbation_std_units: float
    clean_eval: dict


def iterative_harden(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fraud_eval: pd.DataFrame,
    threshold: float,
    n_rounds: int = 3,
) -> tuple[xgb.XGBClassifier, list[HardeningRound]]:
    """Repeats attack -> augment -> retrain for `n_rounds`, evading the
    CURRENT round's model each time (not the original) so each round tests
    whether the attacker can adapt to what was just hardened. Cumulative
    adversarial examples are kept across rounds; a real defender doesn't
    discard last round's evasive examples just because a new round starts.

    Answers the honest question the single-round result ducked: does
    evasion rate keep dropping round over round, or does it plateau (an
    attacker who can also see the hardened model just finds a new
    direction)? Either answer is a real result worth reporting; this
    doesn't assume improvement, it measures it.
    """

    from janus.common.metrics import evaluate

    current_model = model
    accumulated_adversarial = fraud_eval.iloc[0:0].copy()
    rounds: list[HardeningRound] = []

    for round_num in range(1, n_rounds + 1):
        perturbed_X, evasion = greedy_evade(current_model, fraud_eval, threshold)
        evaded_only = perturbed_X.loc[current_model.predict_proba(perturbed_X)[:, 1] < threshold] if len(perturbed_X) else perturbed_X
        accumulated_adversarial = pd.concat([accumulated_adversarial, evaded_only], ignore_index=True)

        current_model = adversarial_retrain(current_model, X_train, y_train, accumulated_adversarial)
        clean_report = evaluate(y_test.to_numpy(), current_model.predict_proba(X_test)[:, 1])

        rounds.append(
            HardeningRound(
                round=round_num,
                evasion_rate=evasion.evasion_rate,
                n_targeted=evasion.n_targeted,
                n_evaded=evasion.n_evaded,
                mean_features_perturbed=evasion.mean_features_perturbed,
                mean_l2_perturbation=evasion.mean_l2_perturbation,
                mean_perturbation_std_units=evasion.mean_perturbation_std_units,
                clean_eval=clean_report.as_dict(),
            )
        )

    return current_model, rounds


# Persistence (writing results to data/processed/*.json) lives in
# janus/orchestrate/persist.py, not scattered __main__ blocks with
# hardcoded paths, see that module for how this gets invoked and saved.
