"""Runs every pillar's pipelines and writes results to data/processed/ as
versioned JSON artifacts the FastAPI backend serves. This is the one place
that decides what gets persisted and under what filename/key structure --
individual generate/defend modules return plain dataclasses/dicts and
never touch the filesystem themselves (see e.g. evasion.py's removed
__main__ block), so this module is the single source of truth for the
on-disk contract.

Each persist_* function is independent and safe to re-run individually;
`if __name__ == "__main__"` runs the full set in dependency order and
prints progress, since the heaviest steps (GNN training, the closed loop)
take real wall-clock time.
"""

from __future__ import annotations

import json
import time

from janus.common import paths


def _save(filename: str, data) -> None:
    paths.ensure_dirs()
    out_path = paths.PROCESSED_DIR / filename
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"wrote {out_path}")


def persist_gbm_baseline() -> None:
    from janus.data.load import load_ulb
    from janus.defend.gbm import train_ulb_baseline

    df = load_ulb()
    trained = train_ulb_baseline(df)
    _save("defend_gbm_ulb.json", {
        **trained.eval_report.as_dict(),
        "train_latency_s": trained.train_latency_s,
        "inference_ms_per_1000_rows": trained.inference_latency_ms_per_1k,
        "dataset": "ULB European Credit Card Fraud (David-Egea/Creditcard-fraud-detection mirror)",
        "n_rows": len(df), "n_fraud": int(df["Class"].sum()),
    })


def persist_gnn_hybrid(sample_frac: float | None = None, epochs: int = 60) -> None:
    from janus.data.load import load_ieee_cis
    from janus.defend.gnn import train_gnn_hybrid

    df = load_ieee_cis(sample_frac=sample_frac)
    result = train_gnn_hybrid(df, epochs=epochs)
    _save("defend_gnn_hybrid.json", {
        "gnn_only": result.gnn_eval.as_dict(),
        "hybrid": result.hybrid_eval.as_dict(),
        "device": result.device, "n_nodes": result.n_nodes, "n_edges": result.n_edges, "epochs": result.epochs,
        "dataset": "IEEE-CIS Fraud Detection (aliceczr/ieee-fraud-detection mirror)",
        "n_rows": len(df), "n_fraud": int(df["isFraud"].sum()), "sample_frac": sample_frac,
    })


def persist_mule_ring() -> None:
    from janus.data.load import load_paysim
    from janus.generate.graph.detect import train_mule_detector
    from janus.generate.graph.mule_ring import inject_benign_bidirectional_accounts, inject_mule_rings

    sample = load_paysim(sample_n=300_000)
    augmented, ring_meta = inject_mule_rings(sample, n_rings=60)
    augmented = inject_benign_bidirectional_accounts(augmented, n_accounts=300)
    model, report = train_mule_detector(augmented, ring_meta)
    _save("generate_graph_mule_ring.json", {
        **report.as_dict(), "n_injected_rings": len(ring_meta),
        "caveat": "small test-positive count; see generalization check for out-of-distribution validation",
    })


def persist_mule_generalization() -> None:
    from janus.generate.graph.generalization_check import run_generalization_check

    _save("generate_graph_mule_generalization.json", run_generalization_check())


def persist_voice_scam() -> None:
    from janus.data.load import load_paysim
    from janus.generate.sequence.voice_scam import inject_voice_scam_transactions, train_voice_scam_detector

    sample = load_paysim(sample_n=150_000)
    augmented = inject_voice_scam_transactions(sample, n_scams=200)
    model, report = train_voice_scam_detector(augmented)
    _save("generate_sequence_voice_scam.json", report.as_dict())


def persist_sequence_transformer() -> None:
    from janus.defend.sequence import train_sequence_transformer
    from janus.generate.sequence.simulator import simulate_population

    all_seqs = simulate_population()
    split = int(len(all_seqs) * 0.7)
    train_seqs, test_seqs = all_seqs[:split], all_seqs[split:]
    result = train_sequence_transformer(train_seqs, test_seqs)
    _save("defend_sequence_transformer.json", {
        **result.eval_report.as_dict(),
        "n_train": len(train_seqs), "n_test": len(test_seqs),
    })


def persist_fidelity_scorecards() -> None:
    from janus.common.features import ulb_feature_cols
    from janus.data.load import load_paysim, load_ulb
    from janus.generate.fidelity.scorer import score_batch
    from janus.generate.graph.mule_ring import build_transfer_graph, inject_benign_bidirectional_accounts, inject_mule_rings
    from janus.generate.sequence.voice_scam import inject_voice_scam_transactions
    from janus.generate.tabular.synthesizer import select_n_components, synthesize_conditional

    scorecards = []

    ulb = load_ulb()
    feature_cols = ulb_feature_cols(ulb)
    scored_cols = feature_cols[:12]

    # The mixture size is measured per class rather than assumed; the two
    # classes differ by three orders of magnitude in row count, so no single
    # value is right for both. Selection runs on its own split and its own
    # seeds (see select_n_components); the batch scored below is generated
    # fresh afterwards, so the reported AUC is not the best of five draws.
    selection = {}
    for label, key in ((0, "legit"), (1, "fraud")):
        chosen = select_n_components(ulb[ulb["Class"] == label], scored_cols)
        selection[key] = chosen.as_dict()
        print(f"  [{key}] selected n_components={chosen.n_components} from {chosen.trials}")

    synth = synthesize_conditional(
        ulb, feature_cols, "Class", n_rows=20_000, fraud_ratio=0.05,
        n_components={
            0: selection["legit"]["selected_n_components"],
            1: selection["fraud"]["selected_n_components"],
        },
        random_state=1234,
    )

    # Scored PER CLASS, which is the only like-for-like comparison for a
    # conditional generator. The previous version compared a synthetic
    # batch generated at a 5% fraud ratio against a real sample carrying
    # ULB's natural 0.172%, so a large part of what the distinguisher was
    # detecting, and most of the correlation delta, since mixing two
    # classes with different means induces correlation that is an artifact
    # of the mixing ratio rather than of either population, was the class
    # mix, which is a parameter of the request and not a property of the
    # generator. Holding the class fixed measures what was actually meant:
    # given that this row is fraud, does it look like real fraud?
    #
    # `Time` is also excluded throughout (janus/common/features.py): a
    # copula cannot reproduce ULB's bimodal two-day activity curve, so
    # leaving it in charged the generator for a capture artifact.
    for label, name in ((0, "tabular_synthesis_ulb_legit"), (1, "tabular_synthesis_ulb_fraud")):
        real_class = ulb[ulb["Class"] == label]
        synth_class = synth.data[synth.data["Class"] == label]
        n = min(len(real_class), len(synth_class), 5000)
        if n < 20:
            continue
        scorecards.append(
            {
                **score_batch(
                    name,
                    real_class.sample(n, random_state=1),
                    synth_class.sample(n, random_state=1),
                    scored_cols,
                ).as_dict(),
                "component_selection": selection["fraud" if label else "legit"],
            }
        )

    paysim_sample = load_paysim(sample_n=300_000)
    augmented, _ = inject_mule_rings(paysim_sample, n_rings=40)
    augmented = inject_benign_bidirectional_accounts(augmented, n_accounts=300)
    real_legit = paysim_sample[(paysim_sample["isFraud"] == 0) & (paysim_sample["type"] == "TRANSFER")]
    synthetic_injected = augmented[augmented["ring_id"].notna()]
    scorecards.append(
        score_batch(
            "graph_mule_ring_amounts",
            real_legit[["amount"]],
            synthetic_injected[["amount"]],
            ["amount"],
            real_graph=build_transfer_graph(paysim_sample[paysim_sample["type"] == "TRANSFER"]),
            synthetic_graph=build_transfer_graph(synthetic_injected),
        ).as_dict()
    )

    voice_augmented = inject_voice_scam_transactions(load_paysim(sample_n=150_000), n_scams=200)
    real_fraud_transfer = voice_augmented[(voice_augmented["isFraud"] == 1) & (voice_augmented["scam_type"].isna())][["amount"]]
    synthetic_scam = voice_augmented[voice_augmented["scam_type"] == "voice_clone_app_fraud"][["amount"]]
    if len(real_fraud_transfer) > 5:
        scorecards.append(score_batch("sequence_voice_scam_amounts", real_fraud_transfer, synthetic_scam, ["amount"]).as_dict())

    _save("generate_fidelity_scorecards.json", scorecards)


def persist_closed_loop(n_rounds: int = 5, attempts_per_round: int = 2) -> None:
    from janus.orchestrate.loop import combined_curve, run_agentic_closed_loop, run_tabular_adversarial_loop

    agentic_rounds = run_agentic_closed_loop(n_rounds=n_rounds, attempts_per_round=attempts_per_round)
    tabular_result = run_tabular_adversarial_loop(n_rounds=3)
    curve = combined_curve(agentic_rounds, tabular_result)

    _save("orchestrate_closed_loop.json", {
        "generated_at": time.time(),
        "agentic_rounds": [r.as_dict() for r in agentic_rounds],
        "tabular_adversarial": tabular_result,
        "combined_curve": curve,
    })


def persist_identity_onboarding() -> None:
    """Category A (Identity & Onboarding); the one taxonomy category that
    previously had no generator and no detector behind any of its four
    entries. See janus/generate/identity/synthetic_identity.py for why the
    hard-negative thin-file population and the per-sophistication recall
    split are the whole point rather than extra credit."""

    from janus.generate.identity.synthetic_identity import generate_onboarding_population, train_onboarding_detector

    population = generate_onboarding_population()
    result = train_onboarding_detector(population.applications)
    _save("generate_identity_onboarding.json", {**result.as_dict(), "population": population.as_dict()})


def persist_meta_ensemble(sample_frac: float | None = None, gnn_epochs: int = 60) -> None:
    """The stacked five-family decision, measured end-to-end for the first
    time, see janus/orchestrate/ensemble.py for the three-split protocol."""

    from janus.data.load import load_ieee_cis
    from janus.orchestrate.ensemble import run_ieee_ensemble

    df = load_ieee_cis(sample_frac=sample_frac)
    result = run_ieee_ensemble(df, gnn_epochs=gnn_epochs)
    _save("defend_meta_ensemble.json", {
        **result.as_dict(),
        "dataset": "IEEE-CIS Fraud Detection (aliceczr/ieee-fraud-detection mirror)",
        "n_rows": len(df), "n_fraud": int(df["isFraud"].sum()), "sample_frac": sample_frac,
    })


def persist_explanations(n_examples: int = 6) -> None:
    """SHAP reason codes for the GBM fast path. `janus/defend/explain.py`
    was importable but had no caller anywhere in the repo, so no
    explanation ever reached the UI; a gap worth closing on its own
    terms, since a decline a risk analyst cannot read a reason for is not
    deployable under EU AI Act-era expectations."""

    import numpy as np

    from janus.data.load import load_ulb
    from janus.defend.explain import explain
    from janus.defend.gbm import train_ulb_baseline

    df = load_ulb()
    trained = train_ulb_baseline(df)
    features = trained.feature_names

    scored = df[features]
    scores = trained.model.predict_proba(scored)[:, 1]
    # Explain the highest-risk rows: those are the declines that would
    # actually need a reason code attached in production.
    top_idx = np.argsort(scores)[::-1][:n_examples]
    sample = scored.iloc[top_idx]
    result = explain(trained.model, sample)

    global_importance = np.abs(result.shap_values).mean(axis=0)
    _save("defend_explanations.json", {
        "model": "gbm_ulb",
        "base_value": result.base_value,
        "global_mean_abs_shap": [
            {"feature": name, "mean_abs_shap": round(float(v), 6)}
            for name, v in sorted(zip(features, global_importance), key=lambda t: t[1], reverse=True)
        ],
        "examples": [
            {
                "risk_score": round(float(scores[idx]), 6),
                "true_label": int(df["Class"].iloc[idx]),
                "top_reasons": [
                    {"feature": f, "shap": round(float(v), 6)}
                    for f, v in result.top_features_for_row(i, k=5)
                ],
            }
            for i, idx in enumerate(top_idx)
        ],
        "note": (
            "SHAP values are in log-odds contribution units against the model's base value. "
            "ULB's V1-V28 are PCA components with no published loadings, so a component name is "
            "not a human-readable reason on this dataset, what this demonstrates is the "
            "mechanism and its cost, which carries over to a feature set with real column names."
        ),
    })


def persist_time_leakage_ablation() -> None:
    """Measures what the excluded `Time` column was actually worth to the
    classifier, so dropping it is a reported decision with a number behind
    it rather than an unverified assertion. Both arms are otherwise the
    identical recipe on the identical split."""

    from janus.data.load import load_ulb
    from janus.defend.gbm import train_ulb_baseline

    df = load_ulb()

    # train_ulb_baseline drops `Time` by name, so the "with" arm renames the
    # column past that exclusion to put the identical values back in front
    # of the identical model.
    with_time = train_ulb_baseline(df.rename(columns={"Time": "capture_clock"}))
    without_time = train_ulb_baseline(df)

    _save("defend_time_leakage_ablation.json", {
        "with_capture_clock": {
            **with_time.eval_report.as_dict(),
            "n_features": len(with_time.feature_names),
        },
        "without_capture_clock": {
            **without_time.eval_report.as_dict(),
            "n_features": len(without_time.feature_names),
        },
        "note": (
            "ULB's `Time` is seconds since the first row of one particular two-day capture. No "
            "live authorization scorer has an equivalent value, so any lift it provides is lift "
            "the deployed model would not get. Reported here so the exclusion is a measured "
            "decision rather than an assertion. Both arms use the identical recipe, split and seed."
        ),
    })


def persist_atlas_coverage() -> None:
    from janus.identify.atlas import AttackAtlas

    atlas = AttackAtlas()
    _save("identify_atlas.json", atlas.to_force_graph())
    _save("identify_coverage.json", atlas.coverage_summary())


if __name__ == "__main__":
    import sys

    steps = {
        "atlas": persist_atlas_coverage,
        "gbm": persist_gbm_baseline,
        "gnn": persist_gnn_hybrid,
        "mule": persist_mule_ring,
        "mule_generalization": persist_mule_generalization,
        "voice": persist_voice_scam,
        "sequence": persist_sequence_transformer,
        "fidelity": persist_fidelity_scorecards,
        "identity": persist_identity_onboarding,
        "meta_ensemble": persist_meta_ensemble,
        "explanations": persist_explanations,
        "time_ablation": persist_time_leakage_ablation,
        "closed_loop": persist_closed_loop,
    }

    requested = sys.argv[1:] or list(steps.keys())
    for name in requested:
        print(f"\n=== {name} ===")
        started = time.time()
        steps[name]()
        print(f"({name} took {time.time() - started:.1f}s)")
