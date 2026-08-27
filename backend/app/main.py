"""JANUS FastAPI service, serves the Attack Atlas, cached defend-family
metrics, fidelity scorecards, and closed-loop history; runs a live
agentic sandbox session over WebSocket; and exposes a "run one more round"
endpoint for the Red vs Blue Arena screen. Every GET route degrades to a
clear "not yet computed" (null) rather than erroring when its backing
artifact under data/processed/ doesn't exist yet, so the frontend is
demoable the moment the repo is cloned, before any pipeline has run.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from janus.common import paths
from janus.defend.firewall import MandateFirewall
from janus.generate.agentic.catalog import seed_catalog
from janus.generate.agentic.credentials_provider import CredentialsProviderAgent, seed_accounts
from janus.generate.agentic.divergence import score
from janus.generate.agentic.red_team import TECHNIQUES, generate_payload
from janus.generate.agentic.shopping_agent import TranscriptEvent, run_shopping_session
from janus.identify.atlas import AttackAtlas

app = FastAPI(title="JANUS API")

# Prototype-scope CORS: wide open so the frontend can hit this from any
# origin during judging/demo without extra config. Not a production
# posture, see README for hardening notes before any real deployment.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _load_json(filename: str):
    path = paths.PROCESSED_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/identify/atlas")
def identify_atlas():
    cached = _load_json("identify_atlas.json")
    if cached is not None:
        return cached
    return AttackAtlas().to_force_graph()


@app.post("/api/identify/propose")
def identify_propose(topic_query: str):
    from janus.identify.agent import IdentifyAgent

    agent = IdentifyAgent()
    return agent.propose_new_attack(topic_query)


@app.get("/api/identify/coverage")
def identify_coverage():
    cached = _load_json("identify_coverage.json")
    if cached is not None:
        return cached
    return AttackAtlas().coverage_summary()


@app.get("/api/defend/gbm")
def defend_gbm():
    return _load_json("defend_gbm_ulb.json")


@app.get("/api/defend/gnn")
def defend_gnn():
    return _load_json("defend_gnn_hybrid.json")


@app.get("/api/defend/sequence")
def defend_sequence():
    return _load_json("defend_sequence_transformer.json")


@app.get("/api/defend/ensemble")
def defend_ensemble():
    """The stacked five-family decision, its per-member breakdown, the tier
    distribution, and a faithful slice of the scored held-out split that the
    prototype's authorization stream replays."""

    return _load_json("defend_meta_ensemble.json")


@app.get("/api/defend/explanations")
def defend_explanations():
    return _load_json("defend_explanations.json")


@app.get("/api/defend/latency")
def defend_latency():
    return _load_json("defend_latency_profile.json")


@app.get("/api/defend/time-ablation")
def defend_time_ablation():
    return _load_json("defend_time_leakage_ablation.json")


@app.get("/api/generate/identity-onboarding")
def generate_identity_onboarding():
    return _load_json("generate_identity_onboarding.json")


@app.get("/api/generate/mule-ring")
def generate_mule_ring():
    return _load_json("generate_graph_mule_ring.json")


@app.get("/api/generate/mule-ring/generalization")
def generate_mule_generalization():
    return _load_json("generate_graph_mule_generalization.json")


@app.get("/api/generate/voice-scam")
def generate_voice_scam():
    return _load_json("generate_sequence_voice_scam.json")


_ULB_CACHE = None


@app.post("/api/generate/simulate")
def generate_simulate(n_rows: int = 3000, fraud_ratio: float = 0.05):
    """Simulation Studio's live batch: fits a Gaussian-copula synthesizer
    on real ULB data (conditional per class) and scores the resulting
    batch's fidelity against a held-out real sample.

    By design, this deployment never keeps raw datasets on the machine
    running the backend (see janus.data.download's docstring and
    janus/orchestrate/persist.py); those only ever exist transiently on
    remote compute (a Kaggle kernel) that writes back small JSON results.
    So this endpoint is live only in an environment where JANUS_DATA_DIR
    actually has the raw ULB csv staged (e.g. inside that Kaggle run
    itself); everywhere else it degrades to a clear "unavailable" response
    rather than downloading gigabytes of data on demand. The cached
    scorecards from /api/generate/fidelity are the real, persisted proof
    of fidelity for this deployment."""

    global _ULB_CACHE
    from janus.common import paths

    if not paths.ULB_PATH.exists():
        return {
            "available": False,
            "reason": (
                "Raw ULB dataset is not staged on this backend. Live batch generation only runs "
                "inside the remote pipeline environment (see janus/orchestrate/persist.py); this "
                "deployment serves the cached scorecards it produced instead."
            ),
        }

    from janus.data.load import load_ulb
    from janus.generate.fidelity.scorer import score_batch
    from janus.generate.tabular.synthesizer import synthesize_conditional

    if _ULB_CACHE is None:
        _ULB_CACHE = load_ulb()
    ulb = _ULB_CACHE

    n_rows = max(200, min(n_rows, 20000))
    fraud_ratio = max(0.001, min(fraud_ratio, 0.5))
    feature_cols = [c for c in ulb.columns if c != "Class"][:8]

    synth = synthesize_conditional(ulb, [c for c in ulb.columns if c != "Class"], "Class", n_rows=n_rows, fraud_ratio=fraud_ratio)
    real_sample = ulb.sample(min(n_rows, len(ulb)), random_state=7)
    card = score_batch(f"live_{n_rows}_{fraud_ratio}", real_sample, synth.data, feature_cols)

    return {
        "available": True,
        "scorecard": card.as_dict(),
        "n_legit": synth.n_legit,
        "n_fraud": synth.n_fraud,
        "fraud_ratio": synth.fraud_ratio,
    }


@app.get("/api/generate/fidelity")
def generate_fidelity():
    return _load_json("generate_fidelity_scorecards.json")


@app.get("/api/orchestrate/closed-loop")
def orchestrate_closed_loop():
    return _load_json("orchestrate_closed_loop.json")


@app.get("/api/orchestrate/latency-budget")
def orchestrate_latency_budget():
    from janus.defend.meta import latency_budget_report

    return latency_budget_report()


def run_sandbox_session(technique_name: str, firewall_enabled: bool) -> dict:
    """One complete sandbox run, returned in full rather than streamed.

    The WebSocket route below streams the same events, but the diagram that
    consumes them stages its own reveal client-side at a fixed cadence
    anyway, it never actually rendered at socket speed. Returning the
    whole transcript from a plain POST therefore looks identical on screen
    and drops the only stateful, long-lived connection in the app, which is
    what let the prototype be deployed as static frontend plus recorded
    transcripts with no behavioural difference.
    """

    if technique_name not in TECHNIQUES:
        return {"error": f"unknown technique {technique_name!r}"}

    from janus.common.llm import SHOPPING_AGENT_MODEL, get_backend

    backend = get_backend()
    technique = TECHNIQUES[technique_name]
    payload, reasoning = generate_payload(technique, [])

    catalog = seed_catalog()
    catalog.poison(technique.target_item_id, payload)
    scenario = technique.build_scenario(catalog)
    creds = CredentialsProviderAgent(seed_accounts(), strict_account_binding=False)
    firewall = MandateFirewall() if firewall_enabled else None

    events: list[dict] = []
    result = run_shopping_session(
        session_user_id=technique.session_user_id,
        user_message=scenario.user_message,
        open_mandate=scenario.open_mandate,
        catalog=catalog,
        credentials_provider=creds,
        firewall=firewall,
        on_event=lambda e: events.append(_event_to_dict(e)),
    )
    divergence = score(scenario, result)

    return {
        "technique": technique_name,
        "firewall_enabled": firewall_enabled,
        # Which agent this ran against. The Hardening screen reports a
        # campaign against a frontier shopping agent, and this sandbox runs
        # against whichever backend is configured; without saying so, the
        # two screens can look like they disagree about the same technique.
        "backend": backend.name,
        "agent": SHOPPING_AGENT_MODEL if backend.name == "openai" else "scripted policy agent",
        "events": [
            {"type": "red_team_payload", "payload": payload, "reasoning": reasoning},
            *events,
            {
                "type": "result",
                "attack_succeeded": divergence.attack_succeeded,
                "incomplete": divergence.incomplete,
                "notes": divergence.notes,
                "firewall_events": (
                    [{"stage": e.stage, "verdict": e.verdict.value, "reasons": e.reasons} for e in firewall.events]
                    if firewall
                    else []
                ),
            },
        ],
    }


@app.post("/api/sandbox/run")
def sandbox_run(technique: str = "branded_whisper", firewall_enabled: bool = False):
    return run_sandbox_session(technique, firewall_enabled)


def _event_to_dict(event: TranscriptEvent) -> dict:
    return {
        "type": "transcript_event",
        "role": event.role,
        "content": event.content,
        "tool_name": event.tool_name,
        "tool_args": event.tool_args,
    }


@app.websocket("/ws/sandbox")
async def sandbox_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        params = await websocket.receive_json()
    except Exception:
        await websocket.close()
        return

    technique_name = params.get("technique", "branded_whisper")
    firewall_enabled = params.get("firewall_enabled", False)

    if technique_name not in TECHNIQUES:
        await websocket.send_json({"type": "error", "message": f"unknown technique {technique_name!r}"})
        await websocket.close()
        return

    import anyio

    technique = TECHNIQUES[technique_name]
    payload, reasoning = generate_payload(technique, [])
    await websocket.send_json({"type": "red_team_payload", "payload": payload, "reasoning": reasoning})

    catalog = seed_catalog()
    catalog.poison(technique.target_item_id, payload)
    scenario = technique.build_scenario(catalog)
    creds = CredentialsProviderAgent(seed_accounts(), strict_account_binding=False)
    firewall = MandateFirewall() if firewall_enabled else None

    def on_event(event: TranscriptEvent) -> None:
        anyio.from_thread.run(websocket.send_json, _event_to_dict(event))

    try:
        result = await anyio.to_thread.run_sync(
            lambda: run_shopping_session(
                session_user_id=technique.session_user_id,
                user_message=scenario.user_message,
                open_mandate=scenario.open_mandate,
                catalog=catalog,
                credentials_provider=creds,
                firewall=firewall,
                on_event=on_event,
            )
        )
        divergence = score(scenario, result)
        firewall_events = (
            [{"stage": e.stage, "verdict": e.verdict.value, "reasons": e.reasons} for e in firewall.events]
            if firewall
            else []
        )
        await websocket.send_json({
            "type": "result",
            "attack_succeeded": divergence.attack_succeeded,
            "incomplete": divergence.incomplete,
            "notes": divergence.notes,
            "firewall_events": firewall_events,
        })
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.post("/api/orchestrate/run-round")
def run_round(technique: str = "branded_whisper", firewall_enabled: bool = True):
    """Runs one fresh agentic attack round live (scripted/local/openai
    backend, whichever is configured) and returns the outcome; the "run
    next round" control on the Red vs Blue Arena screen. Deliberately
    lightweight (a single attempt, not a full multi-round campaign) so it
    responds interactively."""

    from janus.generate.agentic.red_team import run_attack_round

    if technique not in TECHNIQUES:
        return {"error": f"unknown technique {technique!r}"}

    firewall = MandateFirewall() if firewall_enabled else None
    attempt = run_attack_round(technique, 0, [], firewall=firewall)
    return {
        "technique": technique,
        "firewall_enabled": firewall_enabled,
        "attack_succeeded": attempt.divergence.attack_succeeded,
        "incomplete": attempt.divergence.incomplete,
        "notes": attempt.divergence.notes,
        "payload": attempt.payload,
        "reasoning": attempt.reasoning,
    }
