"""Merges the two arms of the closed loop into one artifact.

They are produced in different places for a reason. The tabular arm trains
gradient-boosted models against 285k rows and runs a black-box evader
against them, so it runs on remote compute. The agentic arm is a frontier
model reasoning about what the defense caught last round, pure network
I/O, no dataset, and it needs an API key, which the remote notebooks do not
have and should not be given. This joins the two results and recomputes the
combined curve the Hardening screen plots.

Usage:
    python scripts/merge_closed_loop.py <agentic.json> <tabular.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from janus.common import paths  # noqa: E402


def main() -> None:
    agentic_path, tabular_path = Path(sys.argv[1]), Path(sys.argv[2])
    agentic = json.loads(agentic_path.read_text())
    tabular = json.loads(tabular_path.read_text())

    rounds = agentic["rounds"]
    points = []
    for r in rounds:
        rate = r.get("overall_bypass_rate")
        points.append({
            "round": r["round_num"],
            "arm": "agentic",
            "attacker_win_rate": rate,
            "defense_strength": None if rate is None else round(1 - rate, 4),
            "note": "defense_strength = 1 - overall bypass rate",
        })
    for r in tabular.get("rounds", []):
        points.append({
            "round": r["round"],
            "arm": "tabular_adversarial",
            "attacker_win_rate": round(r["evasion_rate"], 4),
            "defense_strength": round(r["clean_eval"]["pr_auc"], 4),
            "note": (
                "defense_strength = held-out clean PR-AUC (a different axis than evasion rate, "
                "not directly comparable to bypass rate)"
            ),
        })

    fallbacks = sum(
        stats.get("template_fallbacks", 0)
        for r in rounds
        for stats in r["technique_stats"].values()
    )
    attempts = sum(stats["attempts"] for r in rounds for stats in r["technique_stats"].values())

    payload = {
        "generated_at": time.time(),
        "agentic_rounds": rounds,
        "agentic_meta": {
            "backend": agentic.get("backend"),
            "red_team_model": agentic.get("red_team_model"),
            "shopping_agent_model": agentic.get("shopping_agent_model"),
            "wall_clock_s": agentic.get("wall_clock_s"),
            "llm_call_summary": agentic.get("llm_call_summary"),
            "total_attempts": attempts,
            "template_fallbacks": fallbacks,
            "provenance": (
                "The agentic arm runs against a live frontier model and is executed separately from "
                "the remote compute notebooks, which carry no API key by design. "
                f"{fallbacks} of {attempts} payload generations were refused by the provider's own "
                "safety classifier and fell back to the deterministic template library; those attempts "
                "are counted in the bypass rate and are not model-authored."
            ),
        },
        "tabular_adversarial": tabular,
        "combined_curve": points,
    }

    out = paths.PROCESSED_DIR / "orchestrate_closed_loop.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")
    print(f"  agentic: {len(rounds)} rounds, {attempts} attempts, {fallbacks} template fallbacks")
    print(f"  tabular: {len(tabular.get('rounds', []))} rounds")


if __name__ == "__main__":
    main()
