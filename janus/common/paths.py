"""Repo-root-anchored paths, overridable via JANUS_DATA_DIR / JANUS_MODELS_DIR.

The predecessor project resolved every data/model path relative to the process's
current working directory, which meant every script silently read or wrote the
wrong place unless launched from the repo root. Anchoring to this file's location
makes path resolution independent of CWD; the env var overrides exist so a
deployment can point at a mounted volume without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("JANUS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("JANUS_MODELS_DIR", REPO_ROOT / "models"))

ULB_PATH = RAW_DIR / "creditcard.csv"
IEEE_TXN_PATH = RAW_DIR / "ieee_cis" / "train_transaction.csv"
IEEE_IDENTITY_PATH = RAW_DIR / "ieee_cis" / "train_identity.csv"
PAYSIM_PATH = RAW_DIR / "paysim" / "paysim.csv"

ATTACKS_YAML = REPO_ROOT / "janus" / "identify" / "attacks.yaml"
GROUNDING_CORPUS_DIR = REPO_ROOT / "janus" / "identify" / "corpus"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
