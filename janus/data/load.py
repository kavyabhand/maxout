"""Loaders for the three benchmark datasets, downloaded credential-free via
janus.data.download. Paths resolve through janus.common.paths (repo-root
anchored, JANUS_DATA_DIR-overridable) rather than CWD-relative strings.
"""

from __future__ import annotations

import pandas as pd

from janus.common import paths


def load_ulb() -> pd.DataFrame:
    """ULB European Credit Card Fraud dataset. 284,807 transactions, 492
    fraud (0.172%), PCA-anonymized V1-V28 + Time + Amount + Class. No
    entity/graph fields, so this is: (1) the primary statistical-fidelity
    calibration reference, and (2) a clean baseline for the GBM defend
    family in isolation before GNN embeddings are concatenated in."""

    return pd.read_csv(paths.ULB_PATH)


def load_ieee_cis(sample_frac: float | None = None, random_state: int = 42) -> pd.DataFrame:
    """IEEE-CIS Fraud Detection. ~590k transactions, richer categorical/
    identity features (card1-6, addr1-2, P_emaildomain, DeviceInfo); these
    entity fields are what make real graph construction possible (shared
    card/email/device -> edges), unlike ULB's fully-anonymized features.
    This is the dataset the GNN defend family trains on."""

    txn = pd.read_csv(paths.IEEE_TXN_PATH)
    identity = pd.read_csv(paths.IEEE_IDENTITY_PATH)
    df = txn.merge(identity, on="TransactionID", how="left")
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)
    return df


def load_paysim(sample_n: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """PaySim mobile-money simulator. 6,362,620 rows, the background
    legitimate population the graph and sequence generators inject
    synthetic fraud topology/behavior on top of."""

    df = pd.read_csv(paths.PAYSIM_PATH)
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=random_state)
    return df
