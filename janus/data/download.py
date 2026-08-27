"""Credential-free dataset acquisition.

The predecessor project fetched these three benchmarks through the Kaggle
API, which requires an authenticated kaggle.json. That's a real barrier to
"a judge clones this and reproduces it", Kaggle's IEEE-CIS competition
dataset specifically requires accepting competition rules through the web
UI first, which the API can't do on anyone's behalf.

All three datasets are also mirrored as public HuggingFace Hub datasets,
which serve anonymously over plain HTTPS with no account, no token, and no
rate-limit wall for public repos. Verified against the actual CSV headers
before wiring this up (not just repo names):

    ULB European Credit Card Fraud   David-Egea/Creditcard-fraud-detection
        Time,V1..V28,Amount,Class, 284,807 rows, 492 fraud (0.172%)
    IEEE-CIS Fraud Detection         aliceczr/ieee-fraud-detection
        TransactionID,isFraud,TransactionDT,...: 590,540 rows
    PaySim mobile-money simulator    theman10/paysim
        step,type,amount,nameOrig,..., 6,362,620 rows

Each download is verified by exact row count and fraud count after the
fact; a truncated or wrong file fails loudly here rather than silently
skewing every downstream metric.
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass

from janus.common import paths

_CHUNK = 1 << 20  # 1 MiB


@dataclass
class DatasetSpec:
    name: str
    url: str
    dest: object  # pathlib.Path
    expected_rows: int
    expected_positive: int
    label_col: str


ULB = DatasetSpec(
    name="ulb_creditcard",
    url="https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv",
    dest=paths.ULB_PATH,
    expected_rows=284_807,
    expected_positive=492,
    label_col="Class",
)

IEEE_TXN = DatasetSpec(
    name="ieee_train_transaction",
    url="https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/train_transaction.csv",
    dest=paths.IEEE_TXN_PATH,
    expected_rows=590_540,
    expected_positive=20_663,
    label_col="isFraud",
)

IEEE_IDENTITY = DatasetSpec(
    name="ieee_train_identity",
    url="https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/train_identity.csv",
    dest=paths.IEEE_IDENTITY_PATH,
    expected_rows=144_233,
    expected_positive=-1,  # no label column in this file
    label_col="",
)

PAYSIM = DatasetSpec(
    name="paysim",
    url="https://huggingface.co/datasets/theman10/paysim/resolve/main/paysim.csv",
    dest=paths.PAYSIM_PATH,
    expected_rows=6_362_620,
    expected_positive=8_213,
    label_col="isFraud",
)

ALL_DATASETS = [ULB, IEEE_TXN, IEEE_IDENTITY, PAYSIM]


def _download(spec: DatasetSpec, *, force: bool = False) -> None:
    spec.dest.parent.mkdir(parents=True, exist_ok=True)
    if spec.dest.exists() and not force:
        print(f"[{spec.name}] already present at {spec.dest}, skipping (pass force=True to refetch)")
        return

    print(f"[{spec.name}] downloading from {spec.url}")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "janus-fraud-lab/1.0"})
    tmp_path = spec.dest.with_suffix(spec.dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            written += len(chunk)
            if total:
                pct = 100 * written / total
                print(f"\r[{spec.name}] {written/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="", flush=True)
    print()
    tmp_path.replace(spec.dest)


def _verify(spec: DatasetSpec) -> None:
    import pandas as pd

    df = pd.read_csv(spec.dest)
    if len(df) != spec.expected_rows:
        raise ValueError(
            f"[{spec.name}] row count mismatch: expected {spec.expected_rows:}, got {len(df):}. "
            "The download is likely truncated or the mirror changed, delete the file and retry."
        )
    if spec.label_col:
        n_positive = int(df[spec.label_col].sum())
        if n_positive != spec.expected_positive:
            raise ValueError(
                f"[{spec.name}] fraud-label count mismatch: expected {spec.expected_positive:}, got {n_positive:}."
            )
    print(f"[{spec.name}] verified: {len(df):} rows" + (f", {spec.expected_positive:} positive" if spec.label_col else ""))


def download_all(*, force: bool = False, verify: bool = True) -> None:
    paths.ensure_dirs()
    for spec in ALL_DATASETS:
        _download(spec, force=force)
        if verify:
            _verify(spec)


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_all(force=force)
