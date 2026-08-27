"""Which raw columns are legitimate model inputs: decided once, here.

ULB's `Time` column is "seconds elapsed since the first transaction in
this particular two-day capture". It is a property of how the dataset was
cut, not of a transaction: no live authorization scorer has an equivalent
value available at decision time, and any model that leans on it is
partly fitting the capture window rather than fraud. Leaving it in also
quietly corrupted two things that had nothing to do with the classifier:

- The fidelity distinguisher (persist.py passed `feature_cols[:8]`, which
  began with `Time`). A Gaussian copula cannot reproduce ULB's bimodal
  two-day activity curve, so part of the real-vs-synthetic AUC was the
  distinguisher spotting a capture artifact rather than a generator flaw.
- The black-box evader (evasion.py steps by `feature_std * 0.5`). ULB's
  `Time` has a standard deviation of ~47,000 seconds, so a single step
  moved the timestamp by over six hours and dominated the reported L2
  perturbation budget, a number that then meant nothing.

`Amount` stays: it is genuinely available at authorization time and is a
real fraud signal. Only the capture-relative clock is dropped.
"""

from __future__ import annotations

import pandas as pd

ULB_LABEL_COL = "Class"

#: Dropped as non-causal capture artifacts rather than as weak features.
ULB_EXCLUDED_COLS: tuple[str, ...] = ("Time",)


def ulb_feature_cols(df: pd.DataFrame) -> list[str]:
    """The ULB columns a live scorer could actually see, in frame order."""

    excluded = {ULB_LABEL_COL, *ULB_EXCLUDED_COLS}
    return [c for c in df.columns if c not in excluded]
