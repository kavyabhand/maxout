"""The feature-exclusion contract.

`Time` used to reach the classifier, the fidelity distinguisher and the
evasion attacker simply because every one of them built its column list as
"everything except the label". These lock in that it does not come back by
that route again; the failure it caused was silent in all three places.
"""

import pandas as pd

from janus.common.features import ULB_EXCLUDED_COLS, ulb_feature_cols


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"Time": [0.0, 1.0], "V1": [0.1, 0.2], "Amount": [5.0, 9.0], "Class": [0, 1]})


def test_excludes_label_and_capture_clock():
    assert ulb_feature_cols(_frame()) == ["V1", "Amount"]


def test_keeps_amount():
    # Amount is genuinely available at authorization time; only the
    # capture-relative clock is dropped, and confusing the two would
    # quietly remove a real signal.
    assert "Amount" in ulb_feature_cols(_frame())
    assert "Amount" not in ULB_EXCLUDED_COLS


def test_preserves_frame_order():
    df = _frame()[["Amount", "Class", "V1", "Time"]]
    assert ulb_feature_cols(df) == ["Amount", "V1"]


def test_absent_excluded_column_is_not_an_error():
    df = _frame().drop(columns=["Time"])
    assert ulb_feature_cols(df) == ["V1", "Amount"]
