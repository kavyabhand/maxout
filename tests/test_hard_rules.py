"""Tests for janus.defend.hard_rules; the actual
real-time defense functions the firewall calls (as opposed to
test_mandate_rules.py, which tests the shared constraint-checking logic
both the defense and the ground-truth scorer depend on). These tests check
the HardRuleResult / Verdict mapping specifically, i.e. that a violation
detected by mandate_rules actually turns into a BLOCK verdict with a
useful reason, not just that the underlying boolean is correct.
"""

from __future__ import annotations

import time

from janus.common.schemas import (
    AcceptableItem,
    CheckoutJWTPayload,
    CheckoutMandate,
    Item,
    LineItem,
    LineItemRequirement,
    LineItemsConstraint,
    Merchant,
    OpenCheckoutMandate,
    Verdict,
)
from janus.defend.hard_rules import (
    evaluate_account_ref,
    evaluate_checkout,
    evaluate_purchase_summary,
)


def _mandate_and_checkout(quantity_required: int, quantity_offered: int):
    now = int(time.time())
    merchant = Merchant(id="m1", name="M1")
    mandate = OpenCheckoutMandate(
        constraints=[
            LineItemsConstraint(
                items=[LineItemRequirement(id="r1", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=quantity_required)]
            )
        ],
        iat=now,
        exp=now + 3600,
    )
    payload = CheckoutJWTPayload(
        order_id="o1",
        merchant=merchant,
        line_items=[LineItem(id="l1", item=Item(id="sku1", title="Sku 1", price=1000), quantity=quantity_offered)],
        total_price=1000 * quantity_offered,
        currency="USD",
    )
    checkout = CheckoutMandate(checkout_jwt=payload, checkout_hash="h", iat=now, exp=now + 3600)
    return mandate, checkout


class TestEvaluateAccountRef:
    def test_own_session_passes(self):
        result = evaluate_account_ref("user_a", None)
        assert result.verdict == Verdict.PASS

    def test_own_session_explicit_passes(self):
        result = evaluate_account_ref("user_a", "user_a")
        assert result.verdict == Verdict.PASS

    def test_foreign_account_ref_blocked(self):
        """The exact Vault Whisper exploit path."""

        result = evaluate_account_ref("user_a", "user_priya")
        assert result.verdict == Verdict.BLOCK
        assert "user_priya" in result.reasons[0]
        assert "user_a" in result.reasons[0]


class TestEvaluateCheckout:
    def test_compliant_checkout_passes(self):
        mandate, checkout = _mandate_and_checkout(quantity_required=1, quantity_offered=1)
        result = evaluate_checkout(mandate, checkout)
        assert result.verdict == Verdict.PASS

    def test_cart_inflation_blocked(self):
        mandate, checkout = _mandate_and_checkout(quantity_required=1, quantity_offered=2)
        result = evaluate_checkout(mandate, checkout)
        assert result.verdict == Verdict.BLOCK
        assert result.reasons


class TestEvaluatePurchaseSummary:
    def test_correct_summary_passes(self):
        result = evaluate_purchase_summary("Total: $129.99", 12999)
        assert result.verdict == Verdict.PASS

    def test_currency_confusion_blocked(self):
        result = evaluate_purchase_summary("Total: $12.99", 12999)
        assert result.verdict == Verdict.BLOCK
        assert "129.99" in result.reasons[0]
