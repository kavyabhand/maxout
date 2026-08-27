"""Tests for janus.common.mandate_rules; the maximal-flow line-item
constraint checker that BOTH the ground-truth attack scorer
(janus/generate/agentic/divergence.py) and the actual real-time defense
(janus/defend/hard_rules.py) depend on. This is the single most
safety-critical piece of logic in the repo; every "the hard rule closes
this attack" claim depends on it being correct.
"""

from __future__ import annotations

import time

import pytest

from janus.common.mandate_rules import check_constraints, summary_matches_amount
from janus.common.schemas import (
    AcceptableItem,
    AllowedMerchantsConstraint,
    CheckoutJWTPayload,
    CheckoutMandate,
    Item,
    LineItem,
    LineItemRequirement,
    LineItemsConstraint,
    Merchant,
    OpenCheckoutMandate,
)


def _merchant(id_="merchant_a", name="A"):
    return Merchant(id=id_, name=name, website="https://a.example")


def _open_mandate(constraints):
    now = int(time.time())
    return OpenCheckoutMandate(constraints=constraints, iat=now, exp=now + 3600)


def _checkout(merchant, items: list[tuple[str, str, int, int]]):
    """items: list of (line_id, item_id, quantity, price)"""

    now = int(time.time())
    line_items = [
        LineItem(id=lid, item=Item(id=iid, title=iid, price=price), quantity=qty)
        for lid, iid, qty, price in items
    ]
    payload = CheckoutJWTPayload(
        order_id="order_1",
        merchant=merchant,
        line_items=line_items,
        total_price=sum(qty * price for _, _, qty, price in items),
        currency="USD",
    )
    return CheckoutMandate(checkout_jwt=payload, checkout_hash="hash", iat=now, exp=now + 3600)


class TestMerchantConstraint:
    def test_allowed_merchant_passes(self):
        merchant = _merchant("m1")
        mandate = _open_mandate([AllowedMerchantsConstraint(allowed=[merchant])])
        checkout = _checkout(merchant, [("l1", "sku1", 1, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.merchant_allowed is True
        assert result.reasons == []

    def test_disallowed_merchant_blocked(self):
        allowed = _merchant("m1")
        attacker = _merchant("m2", "Attacker Store")
        mandate = _open_mandate([AllowedMerchantsConstraint(allowed=[allowed])])
        checkout = _checkout(attacker, [("l1", "sku1", 1, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.merchant_allowed is False
        assert any("not in allowed set" in r for r in result.reasons)

    def test_no_merchant_constraint_present_defaults_to_allowed(self):
        """If the mandate never constrains merchants at all, absence of a
        constraint must not silently fail closed OR silently fail open in
        a way that looks like a pass for the wrong reason, merchant_allowed
        should stay True (nothing to violate) with no spurious reason."""

        merchant = _merchant("m1")
        mandate = _open_mandate([])
        checkout = _checkout(merchant, [("l1", "sku1", 1, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.merchant_allowed is True
        assert result.reasons == []


class TestLineItemsConstraint:
    def test_exact_match_passes(self):
        merchant = _merchant()
        req = LineItemRequirement(id="r1", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=1)
        mandate = _open_mandate([LineItemsConstraint(items=[req])])
        checkout = _checkout(merchant, [("l1", "sku1", 1, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.line_items_satisfied is True

    def test_quantity_inflation_blocked(self):
        """The exact scenario Cart Inflation targets: mandate requires
        quantity=1, checkout tries quantity=2 of the same, permitted item."""

        merchant = _merchant()
        req = LineItemRequirement(id="r1", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=1)
        mandate = _open_mandate([LineItemsConstraint(items=[req])])
        checkout = _checkout(merchant, [("l1", "sku1", 2, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.line_items_satisfied is False
        assert any("maximal-flow" in r for r in result.reasons)

    def test_quantity_under_delivery_blocked(self):
        """Under-delivery (quantity=0-effective / item missing) must also
        fail; the flow-equality check requires BOTH totals to match, not
        just checkout <= required."""

        merchant = _merchant()
        req = LineItemRequirement(id="r1", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=2)
        mandate = _open_mandate([LineItemsConstraint(items=[req])])
        checkout = _checkout(merchant, [("l1", "sku1", 1, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.line_items_satisfied is False

    def test_item_outside_acceptable_set_blocked(self):
        """A Confused-Merchant/Mandate-Downgrade-style substitution: the
        checkout swaps in an item ID never listed as acceptable."""

        merchant = _merchant()
        req = LineItemRequirement(id="r1", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=1)
        mandate = _open_mandate([LineItemsConstraint(items=[req])])
        checkout = _checkout(merchant, [("l1", "sku_attacker_swap", 1, 1000)])
        result = check_constraints(mandate, checkout)
        assert result.line_items_satisfied is False

    def test_multiple_acceptable_items_any_one_satisfies(self):
        """Branded Whisper's actual shape: the constraint permits several
        SKUs (any is a legitimate protocol-level choice); the checkout
        picking any one of them must satisfy the HARD constraint, even if
        it's not the "genuinely best" one (that's JANUS's own judgment
        layer, not a protocol violation)."""

        merchant = _merchant()
        req = LineItemRequirement(
            id="r1",
            acceptable_items=[AcceptableItem(id="sku_good", title="Good"), AcceptableItem(id="sku_bad", title="Bad")],
            quantity=1,
        )
        mandate = _open_mandate([LineItemsConstraint(items=[req])])
        checkout = _checkout(merchant, [("l1", "sku_bad", 1, 500)])
        result = check_constraints(mandate, checkout)
        assert result.line_items_satisfied is True

    def test_duplicate_item_ids_across_requirements_not_double_counted(self):
        """Two distinct requirements both accepting the same SKU: the spec's
        own maximal-flow construction should require enough TOTAL quantity
        of that SKU to satisfy both, not let one unit satisfy both at once."""

        merchant = _merchant()
        req1 = LineItemRequirement(id="r1", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=1)
        req2 = LineItemRequirement(id="r2", acceptable_items=[AcceptableItem(id="sku1", title="Sku 1")], quantity=1)
        mandate = _open_mandate([LineItemsConstraint(items=[req1, req2])])

        # Only 1 unit total offered, must fail (need 2).
        checkout_insufficient = _checkout(merchant, [("l1", "sku1", 1, 1000)])
        assert check_constraints(mandate, checkout_insufficient).line_items_satisfied is False

        # 2 units offered, must pass.
        checkout_sufficient = _checkout(merchant, [("l1", "sku1", 2, 1000)])
        assert check_constraints(mandate, checkout_sufficient).line_items_satisfied is True


class TestSummaryMatchesAmount:
    def test_exact_match(self):
        assert summary_matches_amount("Total: $129.99 USD", 12999) is True

    def test_decimal_shift_rejected(self):
        """The actual attack pattern discovered by the Red Team Agent
        (BUILDLOG.md 2026-08-15, Currency/Locale Confusion): a decimal-
        shifted figure that reads as plausible but isn't the real total."""

        assert summary_matches_amount("Total: $12.99 USD", 12999) is False

    def test_within_tolerance_passes(self):
        assert summary_matches_amount("about $130 total", 12999) is True

    def test_no_number_at_all_rejected(self):
        assert summary_matches_amount("Your order is confirmed!", 12999) is False

    def test_comma_thousands_separator_handled(self):
        assert summary_matches_amount("Total: $1,299.99", 129999) is True

    def test_zero_amount_edge_case_does_not_crash(self):
        assert summary_matches_amount("free item", 0) is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
