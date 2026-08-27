"""
AP2 mandate data model: deliberately schema-accurate, not a paraphrase.

Modeled directly against github.com/google-agentic-commerce/AP2
(code/sdk/schemas/ap2/*.json, code/sdk/schemas/ucp/types/*.json,
docs/ap2/checkout_mandate.md), pulled 2026-08-15. Field names, required-ness,
and nesting match the real JSON Schema, not the simplified "Intent/Cart/
Payment Mandate" prose used in earlier public write-ups.

Three mandates, one lineage:

    Open Checkout Mandate  (vct=mandate.checkout.open.1)
        issued by the user's trusted surface. A pre-authorization: a set of
        `constraints` (allowed merchants, required line items with
        acceptable substitute IDs + quantity) plus a `cnf` key-binding claim.
        This is what the user actually agreed to.

    Checkout Mandate       (vct=mandate.checkout.1)
        issued by the Shopping Agent, wraps a MERCHANT-SIGNED `checkout_jwt`
        (the concrete cart: order_id, merchant, line_items, total_price)
        plus `checkout_hash` = hash(checkout_jwt). This is what actually
        gets purchased.

    Payment Mandate        (vct=mandate.payment.1)
        `transaction_id` = hash(checkout_jwt), cryptographically binds the
        payment to one specific checkout_jwt byte-for-byte. Says nothing
        about whether that checkout_jwt's CONTENT still reflects the
        Open Checkout Mandate's constraints. That gap; hash-binding proves
        WHICH checkout was paid for, never WHETHER it was the right one, is the reasoning-layer surface JANUS's Category D pillar is built to catch.

We model the JWTs as plain nested objects (unsigned) rather than actually
minting SD-JWTs; the sandbox's job is to exercise the CONTENT-level
reasoning gap, not to reimplement SD-JWT cryptography. Where it matters
(Mandate Firewall's hard-constraint layer), we implement the spec's
maximal-flow line-item matching for real, because getting that wrong is
itself part of the vulnerability surface, see mandate_firewall/hard_rules.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared UCP/AP2 primitive types (ap2/types/*.json, ucp/types/*.json)
# ---------------------------------------------------------------------------

class Merchant(BaseModel):
    """ap2/types/merchant.json, self-reported, NOT cryptographically bound
    to a domain/cert. Matching against `allowed_merchants` is by these plain
    fields. This self-assertion is the structural opening Confused Merchant
    Delegation attacks exploit."""

    id: str
    name: str
    website: str | None = None


class Amount(BaseModel):
    """ap2/types/amount.json, integer minor units per ISO-4217, e.g.
    amount=27999, currency="USD" means $279.99. Using integer minor units
    (not float) is the spec's own defense against float-rounding tricks: Currency/Locale Confusion attacks target the *display/reasoning* layer
    upstream of this field, not the field's arithmetic."""

    amount: int
    currency: str = Field(min_length=3, max_length=3)


class Item(BaseModel):
    """ap2/types/item.json / ucp item.json, product details for a line item."""

    id: str
    title: str
    price: int = Field(ge=0)
    image_url: str | None = None


class LineItem(BaseModel):
    """ucp/types/line_item.json; a concrete line item as it appears inside
    a signed checkout_jwt (as opposed to the abstract `line_item_requirements`
    used inside an Open Checkout Mandate's constraints)."""

    id: str
    item: Item
    quantity: int = Field(ge=1)
    parent_id: str | None = None


class PaymentInstrument(BaseModel):
    id: str
    type: str
    description: str | None = None


class PISP(BaseModel):
    """Payment Initiation Service Provider, ap2/types/pisp.json."""

    legal_name: str
    brand_name: str
    domain_name: str


# ---------------------------------------------------------------------------
# Open Checkout Mandate constraints (docs/ap2/checkout_mandate.md, $defs)
# ---------------------------------------------------------------------------

class AllowedMerchantsConstraint(BaseModel):
    type: Literal["checkout.allowed_merchants"] = "checkout.allowed_merchants"
    allowed: list[Merchant]


class AcceptableItem(BaseModel):
    """A permitted substitute for a line-item requirement. Matching is by
    `id` only (per spec's maximal-flow evaluation algorithm), title is
    display-only and MUST NOT be trusted for matching. An agent that matches
    on title/semantic-similarity instead of id is exactly the reasoning-layer
    bug Cart Inflation / Quantity Drift attacks exploit."""

    id: str
    title: str


class LineItemRequirement(BaseModel):
    id: str
    acceptable_items: list[AcceptableItem]
    quantity: int = Field(gt=0)


class LineItemsConstraint(BaseModel):
    type: Literal["checkout.line_items"] = "checkout.line_items"
    items: list[LineItemRequirement]


Constraint = AllowedMerchantsConstraint | LineItemsConstraint


class KeyBinding(BaseModel):
    """`cnf` claim, RFC 7800 §3.1, simplified to a JWK-shaped dict; we don't
    implement real key-binding crypto in the sandbox."""

    jwk: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# The three mandates
# ---------------------------------------------------------------------------

class OpenCheckoutMandate(BaseModel):
    """vct=mandate.checkout.open.1, the user's actual, ground-truth intent.
    Everything downstream (Checkout Mandate, Payment Mandate) is judged
    against THIS. In the closed loop, this is the object the Red Team Agent
    never gets to touch directly; it can only try to make the Shopping
    Agent produce a Checkout Mandate that diverges from it."""

    vct: Literal["mandate.checkout.open.1"] = "mandate.checkout.open.1"
    constraints: list[Constraint]
    cnf: KeyBinding = Field(default_factory=KeyBinding)
    iat: int
    exp: int


class CheckoutJWTPayload(BaseModel):
    """The payload of the merchant-signed `checkout_jwt`. This is the
    object whose hash gets committed into both the Checkout Mandate
    (checkout_hash) and the Payment Mandate (transaction_id). It is produced
    by the SHOPPING AGENT'S REASONING (product search/selection/ranking)
    and then countersigned by the merchant; the merchant attests "yes,
    this is a real cart I'll fulfill at this price," not "yes, this
    reflects what the user actually wanted." That distinction is load-
    bearing: merchant signature = execution integrity, not reasoning
    integrity.
    """

    order_id: str
    merchant: Merchant
    line_items: list[LineItem]
    total_price: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    shipping_policy: str | None = None
    return_policy: str | None = None
    # UCP's real Checkout schema is additionalProperties:true; we deliberately
    # keep an open `extra` bag to let the sandbox reproduce that looseness
    # (Cart Inflation attacks can smuggle fields here that a naive downstream
    # consumer might act on without schema validation catching them).
    extra: dict = Field(default_factory=dict)


class CheckoutMandate(BaseModel):
    """vct=mandate.checkout.1, what actually gets purchased."""

    vct: Literal["mandate.checkout.1"] = "mandate.checkout.1"
    checkout_jwt: CheckoutJWTPayload
    checkout_hash: str
    iat: int
    exp: int


class PaymentMandate(BaseModel):
    """vct=mandate.payment.1."""

    vct: Literal["mandate.payment.1"] = "mandate.payment.1"
    transaction_id: str
    payee: Merchant
    payment_amount: Amount
    payment_instrument: PaymentInstrument
    pisp: PISP | None = None
    execution_date: str | None = None
    risk_data: dict = Field(default_factory=dict)
    iat: int
    exp: int


# ---------------------------------------------------------------------------
# JANUS-specific wrapper: a completed transaction attempt, carrying the
# ground-truth mandate alongside whatever the Shopping Agent produced, so
# divergence can be scored. Not part of the AP2 spec; this is our harness.
# ---------------------------------------------------------------------------

class MandateChain(BaseModel):
    """The full artifact produced by one run through the sandbox: the
    user's true intent, what the Shopping Agent actually did, and, filled
    in later by the Mandate Firewall / evaluation loop, whether it diverged."""

    open_checkout_mandate: OpenCheckoutMandate
    checkout_mandate: CheckoutMandate
    payment_mandate: PaymentMandate | None = None

    # Populated by janus.generate.agentic.red_team when this chain resulted from an
    # attack attempt (None for clean/legitimate runs).
    attack_technique: str | None = None
    injected_span: str | None = None


class Verdict(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"
