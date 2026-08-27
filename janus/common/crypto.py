"""Minimal hash-binding helpers for the sandbox's mandate chain.

We deliberately do NOT implement real SD-JWT signing/verification; the
sandbox's purpose is to exercise the CONTENT-level reasoning gap between
what a checkout_jwt says and what the user's Open Checkout Mandate actually
authorized, not to reimplement JWT cryptography. `checkout_hash` and
`transaction_id` are computed exactly as the spec describes (base64url
sha-256 of the checkout_jwt payload bytes) so the binding is real; only the
outer signature envelope is skipped.
"""

import base64
import hashlib
import json

from janus.common.schemas import CheckoutJWTPayload


def checkout_jwt_hash(payload: CheckoutJWTPayload) -> str:
    """base64url(sha256(checkout_jwt payload)), matches checkout_hash /
    transaction_id semantics in the real spec (docs/ap2/checkout_mandate.md)."""

    canonical = json.dumps(
        payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
