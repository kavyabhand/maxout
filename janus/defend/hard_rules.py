"""Layer 1(c), deterministic hard-constraint backstop.

The point of this layer: it cannot be reasoned around by any prompt,
because it never looks at prompt content. It only looks at structured
mandate/tool-call fields. Two checks, both already proven to catch the
naive-baseline attack corpus (BUILDLOG.md 2026-08-15):

1. Mandate constraint satisfaction (merchant allow-list + line-items
   maximal-flow match) via janus.common.mandate_rules, catches any
   Mandate-Downgrade / Cart-Inflation style violation of the mandate itself.
2. Account binding; the credentials tool call's account_ref, if present,
   MUST equal the authenticated session's user_id. This is exactly
   CredentialsProviderAgent(strict_account_binding=True); this module's job
   is to make that binding explicit and independently checkable, not to
   reimplement it.

Neither check requires an LLM call, so this layer runs in the low-single-
digit-millisecond range (see BUILDLOG.md's latency measurements) and is
the piece of the firewall the "no frontier LLM in the hot path" latency
argument rests on most heavily.
"""

from __future__ import annotations

from dataclasses import dataclass

from janus.common.mandate_rules import check_constraints, summary_matches_amount
from janus.common.schemas import CheckoutMandate, OpenCheckoutMandate, Verdict


@dataclass
class HardRuleResult:
    verdict: Verdict
    reasons: list[str]


def evaluate_checkout(open_mandate: OpenCheckoutMandate, checkout: CheckoutMandate) -> HardRuleResult:
    check = check_constraints(open_mandate, checkout)
    if check.reasons:
        return HardRuleResult(verdict=Verdict.BLOCK, reasons=check.reasons)
    return HardRuleResult(verdict=Verdict.PASS, reasons=[])


def evaluate_account_ref(session_user_id: str, account_ref: str | None) -> HardRuleResult:
    if account_ref is not None and account_ref != session_user_id:
        return HardRuleResult(
            verdict=Verdict.BLOCK,
            reasons=[f"account_ref={account_ref!r} does not match session user_id={session_user_id!r}"],
        )
    return HardRuleResult(verdict=Verdict.PASS, reasons=[])


def evaluate_purchase_summary(summary: str, total_price_minor_units: int) -> HardRuleResult:
    """Closes Currency/Locale Confusion the same way evaluate_checkout()
    closes Cart Inflation: deterministically, at the point the agent tries
    to state the amount, rather than hoping the content-scrubbing layer
    caught whatever manipulated it. Rejects a confirmation that doesn't
    contain the real total in any recognizable form."""

    if not summary_matches_amount(summary, total_price_minor_units):
        expected = total_price_minor_units / 100
        return HardRuleResult(
            verdict=Verdict.BLOCK,
            reasons=[f"purchase summary {summary!r} does not state the actual total (${expected:.2f})"],
        )
    return HardRuleResult(verdict=Verdict.PASS, reasons=[])
