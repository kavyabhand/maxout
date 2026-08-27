"""Mandate Firewall: the Category D reasoning-layer control, orchestrating:

  1(a) content scrubbing; an ENSEMBLE of the always-on InjectionTextClassifier
      (TF-IDF+LogReg, seeded on the template corpus, live-`.learn()`-able
      round over round) and, if a checkpoint exists, the stronger
      DistilBertInjectionClassifier. Both run when the DistilBERT tier is
      present; the more severe verdict wins. This mirrors the predecessor
      project's fix for a real bug it found: an earlier version only
      updated the TF-IDF-equivalent detector as a fallback when no trained
      classifier was loaded, so a same-session discovery from `.learn()`
      never reached the detector actually protecting the next round. Here
      both tiers always run, so `.learn()` is never a no-op.
  1(b) deterministic hard rules (hard_rules.py): mandate constraint
      satisfaction, account-ref binding, purchase-summary amount
      verification. None read prompt content, so none can be reasoned
      around by any payload however well-crafted.

This is threaded into janus.generate.agentic.shopping_agent.run_shopping_session
as the optional `firewall=` argument, and is what the closed loop
(janus/orchestrate/loop.py) swaps from absent (Round 0) to present and
retrained (Round 1+).
"""

from __future__ import annotations

from dataclasses import dataclass

from janus.common.schemas import CheckoutMandate, OpenCheckoutMandate, Verdict
from janus.defend.hard_rules import evaluate_account_ref, evaluate_checkout, evaluate_purchase_summary
from janus.defend.nlp import DistilBertInjectionClassifier, InjectionTextClassifier, ScrubResult
from janus.generate.agentic.corpus import build_template_corpus


@dataclass
class FirewallEvent:
    stage: str  # "content_scrub" | "account_ref" | "checkout_constraints" | "purchase_summary"
    verdict: Verdict
    reasons: list[str]
    detail: str = ""


_VERDICT_SEVERITY = {Verdict.PASS: 0, Verdict.FLAG: 1, Verdict.BLOCK: 2}


class MandateFirewall:
    def __init__(
        self,
        classifier_checkpoint_dir: str | None = None,
        learned_texts: list[str] | None = None,
    ):
        self.events: list[FirewallEvent] = []

        texts, labels = build_template_corpus()
        if learned_texts:
            texts = texts + list(learned_texts)
            labels = labels + [1] * len(learned_texts)
        self.live_detector = InjectionTextClassifier()
        self.live_detector.fit(texts, labels)

        self.trained_detector: DistilBertInjectionClassifier | None = None
        self.detector_kind = "tfidf_logreg"
        if classifier_checkpoint_dir:
            try:
                self.trained_detector = DistilBertInjectionClassifier(classifier_checkpoint_dir)
                self.detector_kind = "distilbert+tfidf_logreg"
            except FileNotFoundError:
                pass

    def learn(self, texts: list[str]) -> None:
        """Round-over-round retraining hook for the closed loop: fold this
        round's discovered successful payloads in and refit immediately,
        so the very next attempt in the same run already benefits."""

        self.live_detector.learn(texts)

    def scrub_catalog_text(self, text: str) -> str:
        live_result: ScrubResult = self.live_detector.scrub(text)
        result = live_result
        if self.trained_detector is not None:
            trained_result = self.trained_detector.scrub(text)
            if _VERDICT_SEVERITY[trained_result.verdict] >= _VERDICT_SEVERITY[live_result.verdict]:
                result = trained_result

        self.events.append(
            FirewallEvent(stage="content_scrub", verdict=result.verdict, reasons=result.reasons, detail=text[:120])
        )
        return result.cleaned_text

    def check_account_ref(self, session_user_id: str, account_ref: str | None) -> Verdict:
        result = evaluate_account_ref(session_user_id, account_ref)
        self.events.append(FirewallEvent(stage="account_ref", verdict=result.verdict, reasons=result.reasons))
        return result.verdict

    def check_checkout(self, open_mandate: OpenCheckoutMandate, checkout: CheckoutMandate) -> Verdict:
        result = evaluate_checkout(open_mandate, checkout)
        self.events.append(FirewallEvent(stage="checkout_constraints", verdict=result.verdict, reasons=result.reasons))
        return result.verdict

    def check_purchase_summary(self, summary: str, total_price_minor_units: int) -> Verdict:
        result = evaluate_purchase_summary(summary, total_price_minor_units)
        self.events.append(FirewallEvent(stage="purchase_summary", verdict=result.verdict, reasons=result.reasons))
        return result.verdict

    def blocked_events(self) -> list[FirewallEvent]:
        return [e for e in self.events if e.verdict == Verdict.BLOCK]

    def flagged_events(self) -> list[FirewallEvent]:
        return [e for e in self.events if e.verdict == Verdict.FLAG]
