"""End-to-end offline test of the scripted (no-credentials) agentic path:
the naive scripted shopping policy is susceptible to each of the four
Category D techniques' injected directives, and the Mandate Firewall's
deterministic hard rules close three of them outright (the fourth,
branded_whisper, has no structural ground truth to check against by
design, see TAXONOMY-equivalent notes in attacks.yaml D14/D16).

conftest.py forces JANUS_LLM_BACKEND=scripted for the whole suite, so
run_shopping_session here exercises _run_scripted_session with zero
network calls.
"""

from __future__ import annotations

from janus.defend.firewall import MandateFirewall
from janus.generate.agentic.catalog import seed_catalog
from janus.generate.agentic.credentials_provider import CredentialsProviderAgent, seed_accounts
from janus.generate.agentic.divergence import score
from janus.generate.agentic.red_team import TECHNIQUES, generate_payload
from janus.generate.agentic.scenarios import trail_shoe_scenario
from janus.generate.agentic.shopping_agent import run_shopping_session


def _run(technique_name: str, firewall=None):
    technique = TECHNIQUES[technique_name]
    payload, _ = generate_payload(technique, [])
    catalog = seed_catalog()
    catalog.poison(technique.target_item_id, payload)
    scenario = technique.build_scenario(catalog)
    creds = CredentialsProviderAgent(seed_accounts(), strict_account_binding=False)

    result = run_shopping_session(
        session_user_id=technique.session_user_id,
        user_message=scenario.user_message,
        open_mandate=scenario.open_mandate,
        catalog=catalog,
        credentials_provider=creds,
        firewall=firewall,
    )
    return score(scenario, result)


class TestScriptedPolicyVulnerableWithoutFirewall:
    def test_vault_whisper_leaks_without_firewall(self):
        report = _run("vault_whisper")
        assert report.vault_leak is not None
        assert report.attack_succeeded is True

    def test_cart_inflation_violates_without_firewall(self):
        report = _run("cart_inflation")
        assert report.hard_constraint_violated is True
        assert report.attack_succeeded is True

    def test_currency_confusion_mismatches_without_firewall(self):
        report = _run("currency_locale_confusion")
        assert report.summary_matches_amount is False
        assert report.attack_succeeded is True


class TestFirewallClosesDeterministicTechniques:
    def test_vault_whisper_closed_by_firewall(self):
        firewall = MandateFirewall()
        report = _run("vault_whisper", firewall=firewall)
        assert report.vault_leak is None

    def test_cart_inflation_closed_by_firewall(self):
        firewall = MandateFirewall()
        report = _run("cart_inflation", firewall=firewall)
        assert report.hard_constraint_violated is False

    def test_currency_confusion_closed_by_firewall(self):
        firewall = MandateFirewall()
        report = _run("currency_locale_confusion", firewall=firewall)
        assert report.summary_matches_amount is not False
