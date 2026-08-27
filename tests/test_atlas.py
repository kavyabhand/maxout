"""Tests for janus.identify.atlas, coverage must be DERIVED from
attacks.yaml's own status/simulated_by/detected_by fields, never
independently asserted, so the UI's "N simulated" claim can't drift out
of sync with what the pipelines actually do.
"""

from __future__ import annotations

from janus.identify.atlas import AttackAtlas, load_attacks


class TestLoadAttacks:
    def test_loads_all_seventeen(self):
        attacks = load_attacks()
        assert len(attacks) == 17

    def test_every_attack_has_a_valid_status(self):
        for a in load_attacks():
            assert a.status in ("simulated", "modeled", "taxonomy_only")

    def test_simulated_attacks_declare_at_least_one_simulator_and_detector(self):
        for a in load_attacks():
            if a.status == "simulated":
                assert len(a.simulated_by) > 0, f"{a.id} marked simulated but has no simulated_by"
                assert len(a.detected_by) > 0, f"{a.id} marked simulated but has no detected_by"


class TestCoverageSummary:
    def test_status_counts_sum_to_total(self):
        atlas = AttackAtlas()
        summary = atlas.coverage_summary()
        assert sum(summary["by_status"].values()) == summary["total_attacks"]

    def test_category_counts_sum_to_total(self):
        atlas = AttackAtlas()
        summary = atlas.coverage_summary()
        assert sum(c["total"] for c in summary["by_category"].values()) == summary["total_attacks"]

    def test_four_categories_present(self):
        atlas = AttackAtlas()
        summary = atlas.coverage_summary()
        assert set(summary["by_category"].keys()) == {"A", "B", "C", "D"}


class TestForceGraph:
    def test_every_attack_node_present(self):
        atlas = AttackAtlas()
        graph = atlas.to_force_graph()
        node_ids = {n["id"] for n in graph["nodes"]}
        for a in atlas.attacks:
            assert a.id in node_ids

    def test_edges_reference_existing_nodes(self):
        atlas = AttackAtlas()
        graph = atlas.to_force_graph()
        node_ids = {n["id"] for n in graph["nodes"]}
        for e in graph["edges"]:
            assert e["source"] in node_ids
            assert e["target"] in node_ids


class TestReportedCoverageCount:
    """The walkthrough and the README both state, in prose, how many atlas
    entries are not simulated end to end. That sentence was hardcoded and
    went stale: it read "Twelve" from a run in which five vectors were
    simulated, and stayed there after three more were built, understating
    the project's own coverage in the documents a reader consults for
    exactly that figure. It is now derived from the coverage summary, and
    this test holds the derivation to the atlas.
    """

    def test_not_simulated_word_matches_atlas(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from build_report import not_simulated_word

        atlas = AttackAtlas()
        summary = atlas.coverage_summary()
        expected = summary["total_attacks"] - summary["by_status"]["simulated"]

        words = {
            9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
            14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
        }
        assert not_simulated_word(summary) == words.get(expected, str(expected))

    def test_absent_coverage_does_not_assert_a_number(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from build_report import not_simulated_word

        assert not_simulated_word(None) == "Several"
