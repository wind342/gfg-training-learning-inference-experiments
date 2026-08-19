from __future__ import annotations

from pathlib import Path

from experiments.w3c_prov_projection_v1.src.negative_controls import run_negative_controls
from experiments.w3c_prov_projection_v1.src.official_tests import run_official_tests
from experiments.w3c_prov_projection_v1.src.science_runs import output_modes, run_full, strict_projection_counterexamples
from experiments.w3c_prov_projection_v1.src.validation import exact_comparison, relation_multiplicity, validate_profile_documents


ROOT = Path(__file__).resolve().parents[1]


def test_p1_exact_derivability_and_constraints() -> None:
    run = run_full()
    comparison = exact_comparison(run.candidate_from_provn, run.reference_from_provo, len(run.snapshot.tables.generation_bindings))
    constraints = validate_profile_documents(run.candidate_records, run.candidate_provn, run.native_ttl)
    multiplicity = relation_multiplicity(run.candidate_records, len(run.snapshot.tables.generation_bindings))
    assert comparison["status"] == "SUPPORTED"
    assert constraints["status"] == "SUPPORTED"
    assert multiplicity["status"] == "SUPPORTED"
    assert comparison["blocking_metrics"] == []


def test_p2_four_full_valid_counterexample_groups() -> None:
    counterexamples, reverse = strict_projection_counterexamples()
    assert counterexamples["status"] == "SUPPORTED"
    assert counterexamples["requested_group_count"] == 4
    assert counterexamples["valid_group_count"] == 4
    assert all(group["snapshots_differ"] for group in counterexamples["groups"])
    assert all(group["provn_bytes_equal"] and group["provo_normalized_equal"] for group in counterexamples["groups"])
    assert reverse["status"] == "SUPPORTED"
    assert reverse["same_prov_has_multiple_valid_snapshots"]


def test_five_output_modes_are_byte_identical() -> None:
    result = output_modes()
    assert result["status"] == "SUPPORTED"
    assert result["mode_count"] == 5
    assert result["all_bytes_equal"]
    assert result["all_metadata_equal"]
    assert result["forbidden_output_token_count"] == 0


def test_all_32_negative_controls_are_detected_and_honestly_classified() -> None:
    result = run_negative_controls(ROOT / "src")
    assert result["status"] == "SUPPORTED"
    assert result["negative_control_count"] == result["detected_count"] == 32
    assert result["classification_counts"] == {
        "END_TO_END": 2,
        "ISOLATION": 4,
        "VALIDATOR_UNIT": 21,
        "OFFICIAL_CONSTRAINT": 5,
    }


def test_frozen_official_applicable_suite() -> None:
    result = run_official_tests(ROOT / "runtime" / "official_tests")
    assert result["status"] == "SUPPORTED"
    assert result["official_test_total"] == 291
    assert result["historical_implementation_report_total"] == 280
    assert result["applicable_test_count"] == result["passed_applicable_count"] == 53
    assert result["failed_applicable_count"] == 0
