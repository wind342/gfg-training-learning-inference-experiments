from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from generation_relation_core.snapshots import SnapshotValidation

from experiments.provenance_semiring_projection_v1.src.candidate_nx import CandidateProjectionError, project_snapshot_to_nx
from experiments.provenance_semiring_projection_v1.src.core_capture import core_snapshot_from_events
from experiments.provenance_semiring_projection_v1.src.native_nx import evaluate_native_nx
from experiments.provenance_semiring_projection_v1.src.nx_polynomial import NXPolynomial
from experiments.provenance_semiring_projection_v1.src.ordinary_execution import execute_ordinary
from experiments.provenance_semiring_projection_v1.src.profile_runtime import load_profile
from experiments.provenance_semiring_projection_v1.src.structural import variable_for_source
from experiments.provenance_semiring_projection_v1.src.workloads import workload_by_id


REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_BASE = "7320fe8a2d690fc87da77d0739b432ea1812d63b"


def _require_frozen_history() -> None:
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{FROZEN_BASE}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not available:
        pytest.skip("GF-P02 frozen source-history object is not part of the companion clone")


def test_native_selection_excludes_disposed_source_variable_from_outputs() -> None:
    result = evaluate_native_nx(workload_by_id("W1"))
    all_variables = {item["variable"] for item in result["source_variables"]}
    output_variables = {factor["variable"] for output in result["outputs"] for term in output["polynomial"]["terms"] for factor in term["monomial"]}
    assert len(all_variables) == 3
    assert len(output_variables) == 2
    assert output_variables < all_variables


def test_native_join_has_one_product_with_two_factors() -> None:
    output = evaluate_native_nx(workload_by_id("W2"))["outputs"][0]
    assert len(output["polynomial"]["terms"]) == 1
    assert len(output["polynomial"]["terms"][0]["monomial"]) == 2


def test_native_union_adds_alternative_source_terms() -> None:
    output = evaluate_native_nx(workload_by_id("W8"))["outputs"][0]
    assert len(output["polynomial"]["terms"]) == 2


def test_source_variable_is_full_sha256_of_complete_identity() -> None:
    identity = "stable:source:identity"
    assert variable_for_source(identity) == "x_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def test_zero_polynomial_is_explicit_and_round_trips() -> None:
    document = {"schema_version": "nx-polynomial-v1", "terms": []}
    assert NXPolynomial.from_document(document).to_document() == document


def test_nonpositive_serialized_coefficient_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        NXPolynomial.from_document({"schema_version": "nx-polynomial-v1", "terms": [{"coefficient": 0, "monomial": []}]})


def test_candidate_cycle_detection_uses_generated_origin_graph() -> None:
    _, _, snapshot, validation = core_snapshot_from_events(workload_by_id("W7"))
    terminal = next(row for row in snapshot.tables.perceptual_support_records if row["support_payload"]["terminal"])
    snapshot.tables.generated_origins[0]["origin_payload"]["prior_support_id"] = terminal["support_id"]
    synthetic_validation = SnapshotValidation(snapshot_id=snapshot.snapshot_id, relation_evidence=validation.relation_evidence)
    with pytest.raises(CandidateProjectionError, match="cycle"):
        project_snapshot_to_nx(snapshot, synthetic_validation)


def test_positive_ra_profile_keeps_negation_out_of_scope() -> None:
    profile = load_profile("positive_relational_algebra_profile_v1.json")
    assert "negation" in profile["excluded_operators"]
    assert "natural/equi-join" in profile["included_operators"]


def test_protected_core_protocol_compat_and_core_tests_match_frozen_base() -> None:
    _require_frozen_history()
    completed = subprocess.run(
        ["git", "diff", "--quiet", FROZEN_BASE, "--", "src/generation_relation_core", "protocol", "compat", "tests/core"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_existing_database_experiment_tree_matches_frozen_base() -> None:
    _require_frozen_history()
    current = subprocess.check_output(["git", "rev-parse", "HEAD:experiments/database_lineage"], cwd=REPO_ROOT, text=True).strip()
    frozen = subprocess.check_output(["git", "rev-parse", f"{FROZEN_BASE}:experiments/database_lineage"], cwd=REPO_ROOT, text=True).strip()
    assert current == frozen == "64b5365d9a828a645c99b536254a07a2519f0cc0"


def test_frozen_ordinary_executor_w2_result_values_are_exact() -> None:
    ordinary, _measurements = execute_ordinary(workload_by_id("W2"))
    rows = json.loads(ordinary)["rows"]
    assert rows == [{"key": 1, "left_value": "L1", "right_value": "S1", "s_key": 1}]
