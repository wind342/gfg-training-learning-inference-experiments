from __future__ import annotations

from generation_relation_core.snapshots import SnapshotValidation

from experiments.provenance_semiring_projection_v1.src.candidate_nx import project_snapshot_to_nx
from experiments.provenance_semiring_projection_v1.src.core_capture import core_snapshot_from_events
from experiments.provenance_semiring_projection_v1.src.workloads import workload_by_id


def test_collector_is_write_only_and_snapshot_validates() -> None:
    ordinary, measurements, snapshot, validation = core_snapshot_from_events(workload_by_id("W2"))
    assert ordinary.endswith(b"\n")
    assert measurements["occurrence_count"] >= 1
    assert isinstance(validation, SnapshotValidation)
    assert validation.snapshot_id == snapshot.snapshot_id


def test_candidate_recurses_through_generated_origin() -> None:
    _, _, snapshot, validation = core_snapshot_from_events(workload_by_id("W7"))
    assert snapshot.tables.generated_origins
    result = project_snapshot_to_nx(snapshot, validation)
    outputs = result["outputs"]
    assert isinstance(outputs, list) and len(outputs) == 1
    assert len(outputs[0]["polynomial"]["terms"]) == 2


def test_candidate_records_join_product_and_self_join_exponent() -> None:
    _, _, snapshot, validation = core_snapshot_from_events(workload_by_id("W6"))
    result = project_snapshot_to_nx(snapshot, validation)
    term = result["outputs"][0]["polynomial"]["terms"][0]
    assert term["coefficient"] == 1
    assert term["monomial"][0]["exponent"] == 2


def test_dispositions_are_valid_closure_but_not_polynomial_factors() -> None:
    _, _, snapshot, validation = core_snapshot_from_events(workload_by_id("W1"))
    assert snapshot.tables.explicit_dispositions
    result = project_snapshot_to_nx(snapshot, validation)
    variables = {item["variable"] for item in result["source_variables"]}
    polynomial_variables = {
        factor["variable"]
        for output in result["outputs"]
        for term in output["polynomial"]["terms"]
        for factor in term["monomial"]
    }
    assert polynomial_variables < variables
