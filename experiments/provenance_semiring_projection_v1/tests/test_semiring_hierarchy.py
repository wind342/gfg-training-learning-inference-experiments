from __future__ import annotations

from experiments.provenance_semiring_projection_v1.src.native_lower_k import evaluate_direct_lower_domains
from experiments.provenance_semiring_projection_v1.src.native_nx import evaluate_native_nx
from experiments.provenance_semiring_projection_v1.src.semiring_homomorphisms import derive_lower_domains_from_nx
from experiments.provenance_semiring_projection_v1.src.workloads import workload_by_id


def test_direct_lower_k_equals_nx_homomorphic_image_on_semiring_features() -> None:
    for workload_id in ("W3", "W4", "W6", "W10", "W12"):
        workload = workload_by_id(workload_id)
        direct = evaluate_direct_lower_domains(workload)
        derived = derive_lower_domains_from_nx(evaluate_native_nx(workload))
        assert direct == derived


def test_coefficient_and_exponent_are_strictly_forgotten_by_lower_maps() -> None:
    w4 = derive_lower_domains_from_nx(evaluate_native_nx(workload_by_id("W4")))
    domains = {domain["domain_id"]: domain["outputs"][0]["annotation"] for domain in w4["domains"]}
    assert domains["bag_naturals"] == 2
    assert domains["boolean"] is True
    assert len(domains["flat_source_support_view"]["variables"]) == 2
    assert len(domains["positive_boolean_lineage"]["terms"]) == 1


def test_flat_support_is_a_task_projection_not_a_semiring_target() -> None:
    workload = workload_by_id("W3")
    direct = evaluate_direct_lower_domains(workload)
    domain_ids = {domain["domain_id"] for domain in direct["domains"]}
    assert "flat_source_support_view" in domain_ids
    assert "why_powerset" not in domain_ids
