from __future__ import annotations

from experiments.order_refund_freeze_inter_fact_relations_v1.src.capture_auditor import (
    CAPTURE_COMPLETE,
    audit_capture,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.generation_fact_collector import (
    collect_atomic_facts,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.orchestrator import (
    run_workflow,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.relation_sidecar_collector import (
    collect_relation_sidecar,
)


def test_facts_keep_exactly_five_coordinates_and_sidecar_is_external() -> None:
    run = run_workflow(
        "CONCURRENT_REFUND_WINS", repeat_index=3, capture_enabled=True
    )
    atomic = collect_atomic_facts(run)
    sidecar = collect_relation_sidecar(run, atomic)
    assert atomic["coordinate_names"] == ["u", "tau", "omega_bar", "z", "rho"]
    assert atomic["sixth_coordinate_present"] is False
    assert all(
        set(row["coordinates"]) == {"u", "tau", "omega_bar", "z", "rho"}
        for row in atomic["facts"]
    )
    assert sidecar["program_order_exactness"]["status"] == "PASS"
    assert sidecar["relation_type_counts"]["reads_from"] == 2
    assert sidecar["relation_type_counts"]["conflicts_with"] == 1


def test_capture_complete_is_machine_audited() -> None:
    run = run_workflow(
        "CONCURRENT_FREEZE_WINS", repeat_index=3, capture_enabled=True
    )
    atomic = collect_atomic_facts(run)
    sidecar = collect_relation_sidecar(run, atomic)
    audit = audit_capture(run, atomic, sidecar)
    assert audit["status"] == CAPTURE_COMPLETE
    assert audit["concurrency_inference_allowed"] is True
    assert audit["global_scheduler_completeness_machine_proved"] is False
    assert audit["concurrency_scope"] == "CONTROLLED_CAPTURE_SCOPE_ONLY"
