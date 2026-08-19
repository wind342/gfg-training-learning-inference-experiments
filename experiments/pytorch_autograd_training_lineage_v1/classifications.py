from __future__ import annotations

from typing import Any

from .core_capture import CoreTrainingCollector
from .lineage import TrainingLineageIndex
from .pipeline import TrainingSpec, run_training_step


def analyze_zero_gradient_vs_nonparticipation() -> dict[str, Any]:
    spec = TrainingSpec(workload="zero_gradient_and_unused_sources")
    collector = CoreTrainingCollector()
    run = run_training_step(spec, collector=collector)
    capture = collector.finalize(evidence_context=spec.evidence_context)
    index = TrainingLineageIndex(capture.snapshot, capture.validation)
    zero_ref = "source:parameter:p_zero:before"
    unused_ref = "source:parameter:p_unused:before"
    zero_query = index.forward_lineage(index.source_id_for_ref(zero_ref))
    unused_query = index.forward_lineage(index.source_id_for_ref(unused_ref))
    zero_gradient = run.ordinary_result["gradients"]["parameter:p_zero"]
    unused_gradient = run.ordinary_result["gradients"]["parameter:p_unused"]
    unused_source_id = index.source_id_for_ref(unused_ref)
    dispositions = [
        binding for binding in capture.snapshot.tables.generation_bindings
        if binding["origin_reference"].get("source_information_id") == unused_source_id
        and binding["outcome_reference"]["kind"] == "disposition"
    ]
    zero_values = zero_gradient["value"]
    checks = {
        "p_unused_has_explicit_disposition": len(dispositions) == 1,
        "p_unused_has_no_participation_path": unused_query["path_count"] == 0,
        "p_unused_grad_is_none": unused_gradient is None,
        "p_zero_gradient_support_exists": "step_0:gradient:parameter:p_zero" in zero_query["outcome_keys"],
        "p_zero_has_generation_path": zero_query["path_count"] > 0,
        "p_zero_grad_is_zero_tensor": zero_gradient is not None and all(value == 0.0 for value in zero_values),
    }
    return {
        "checks": checks,
        "p_unused_classification": "DID_NOT_PARTICIPATE",
        "p_unused_forward_query": unused_query,
        "p_zero_classification": "PARTICIPATED_WITH_ZERO_DERIVATIVE",
        "p_zero_forward_query": zero_query,
        "status": (
            "ZERO_GRADIENT_PARTICIPATION_DISTINCTION_SUPPORTED"
            if all(checks.values())
            else "ZERO_GRADIENT_PARTICIPATION_DISTINCTION_NOT_ESTABLISHED"
        ),
    }
