from __future__ import annotations

import math
from typing import Any

from .core_capture import CoreTrainingCollector
from .independent_reference import ReceiptLineageReference
from .lineage import TrainingLineageIndex
from .native_graph import observe_native_autograd_graph
from .pipeline import TrainingSpec, run_training_step


def _capture(mode: str):
    spec = TrainingSpec(
        workload="checkpoint_external_state",
        sample_identity="checkpoint_sample",
        checkpoint_mode=mode,
    )
    collector = CoreTrainingCollector()
    run = run_training_step(spec, collector=collector, native_observer=observe_native_autograd_graph)
    capture = collector.finalize(evidence_context=spec.evidence_context)
    return run, capture


def _all_finite(value: Any) -> bool:
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return isinstance(value, (int, float)) and math.isfinite(value)


def _snapshot_payload(capture: Any) -> dict[str, Any]:
    return {
        "record": capture.snapshot.record,
        "tables": {
            key: value for key, value in vars(capture.snapshot.tables).items()
        },
    }


def analyze_checkpoint_divergence() -> dict[str, Any]:
    no_checkpoint, no_capture = _capture("no_checkpoint")
    stable, stable_capture = _capture("stable")
    divergent, divergent_capture = _capture("divergent")
    stable_gradient = stable.ordinary_result["gradients"]["parameter:p"]["value"]
    no_gradient = no_checkpoint.ordinary_result["gradients"]["parameter:p"]["value"]
    divergent_gradient = divergent.ordinary_result["gradients"]["parameter:p"]["value"]
    target_key = "step_0:parameter:p:after"
    divergent_index = TrainingLineageIndex(divergent_capture.snapshot, divergent_capture.validation)
    target_id = divergent_index.support_id_for_key(target_key)
    reverse = divergent_index.reverse_lineage(target_id)
    scale_ref = "source:external:scale:recomputation"
    scale_source_id = divergent_index.source_id_for_ref(scale_ref)
    forward = divergent_index.forward_lineage(scale_source_id)
    reference = ReceiptLineageReference(divergent_capture.execution_receipts)
    reverse_reference = reference.reverse_paths(target_key)
    forward_reference = reference.forward_paths(scale_ref)
    recompute_occurrences = [
        row for row in divergent_capture.snapshot.tables.generation_occurrences
        if row["occurrence_stage"] == "backward_recomputation"
    ]
    original_occurrences = [
        row for row in divergent_capture.snapshot.tables.generation_occurrences
        if row["occurrence_stage"] == "forward_original"
    ]
    parameter_after = divergent.ordinary_result["parameter_after"]["p"]["value"]
    optimizer_after_key = "step_0:optimizer_state:after"
    required_forward_targets = {
        "step_0:gradient:parameter:p",
        optimizer_after_key,
        target_key,
    }
    checks = {
        "backward_completed_without_exception": divergent.ordinary_result["exception"] is None,
        "default_determinism_check_did_not_raise": divergent.ordinary_result["exception"] is None,
        "divergent_gradient_differs": divergent_gradient != stable_gradient,
        "divergent_gradient_finite": _all_finite(divergent_gradient),
        "divergent_parameter_update_differs": divergent.ordinary_result["parameter_after"] != stable.ordinary_result["parameter_after"],
        "forward_output_equal_before_backward": stable.ordinary_result["forward_output"] == divergent.ordinary_result["forward_output"],
        "native_graph_stable_divergent_exact": stable.native_observation == divergent.native_observation,
        "no_checkpoint_stable_gradient_exact": no_gradient == stable_gradient,
        "no_checkpoint_stable_parameter_update_exact": no_checkpoint.ordinary_result["parameter_after"] == stable.ordinary_result["parameter_after"],
        "original_and_recomputation_occurrences_distinct": bool(original_occurrences and recompute_occurrences) and not ({row["generation_occurrence_id"] for row in original_occurrences} & {row["generation_occurrence_id"] for row in recompute_occurrences}),
        "recomputation_actually_executed": bool(recompute_occurrences),
        "reverse_query_exact_against_receipts": reverse["paths"] == reverse_reference,
        "reverse_trace_reaches_divergent_scale": scale_ref in reverse["source_keys"],
        "forward_query_exact_against_receipts": forward["paths"] == forward_reference,
        "forward_trace_reaches_all_required_outcomes": required_forward_targets <= set(forward["outcome_keys"]),
    }
    return {
        "checks": checks,
        "divergent_gradient": divergent_gradient,
        "divergent_parameter_after": parameter_after,
        "forward_trace": forward,
        "graph_sha256": divergent.native_observation["canonical_graph_sha256"],
        "no_checkpoint_gradient": no_gradient,
        "recomputation_occurrence_ids": sorted(row["generation_occurrence_id"] for row in recompute_occurrences),
        "reverse_trace": reverse,
        "stable_gradient": stable_gradient,
        "validated_snapshots": {
            "divergent": _snapshot_payload(divergent_capture),
            "no_checkpoint": _snapshot_payload(no_capture),
            "stable": _snapshot_payload(stable_capture),
        },
        "status": (
            "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED"
            if all(checks.values())
            else "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_NOT_ESTABLISHED"
        ),
    }
