from __future__ import annotations

import hashlib
from typing import Any

from .core_capture import CoreTrainingCollector
from .native_graph import observe_native_autograd_graph
from .pipeline import TrainingSpec, run_training_step


def _capture(spec: TrainingSpec) -> tuple[Any, Any]:
    collector = CoreTrainingCollector()
    run = run_training_step(spec, collector=collector, native_observer=observe_native_autograd_graph)
    return run, collector.finalize(evidence_context=spec.evidence_context)


def build_strict_projection_counterexamples() -> dict[str, Any]:
    identity_a, gamma_identity_a = _capture(TrainingSpec(
        workload="branch_and_merge",
        sample_identity="sample_a",
        evidence_context="environment_a",
    ))
    identity_b, gamma_identity_b = _capture(TrainingSpec(
        workload="branch_and_merge",
        sample_identity="sample_b",
        evidence_context="environment_a",
    ))
    evidence_a, gamma_evidence_a = _capture(TrainingSpec(
        workload="shared_tensor_reuse",
        sample_identity="sample_shared",
        evidence_context="operation_context_a",
    ))
    evidence_b, gamma_evidence_b = _capture(TrainingSpec(
        workload="shared_tensor_reuse",
        sample_identity="sample_shared",
        evidence_context="operation_context_b",
    ))
    stable, gamma_stable = _capture(TrainingSpec(
        workload="checkpoint_external_state",
        sample_identity="checkpoint_sample",
        checkpoint_mode="stable",
    ))
    divergent, gamma_divergent = _capture(TrainingSpec(
        workload="checkpoint_external_state",
        sample_identity="checkpoint_sample",
        checkpoint_mode="divergent",
    ))
    pairs = [
        {
            "gamma_a_snapshot_id": gamma_identity_a.snapshot.snapshot_id,
            "gamma_b_snapshot_id": gamma_identity_b.snapshot.snapshot_id,
            "gamma_different": gamma_identity_a.snapshot.snapshot_id != gamma_identity_b.snapshot.snapshot_id,
            "graph_equal": identity_a.native_observation == identity_b.native_observation,
            "ordinary_output_equal": identity_a.ordinary_bytes == identity_b.ordinary_bytes,
            "pair": "different_sample_identity_equal_value",
        },
        {
            "gamma_a_snapshot_id": gamma_evidence_a.snapshot.snapshot_id,
            "gamma_b_snapshot_id": gamma_evidence_b.snapshot.snapshot_id,
            "gamma_different": gamma_evidence_a.snapshot.snapshot_id != gamma_evidence_b.snapshot.snapshot_id,
            "graph_equal": evidence_a.native_observation == evidence_b.native_observation,
            "ordinary_output_equal": evidence_a.ordinary_bytes == evidence_b.ordinary_bytes,
            "pair": "different_evidence_context_equal_computation",
        },
        {
            "forward_output_equal": stable.ordinary_result["forward_output"] == divergent.ordinary_result["forward_output"],
            "gamma_a_snapshot_id": gamma_stable.snapshot.snapshot_id,
            "gamma_b_snapshot_id": gamma_divergent.snapshot.snapshot_id,
            "gamma_different": gamma_stable.snapshot.snapshot_id != gamma_divergent.snapshot.snapshot_id,
            "gradient_different": stable.ordinary_result["gradients"] != divergent.ordinary_result["gradients"],
            "graph_equal": stable.native_observation == divergent.native_observation,
            "pair": "checkpoint_recomputation_external_state",
        },
    ]
    return {
        "conclusion": "graph equality does not identify complete training-generation facts",
        "counterexample_count": len(pairs),
        "pairs": pairs,
        "status": (
            "PYTORCH_AUTOGRAD_STRICT_PROJECTION_SUPPORTED"
            if all(row["gamma_different"] and row["graph_equal"] for row in pairs)
            else "STRICT_PROJECTION_NOT_ESTABLISHED"
        ),
    }
