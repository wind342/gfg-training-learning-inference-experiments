from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from .candidate_projection import project_core_to_autograd_graph
from .core_capture import CoreCaptureResult, CoreTrainingCollector
from .native_graph import observe_native_autograd_graph
from .pipeline import TrainingRun, TrainingSpec, run_training_step
from .projection_analysis import compare_canonical_graphs


@dataclass(frozen=True)
class CaptureModeSuite:
    output_only: TrainingRun
    native_autograd_only: TrainingRun
    core_only: TrainingRun
    core_only_capture: CoreCaptureResult
    core_only_candidate: dict[str, Any]
    core_and_native: TrainingRun
    core_and_native_capture: CoreCaptureResult
    core_and_native_candidate: dict[str, Any]
    report: dict[str, Any]


def run_four_capture_modes(
    spec: TrainingSpec,
    profile: dict[str, Any],
    crosswalk: dict[str, Any],
) -> CaptureModeSuite:
    output_only = run_training_step(spec, collector=None, native_observer=None)
    native_only = run_training_step(
        spec,
        collector=None,
        native_observer=observe_native_autograd_graph,
    )
    core_collector = CoreTrainingCollector()
    core_only = run_training_step(spec, collector=core_collector, native_observer=None)
    core_capture = core_collector.finalize(evidence_context=spec.evidence_context)
    core_candidate = project_core_to_autograd_graph(
        core_capture.snapshot,
        core_capture.validation,
        profile,
        crosswalk,
    )
    dual_collector = CoreTrainingCollector()
    dual = run_training_step(
        spec,
        collector=dual_collector,
        native_observer=observe_native_autograd_graph,
    )
    dual_capture = dual_collector.finalize(evidence_context=spec.evidence_context)
    dual_candidate = project_core_to_autograd_graph(
        dual_capture.snapshot,
        dual_capture.validation,
        profile,
        crosswalk,
    )
    ordinary = {
        "output_only": output_only.ordinary_bytes,
        "native_autograd_only": native_only.ordinary_bytes,
        "core_only": core_only.ordinary_bytes,
        "core_and_native": dual.ordinary_bytes,
    }
    ordinary_hashes = {
        mode: hashlib.sha256(value).hexdigest() for mode, value in ordinary.items()
    }
    projection = compare_canonical_graphs(dual.native_observation, dual_candidate)
    report = {
        "all_ordinary_bytes_exact": len(set(ordinary.values())) == 1,
        "all_ordinary_sha256": ordinary_hashes,
        "candidate_core_dual_exact": core_candidate == dual_candidate,
        "core_snapshot_core_dual_exact": core_capture.snapshot.record == dual_capture.snapshot.record,
        "dual_native_candidate_comparison": projection,
        "graph_topology_transitive_equivalence": all([
            native_only.native_observation == dual.native_observation,
            core_candidate == dual_candidate,
            projection["exact"],
        ]),
        "mode_call_signature": "run_training_step(spec, collector=None, native_observer=None)",
        "native_graph_native_dual_exact": native_only.native_observation == dual.native_observation,
        "output_only_graph_directly_observed": False,
        "shared_training_entrypoint": True,
        "workload": spec.workload,
    }
    return CaptureModeSuite(
        output_only,
        native_only,
        core_only,
        core_capture,
        core_candidate,
        dual,
        dual_capture,
        dual_candidate,
        report,
    )
