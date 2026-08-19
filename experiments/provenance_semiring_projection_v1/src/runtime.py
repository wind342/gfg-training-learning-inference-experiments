from __future__ import annotations

from typing import Any, Callable

from .candidate_nx import project_snapshot_to_nx
from .core_capture import core_snapshot_from_events
from .exact_comparison import compare_nx_corpora
from .native_nx import evaluate_native_nx
from .ordinary_execution import execute_ordinary
from .workloads import workload_by_id


def run_relational_generation(spec: dict[str, Any], collector: Callable[[dict[str, Any]], object] | None = None) -> dict[str, Any]:
    """Public experiment runtime entry with four explicitly separated modes."""
    mode = spec.get("mode")
    workload = workload_by_id(spec["workload_id"])
    variant = spec.get("variant")
    if mode == "output_only":
        ordinary, measurements = execute_ordinary(workload, variant=variant, collector=collector)
        return {"mode": mode, "ordinary_output": ordinary, "measurements": measurements}
    if collector is not None:
        raise ValueError("an external collector is accepted only in output_only mode")
    if mode == "native_semiring_only":
        return {"mode": mode, "native": evaluate_native_nx(workload, variant=variant)}
    if mode == "core_only":
        ordinary, measurements, snapshot, validation = core_snapshot_from_events(workload, variant=variant)
        return {
            "mode": mode,
            "ordinary_output": ordinary,
            "measurements": measurements,
            "snapshot_id": snapshot.snapshot_id,
            "candidate": project_snapshot_to_nx(snapshot, validation),
        }
    if mode == "dual":
        native = evaluate_native_nx(workload, variant=variant)
        ordinary, measurements, snapshot, validation = core_snapshot_from_events(workload, variant=variant)
        candidate_projection = project_snapshot_to_nx(snapshot, validation)
        candidate = {
            "schema_version": "core-projected-nx-corpus-v1",
            "results": [{
                "workload_id": workload["id"],
                "variant": native["variant"],
                "source_variables": candidate_projection["source_variables"],
                "outputs": candidate_projection["outputs"],
            }],
        }
        native_corpus = {"schema_version": "native-nx-corpus-v1", "results": [native]}
        semantic_exact = (
            native["source_variables"] == candidate_projection["source_variables"]
            and native["outputs"] == candidate_projection["outputs"]
        )
        return {
            "mode": mode,
            "ordinary_output": ordinary,
            "measurements": measurements,
            "snapshot_id": snapshot.snapshot_id,
            "native": native_corpus["results"][0],
            "candidate": candidate["results"][0],
            "semantic_exact": semantic_exact,
        }
    raise ValueError(f"unknown runtime mode: {mode!r}")

