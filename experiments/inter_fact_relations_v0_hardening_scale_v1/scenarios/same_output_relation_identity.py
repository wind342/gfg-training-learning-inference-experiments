from __future__ import annotations

from typing import Any

from ..common import canonical_sha256, content_id
from ..src.run_identity import compare_runs
from .common import RuntimeScenarioBuilder


def _build_run(run_label: str, with_message: bool) -> RuntimeScenarioBuilder:
    builder = RuntimeScenarioBuilder(
        label="same-output-relation-identity-v1",
        execution_run_id=content_id("run1_", {"runtime_instance": run_label}),
    )
    occurrence_a, _ = builder.add_occurrence(
        actor_id="actor-A",
        sequence_index=0,
        operation="emit-same-value-A",
        semantic_slot=0,
        scope_id="same-output",
        fact_count=1,
    )
    occurrence_b, _ = builder.add_occurrence(
        actor_id="actor-B",
        sequence_index=0,
        operation="emit-same-value-B",
        semantic_slot=1,
        scope_id="same-output",
        fact_count=1,
    )
    if with_message:
        builder.add_message(
            occurrence_a["concrete_occurrence_instance_id"],
            occurrence_b["concrete_occurrence_instance_id"],
            channel_id="same-output-message-channel",
            payload={"control": "causal"},
        )
    return builder


def run() -> dict[str, Any]:
    causal = _build_run("causal-run", True)
    concurrent = _build_run("concurrent-run", False)
    ordinary_output = {"result": "same"}
    comparison = compare_runs(
        left_output=ordinary_output,
        right_output=ordinary_output,
        left_facts=causal.facts,
        right_facts=concurrent.facts,
        left_relation_hash=canonical_sha256(causal.primitive_relations),
        right_relation_hash=canonical_sha256(concurrent.primitive_relations),
    )
    return {
        "status": "PASS",
        **comparison,
        "semantic_occurrence_keys_equal": sorted(
            row["semantic_occurrence_key"] for row in causal.occurrences
        )
        == sorted(
            row["semantic_occurrence_key"] for row in concurrent.occurrences
        ),
        "concrete_occurrence_instance_ids_equal": sorted(
            row["concrete_occurrence_instance_id"] for row in causal.occurrences
        )
        == sorted(
            row["concrete_occurrence_instance_id"]
            for row in concurrent.occurrences
        ),
        "core_content_occurrence_ids_equal": sorted(
            row["core_content_occurrence_id"] for row in causal.occurrences
        )
        == sorted(
            row["core_content_occurrence_id"] for row in concurrent.occurrences
        ),
        "relation_sidecar_run_membership_equal": causal.run_id
        == concurrent.run_id,
    }
