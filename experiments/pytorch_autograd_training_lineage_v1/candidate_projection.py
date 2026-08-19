from __future__ import annotations

from typing import Any

from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from .graph_canonicalization import canonicalize_graph


def project_core_to_autograd_graph(
    snapshot: ValidatedSnapshot,
    validation: SnapshotValidation,
    profile: dict[str, Any],
    crosswalk: dict[str, Any],
) -> dict[str, Any]:
    """Project only validated Core facts through the pre-frozen crosswalk."""
    if validation.snapshot_id != snapshot.snapshot_id:
        raise ValueError("CANDIDATE_SNAPSHOT_VALIDATION_MISMATCH")
    if profile["torch_version"] != crosswalk["torch_version"]:
        raise ValueError("CANDIDATE_PROFILE_CROSSWALK_VERSION_MISMATCH")
    declarations = {
        row["core_occurrence_type"]: row for row in crosswalk["declarations"]
    }
    tables = snapshot.tables
    occurrences = {
        row["generation_occurrence_id"]: row for row in tables.generation_occurrences
    }
    sources = {
        row["source_information_id"]: row for row in tables.source_information_records
    }
    generated = {
        row["generated_origin_id"]: row for row in tables.generated_origins
    }
    supports = {
        row["support_id"]: row for row in tables.perceptual_support_records
    }
    producer_by_support: dict[str, str] = {}
    bindings_by_occurrence: dict[str, list[dict]] = {}
    for binding in tables.generation_bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            producer_by_support.setdefault(
                outcome["support_id"], binding["generation_occurrence_id"]
            )
        bindings_by_occurrence.setdefault(binding["generation_occurrence_id"], []).append(binding)

    selected_occurrences = {
        occurrence_id: occurrence
        for occurrence_id, occurrence in occurrences.items()
        if occurrence["occurrence_stage"] == "forward"
        and occurrence["occurrence_type"] in declarations
    }
    raw_nodes: dict[str, dict[str, Any]] = {}
    raw_edges: list[dict[str, Any]] = []
    root_key: str | None = None
    for occurrence_id, occurrence in selected_occurrences.items():
        raw_nodes[occurrence_id] = {
            "key": occurrence_id,
            "node_type": declarations[occurrence["occurrence_type"]]["native_node_name"],
        }
        output_ref = occurrence["occurrence_payload"]["output_ref"]
        output_support = next(
            row for row in supports.values()
            if row["support_payload"]["support_key"] == output_ref
        )
        if output_support["support_payload"]["support_kind"] == "loss":
            if root_key is not None:
                raise ValueError("CANDIDATE_MULTIPLE_GRAPH_ROOTS")
            root_key = occurrence_id

    if root_key is None:
        raise ValueError("CANDIDATE_GRAPH_ROOT_MISSING")

    leaf_type = crosswalk["leaf_declaration"]["native_node_name"]
    for occurrence_id in selected_occurrences:
        input_bindings = [
            binding for binding in bindings_by_occurrence.get(occurrence_id, [])
            if binding["relation_role"].startswith("operation_input|")
        ]
        parsed: list[tuple[int, dict]] = []
        for binding in input_bindings:
            parts = dict(
                item.split("=", 1)
                for item in binding["relation_role"].split("|")[1:]
            )
            parsed.append((int(parts["slot"]), binding))
        parsed.sort(key=lambda item: item[0])
        if [slot for slot, _binding in parsed] != list(range(len(parsed))):
            raise ValueError("CANDIDATE_INPUT_SLOT_GAP")
        for slot, binding in parsed:
            origin = binding["origin_reference"]
            target_key: str | None
            if origin["kind"] == "generated_origin":
                source_support_id = generated[origin["generated_origin_id"]]["origin_payload"]["source_support_id"]
                target_key = producer_by_support[source_support_id]
                if target_key not in selected_occurrences:
                    raise ValueError("CANDIDATE_UNDECLARED_OPERATION_DEPENDENCY")
            else:
                source = sources[origin["source_information_id"]]
                tensor = source["source_payload"].get("tensor")
                if tensor is None or not tensor.get("requires_grad", False):
                    target_key = None
                else:
                    target_key = "leaf:" + source["source_information_id"]
                    raw_nodes.setdefault(target_key, {"key": target_key, "node_type": leaf_type})
            raw_edges.append({
                "output_nr": 0,
                "slot": slot,
                "source_key": occurrence_id,
                "target_key": target_key,
            })
    return canonicalize_graph({
        "edges": raw_edges,
        "nodes": list(raw_nodes.values()),
        "root_key": root_key,
    })
