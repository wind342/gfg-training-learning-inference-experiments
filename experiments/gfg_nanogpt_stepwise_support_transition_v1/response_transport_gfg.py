from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    GraphRef,
    SupportGFGWriter,
)

from .reciprocal_gfg_validator import _main_object_index
from .stepwise_gfg import _external_tensor, _object, _occurrence


GRAPH_SCHEMA = "nanogpt-response-transport-cross-gfg-v1"
BLOCK_SCHEMA = "nanogpt-response-transport-cross-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-response-transport-cross-gfg-manifest-v1"


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    target = graph_root / "tensor-objects"
    target.mkdir(parents=True, exist_ok=True)
    staged = 0
    for source in sorted((evidence_root / "tensor-objects").glob("*.npy")):
        destination = target / source.name
        if destination.exists():
            require(file_sha256(destination) == file_sha256(source), "SST_RESPONSE_TRANSPORT_GFG_TENSOR_COLLISION")
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        staged += 1
    return staged


def _load_source_graph(graph_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_json(graph_root / "local_response_jk_gfg_manifest.json")
    validation = read_json(graph_root / "local_response_jk_gfg_validation.json")
    require(manifest["status"] in {"CAPTURE_CLOSED", "PASS"}, "SST_RESPONSE_TRANSPORT_GFG_SOURCE_NOT_CLOSED")
    require(validation["status"] == "PASS", "SST_RESPONSE_TRANSPORT_GFG_SOURCE_NOT_VALIDATED")
    require(validation["manifest_sha256"] == manifest["manifest_sha256"], "SST_RESPONSE_TRANSPORT_GFG_SOURCE_VALIDATION_MISMATCH")
    return manifest, _main_object_index(graph_root / str(manifest["database"]))


def build_response_transport_gfg(
    *,
    evidence_root: Path,
    graph_root: Path,
    protocol_path: Path,
    a_skip_graph_root: Path,
    b_skip_graph_root: Path,
    a_native_full_graph_root: Path,
    b_native_full_graph_root: Path,
) -> dict[str, Any]:
    require(not graph_root.exists(), "SST_RESPONSE_TRANSPORT_GFG_ROOT_EXISTS")
    graph_root.mkdir(parents=True)
    protocol = read_json(protocol_path)
    require(protocol["output_graph_schema"] == GRAPH_SCHEMA, "SST_RESPONSE_TRANSPORT_GFG_PROTOCOL_SCHEMA_MISMATCH")
    validation = read_json(evidence_root / "response_transport_cross_validation.json")
    require(validation["status"] == "PASS", "SST_RESPONSE_TRANSPORT_GFG_EVIDENCE_NOT_VALIDATED")
    graph_roots = {
        "A": {"skip": a_skip_graph_root, "native_full": a_native_full_graph_root},
        "B": {"skip": b_skip_graph_root, "native_full": b_native_full_graph_root},
    }
    sources: dict[str, dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]]] = {
        donor: {kind: _load_source_graph(root) for kind, root in kinds.items()}
        for donor, kinds in graph_roots.items()
    }
    staged_count = _stage_tensors(evidence_root, graph_root)
    writer = SupportGFGWriter(
        graph_root / "response_transport_cross_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id="nanogpt-response-transport-cross-v1",
        source_bundle_id=str(validation["validation_sha256"]),
        contract_sha256=file_sha256(protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("response_transport_contract", 0)
    contract_ref = _object(
        writer,
        semantic_key="response-transport-cross:frozen-contract",
        role="frozen_response_transport_cross_contract",
        optimizer_step=0,
        payload={
            "protocol": protocol,
            "protocol_sha256": file_sha256(protocol_path),
            "evidence_validation_sha256": validation["validation_sha256"],
        },
        object_kind="frozen_contract",
    )
    writer.flush_block()

    context_catalog: dict[str, str] = {}
    transport_catalog: dict[str, str] = {}
    for donor in ("A", "B"):
        for receiver in ("A", "B"):
            row = validation["rows"][donor][receiver]
            skip_manifest, skip_objects = sources[donor]["skip"]
            full_manifest, full_objects = sources[donor]["native_full"]
            skip_summary_id = str(skip_manifest["response_summary_catalog"][receiver])
            full_summary_id = str(full_manifest["response_summary_catalog"][receiver])
            skip_summary = writer.origin(
                skip_objects[skip_summary_id],
                source_bundle_id=str(skip_manifest["manifest_sha256"]),
                source_graph_schema=str(skip_manifest["schema"]),
            )
            full_summary = writer.origin(
                full_objects[full_summary_id],
                source_bundle_id=str(full_manifest["manifest_sha256"]),
                source_graph_schema=str(full_manifest["schema"]),
            )
            optimizer_step = int(full_objects[full_summary_id]["optimizer_step"])
            writer.start_block(f"response_transport_{donor}_{receiver}", optimizer_step)
            transport_refs: list[GraphRef] = []
            for role, numeric in sorted(row["numeric"].items()):
                occurrence = _occurrence(
                    writer,
                    occurrence_type="response_state_transport_computation_occurrence",
                    optimizer_step=optimizer_step,
                    operation="subtract_native_full_and_skip_local_first_order_response",
                    contract_id=str(protocol["protocol_id"]),
                    payload={"donor_label": donor, "receiver_label": receiver, "numeric_role": role},
                )
                outcome = _external_tensor(
                    writer,
                    reference=numeric["response_transport"],
                    semantic_key=f"response-transport:{donor}:{receiver}:{role}",
                    role=f"local_response_transport_of:{role}",
                    optimizer_step=optimizer_step,
                )
                writer.bind(
                    occurrence,
                    [
                        (skip_summary, "skip_state_local_response_summary"),
                        (full_summary, "native_full_state_local_response_summary"),
                        (contract_ref, "frozen_response_transport_contract"),
                    ],
                    outcome,
                    payload={"outcome_kind": "receiver_local_response_transport"},
                )
                transport_refs.append(outcome)
                transport_catalog[f"{donor}:{receiver}:{role}"] = outcome.object_id

            summary_occurrence = _occurrence(
                writer,
                occurrence_type="response_transport_context_adjudication_occurrence",
                optimizer_step=optimizer_step,
                operation="apply_frozen_response_transport_context_criterion",
                contract_id=str(protocol["protocol_id"]),
                payload={"donor_label": donor, "receiver_label": receiver},
            )
            summary = _object(
                writer,
                semantic_key=f"response-transport:{donor}:{receiver}:context-adjudication",
                role="validated_response_transport_context_adjudication",
                optimizer_step=optimizer_step,
                payload={key: value for key, value in row.items() if key != "numeric"},
                object_kind="validated_analysis_result",
            )
            writer.bind(
                summary_occurrence,
                [(value, "numeric_response_transport") for value in transport_refs]
                + [(contract_ref, "frozen_response_transport_contract")],
                summary,
                payload={"outcome_kind": "response_transport_context_adjudication"},
            )
            context_catalog[f"{donor}:{receiver}"] = summary.object_id
            writer.flush_block()

    writer.start_block("response_transport_overall_adjudication", 0)
    context_objects: list[GraphRef] = []
    # Context objects are local to already flushed blocks, so reconstruct their graph references.
    for donor in ("A", "B"):
        for receiver in ("A", "B"):
            row = validation["rows"][donor][receiver]
            payload = {key: value for key, value in row.items() if key != "numeric"}
            context_objects.append(
                GraphRef(
                    context_catalog[f"{donor}:{receiver}"],
                    payload_sha256(payload),
                    "validated_response_transport_context_adjudication",
                    "derived_object",
                )
            )
    overall_occurrence = _occurrence(
        writer,
        occurrence_type="response_transport_overall_adjudication_occurrence",
        optimizer_step=0,
        operation="apply_frozen_response_transport_overall_criterion",
        contract_id=str(protocol["protocol_id"]),
        payload={"context_count": 4},
    )
    overall = _object(
        writer,
        semantic_key="response-transport:overall-adjudication",
        role="validated_response_transport_overall_adjudication",
        optimizer_step=0,
        payload={
            "mechanical_scientific_outcome": validation["mechanical_scientific_outcome"],
            "strict_transport_hypothesis_satisfied": validation["strict_transport_hypothesis_satisfied"],
            "response_transport_falsified": validation["response_transport_falsified"],
            "observation_omission_supported": validation["observation_omission_supported"],
        },
        object_kind="validated_analysis_result",
    )
    writer.bind(
        overall_occurrence,
        [(value, "validated_context_adjudication") for value in context_objects]
        + [(contract_ref, "frozen_response_transport_contract")],
        overall,
        payload={"outcome_kind": "response_transport_overall_adjudication"},
    )
    writer.flush_block()
    closed = writer.close()
    material = {
        **closed,
        "response_transport_protocol_sha256": file_sha256(protocol_path),
        "evidence_validation_sha256": validation["validation_sha256"],
        "input_graph_manifests": {
            f"{donor}:{kind}": values[0]["manifest_sha256"]
            for donor, kinds in sources.items()
            for kind, values in kinds.items()
        },
        "context_catalog": context_catalog,
        "transport_catalog": transport_catalog,
        "overall_adjudication_object_id": overall.object_id,
        "staged_tensor_payload_count": staged_count,
        "future_information_used": False,
    }
    manifest = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(graph_root / "response_transport_cross_gfg_manifest.json", manifest)
    return manifest


__all__ = ["build_response_transport_gfg", "GRAPH_SCHEMA"]
