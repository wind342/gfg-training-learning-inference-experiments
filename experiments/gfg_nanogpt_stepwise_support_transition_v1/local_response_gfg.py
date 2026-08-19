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
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .contracts import ComponentRegistry, ProbeContract
from .execution import _read_checked
from .local_response import LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES
from .reciprocal_gfg_validator import _main_object_index
from .stepwise_gfg import _bind_all, _emit_probe, _external_tensor, _object, _occurrence


GRAPH_SCHEMA = "nanogpt-local-response-jk-gfg-v1"
BLOCK_SCHEMA = "nanogpt-local-response-jk-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-local-response-jk-gfg-manifest-v1"


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    target = graph_root / "tensor-objects"
    target.mkdir(parents=True, exist_ok=True)
    staged: set[str] = set()
    for label in ("A", "B"):
        source_root = evidence_root / f"receiver-{label}" / "tensor-objects"
        for source in sorted(source_root.glob("*.npy")):
            destination = target / source.name
            source_sha = file_sha256(source)
            if destination.exists():
                require(file_sha256(destination) == source_sha, "SST_LOCAL_RESPONSE_GFG_TENSOR_COLLISION")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            staged.add(source.name)
    return len(staged)


def _prior_origin(
    writer: SupportGFGWriter,
    *,
    object_index: dict[str, dict[str, Any]],
    object_id: str,
    prior_manifest: dict[str, Any],
) -> GraphRef:
    require(object_id in object_index, f"SST_LOCAL_RESPONSE_GFG_PRIOR_OBJECT_MISSING:{object_id}")
    return writer.origin(
        object_index[object_id],
        source_bundle_id=str(prior_manifest["manifest_sha256"]),
        source_graph_schema=str(prior_manifest["schema"]),
    )


def build_local_response_gfg(
    *,
    evidence_root: Path,
    reciprocal_graph_root: Path,
    graph_root: Path,
    local_response_protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    require(not graph_root.exists(), "SST_LOCAL_RESPONSE_GFG_ROOT_EXISTS")
    graph_root.mkdir(parents=True)
    protocol = read_json(local_response_protocol_path)
    branches = tuple(str(value) for value in protocol["branches"])
    require(
        branches in {LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES},
        "SST_LOCAL_RESPONSE_GFG_BRANCHES_INVALID",
    )
    receiver_state_kind = str(protocol.get("receiver_state_kind", "skip"))
    validation = read_json(evidence_root / "local_response_jk_validation.json")
    require(validation["status"] == "PASS", "SST_LOCAL_RESPONSE_GFG_EVIDENCE_NOT_VALIDATED")
    prior_manifest = read_json(reciprocal_graph_root / "reciprocal_matched_pair_gfg_manifest.json")
    prior_database = reciprocal_graph_root / str(prior_manifest["database"])
    prior_objects = _main_object_index(prior_database)
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    staged_count = _stage_tensors(evidence_root, graph_root)
    writer = SupportGFGWriter(
        graph_root / "local_response_jk_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id="nanogpt-local-response-jk-v1",
        source_bundle_id=str(prior_manifest["manifest_sha256"]),
        contract_sha256=file_sha256(local_response_protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("local_response_contract", 0)
    contract_ref = _object(
        writer,
        semantic_key="local-response-jk:frozen-protocol",
        role="frozen_local_response_protocol",
        optimizer_step=0,
        payload={
            "protocol": protocol,
            "protocol_sha256": file_sha256(local_response_protocol_path),
            "evidence_validation_sha256": validation["validation_sha256"],
            "prior_reciprocal_manifest_sha256": prior_manifest["manifest_sha256"],
        },
        object_kind="frozen_contract",
    )
    registry_ref = _object(
        writer,
        semantic_key="local-response-jk:component-registry",
        role="versioned_component_registry",
        optimizer_step=0,
        payload={"registry": read_json(component_registry_path), "registry_sha256": registry.source_sha256},
        object_kind="frozen_contract",
    )
    probe_ref = _object(
        writer,
        semantic_key="local-response-jk:probe-contract",
        role="versioned_probe_contract",
        optimizer_step=0,
        payload={"probe_contract": read_json(probe_contract_path), "probe_contract_sha256": probe_contract.source_sha256},
        object_kind="frozen_contract",
    )
    writer.flush_block()

    donor_update = _prior_origin(
        writer,
        object_index=prior_objects,
        object_id=str(protocol["donor_update"]["source_object_id"]),
        prior_manifest=prior_manifest,
    )
    state_catalog: dict[str, str] = {}
    response_summaries: dict[str, GraphRef] = {}
    for label in ("A", "B"):
        endpoint = next(row for row in protocol["receivers"] if row["label"] == label)
        optimizer_step = int(endpoint["optimizer_step"]) + 1
        prior_state_id = str(
            prior_manifest["state_catalog"][
                f"{label}:1:{receiver_state_kind}"
            ]
        )
        prior_state = _prior_origin(
            writer,
            object_index=prior_objects,
            object_id=prior_state_id,
            prior_manifest=prior_manifest,
        )
        entry_root = evidence_root / f"receiver-{label}"
        response = _read_checked(entry_root / "local_response_jk.json", "nanogpt-local-response-jk-v1")
        writer.start_block(f"local_response_receiver_{label}", optimizer_step)
        branch_states: dict[str, GraphRef] = {}
        probe_summaries: dict[str, GraphRef] = {}
        for branch in branches:
            state_record = _read_checked(entry_root / "h-001" / f"{branch}-state.json", "nanogpt-local-response-state-v1")
            occurrence = _occurrence(
                writer,
                occurrence_type="local_parameter_displacement_application_occurrence",
                optimizer_step=optimizer_step,
                operation="apply_scaled_realized_parameter_delta",
                contract_id=str(protocol["protocol_id"]),
                payload={
                    "receiver_label": label,
                    "branch": branch,
                    "scale": state_record["scale"],
                    "adam_state_transplanted": False,
                    "receiver_state_kind": receiver_state_kind,
                },
            )
            state_outcome = _object(
                writer,
                semantic_key=f"local-response:{label}:h1:{branch}:state",
                role=f"{branch}_local_response_state",
                optimizer_step=optimizer_step,
                payload={"state_record": state_record, "receiver_label": label, "branch": branch},
                object_kind="restorable_local_response_state",
            )
            sources = [(prior_state, "immutable_receiver_prestate"), (contract_ref, "frozen_local_response_contract")]
            if branch != "baseline":
                sources.append((donor_update, "scaled_realized_donor_parameter_delta"))
            writer.bind(occurrence, sources, state_outcome, payload={"outcome_kind": "restorable_local_response_state"})
            branch_states[branch] = state_outcome
            state_catalog[f"{label}:{branch}"] = state_outcome.object_id
            observation = _read_checked(
                entry_root / "probe-observations" / "CSRG-4C-v1" / f"{state_record['state']['state_id']}.json",
                "nanogpt-stepwise-probe-observation-v1",
            )
            probe_summaries[branch] = _emit_probe(
                writer,
                observation=observation,
                state_origin=state_outcome,
                validation_sources=[],
                contract_ref=probe_ref,
                registry_ref=registry_ref,
                semantic_prefix=f"local-response:{label}:h1:{branch}",
                optimizer_step=optimizer_step,
                probe_contract=probe_contract,
            )

        response_occurrence = _occurrence(
            writer,
            occurrence_type="central_finite_difference_response_occurrence",
            optimizer_step=optimizer_step,
            operation="compute_registered_central_finite_difference_responses",
            contract_id=str(protocol["protocol_id"]),
            payload={"receiver_label": label, "epsilon": protocol["epsilon"], "categorical_values_subtracted": False},
        )
        response_sources = [
            (probe_summaries["baseline"], "baseline_probe_result"),
            (probe_summaries["plus_epsilon"], "plus_epsilon_probe_result"),
            (probe_summaries["minus_epsilon"], "minus_epsilon_probe_result"),
            (contract_ref, "frozen_finite_difference_contract"),
        ]
        response_outputs: list[tuple[GraphRef, str]] = []
        for key, refs in sorted(response["numeric_responses"].items()):
            numeric_fields = ["j_first_order", "k_curvature"]
            if "full_delta" in refs:
                numeric_fields.append("full_delta")
            for field in numeric_fields:
                output = _external_tensor(
                    writer,
                    reference=refs[field],
                    semantic_key=f"local-response:{label}:h1:{key}:{field}",
                    role=f"{field}_of:{key}",
                    optimizer_step=optimizer_step,
                )
                response_outputs.append((output, field + ":" + key))
        for key, refs in sorted(response["categorical_transitions"].items()):
            categorical_fields = [
                "baseline",
                "plus",
                "minus",
                "plus_changed_mask",
                "minus_changed_mask",
            ]
            if "full" in refs:
                categorical_fields.extend(["full", "full_changed_mask"])
            for field in categorical_fields:
                output = _external_tensor(
                    writer,
                    reference=refs[field],
                    semantic_key=f"local-response:{label}:h1:{key}:{field}",
                    role=f"categorical_{field}_of:{key}",
                    optimizer_step=optimizer_step,
                )
                response_outputs.append((output, field + ":" + key))
        _bind_all(writer, response_occurrence, response_sources, response_outputs)
        summary = _object(
            writer,
            semantic_key=f"local-response:{label}:h1:validated-response-summary",
            role="validated_local_response_summary",
            optimizer_step=optimizer_step,
            payload={
                "receiver_label": label,
                "response_result_sha256": response["result_sha256"],
                "validation": validation["receiver_rows"][label],
                "numeric_response_count": len(response["numeric_responses"]),
                "categorical_response_count": len(response["categorical_transitions"]),
            },
            object_kind="validated_analysis_result",
        )
        writer.bind(response_occurrence, [*response_sources, *response_outputs], summary, payload={"outcome_kind": "validated_local_response_summary"})
        response_summaries[label] = summary
        writer.flush_block()

    writer.start_block("local_response_receiver_comparison", 0)
    compare_occurrence = _occurrence(
        writer,
        occurrence_type="receiver_local_response_comparison_occurrence",
        optimizer_step=0,
        operation="compare_receiver_conditioned_local_responses",
        contract_id=str(protocol["protocol_id"]),
        payload={"same_realized_donor_parameter_delta": True, "receiver_labels": ["A", "B"]},
    )
    comparison = _object(
        writer,
        semantic_key="local-response:A-vs-B:receiver-conditioned-JK-contrasts",
        role="receiver_conditioned_local_response_contrasts",
        optimizer_step=0,
        payload={
            "receiver_contrasts": validation["receiver_contrasts"],
            "interpretation_performed": False,
            "scope": protocol["adjudication"]["scope"],
        },
        object_kind="validated_analysis_result",
    )
    writer.bind(
        compare_occurrence,
        [(response_summaries["A"], "receiver_A_local_response"), (response_summaries["B"], "receiver_B_local_response"), (contract_ref, "frozen_comparison_contract")],
        comparison,
        payload={"outcome_kind": "receiver_conditioned_local_response_contrasts"},
    )
    writer.flush_block()
    closed = writer.close()
    material = {
        **closed,
        "status": "PASS",
        "local_response_protocol_sha256": file_sha256(local_response_protocol_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "evidence_validation_sha256": validation["validation_sha256"],
        "prior_reciprocal_manifest_sha256": prior_manifest["manifest_sha256"],
        "staged_tensor_payload_count": staged_count,
        "state_catalog": state_catalog,
        "response_summary_catalog": {label: ref.object_id for label, ref in response_summaries.items()},
        "comparison_object_id": comparison.object_id,
    }
    if branches == LOCAL_RESPONSE_TRANSPORT_BRANCHES or receiver_state_kind != "skip":
        material.update(
            {
                "receiver_state_kind": receiver_state_kind,
                "branches": list(branches),
            }
        )
    manifest = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(graph_root / "local_response_jk_gfg_manifest.json", manifest)
    return manifest
