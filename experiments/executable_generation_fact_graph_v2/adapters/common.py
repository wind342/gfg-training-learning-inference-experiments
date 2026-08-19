from __future__ import annotations

from typing import Any

from ..canonical_graph import canonical_hash, content_id


def empty_relation_store(execution_run_id: str) -> dict[str, Any]:
    material = {
        "schema_version": "validated-primitive-relation-store-v2",
        "execution_run_id": execution_run_id,
        "relations": [],
        "evidence": [],
    }
    return {
        **material,
        "relation_store_id": content_id("gfrstore2_", material),
    }


def complete_capture_audit(
    execution_run_id: str,
    *,
    domain: str,
) -> dict[str, Any]:
    material = {
        "execution_run_id": execution_run_id,
        "status": "CAPTURE_COMPLETE",
        "concurrency_inference_allowed": False,
        "concurrency_scope": "NOT_APPLICABLE",
        "domain": domain,
    }
    return {
        **material,
        "capture_audit_id": content_id("gfca2_", material),
    }


def normalize_native_relation_sidecar(
    native_sidecar: dict[str, Any],
) -> dict[str, Any]:
    """Add graph compiler fields without changing native relation rows."""
    relations = []
    for native in native_sidecar.get("relations", []):
        endpoint_level = native["endpoint_level"]
        relations.append(
            {
                **native,
                "source_endpoint_kind": endpoint_level,
                "target_endpoint_kind": endpoint_level,
                "relation_payload": native.get("relation_payload", {}),
                "primitive_or_derived": "primitive",
                "rule_id": native.get("rule_id"),
                "input_relation_refs": list(
                    native.get("input_relation_refs", [])
                ),
                "native_relation": native,
            }
        )
    identity_material = {
        "execution_run_id": native_sidecar["execution_run_id"],
        "native_schema_version": native_sidecar.get("schema_version"),
        "native_relation_ids": sorted(
            row["relation_id"] for row in native_sidecar.get("relations", [])
        ),
        "native_evidence_ids": sorted(
            row["evidence_id"] for row in native_sidecar.get("evidence", [])
        ),
    }
    return {
        **{
            key: value
            for key, value in native_sidecar.items()
            if key not in {"relations", "evidence"}
        },
        "relation_store_id": content_id(
            "gfrstore2_", identity_material
        ),
        "relations": relations,
        "evidence": list(native_sidecar.get("evidence", [])),
    }


def native_sidecar_semantic_material(
    sidecar: dict[str, Any],
) -> dict[str, Any]:
    """Canonical native content, excluding graph-internal store identity."""
    return {
        **{
            key: value
            for key, value in sidecar.items()
            if key
            not in {
                "relation_store_id",
                "relations",
                "evidence",
            }
        },
        "relations": sorted(
            (
                row.get("native_relation", row)
                for row in sidecar.get("relations", [])
            ),
            key=lambda row: row["relation_id"],
        ),
        "evidence": sorted(
            sidecar.get("evidence", []),
            key=lambda row: row["evidence_id"],
        ),
    }


def semantic_sidecar_hash(sidecar: dict[str, Any]) -> str:
    return canonical_hash(native_sidecar_semantic_material(sidecar))
