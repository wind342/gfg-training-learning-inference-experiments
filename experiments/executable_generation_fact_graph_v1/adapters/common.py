from __future__ import annotations

from typing import Any

from ..canonical_graph import content_id


def empty_relation_store(execution_run_id: str) -> dict[str, Any]:
    return {
        "relation_store_id": content_id(
            "gfrstore1_",
            {"execution_run_id": execution_run_id, "relations": []},
        ),
        "execution_run_id": execution_run_id,
        "relations": [],
        "evidence": [],
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
        "capture_audit_id": content_id("gfca1_", material),
    }
