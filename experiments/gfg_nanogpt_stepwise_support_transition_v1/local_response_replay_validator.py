from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    payload_sha256,
    read_json,
    relative_file_manifest,
    require,
    write_json,
)


def _without_path_and_result(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: child
        for key, child in value.items()
        if key not in {"output_root", "result_sha256"}
    }


def validate_local_response_replay(*, primary_root: Path, replay_root: Path) -> dict[str, Any]:
    primary_receipt = read_json(primary_root / "local_response_pair_receipt.json")
    replay_receipt = read_json(replay_root / "local_response_pair_receipt.json")
    require(
        _without_path_and_result(primary_receipt) == _without_path_and_result(replay_receipt),
        "SST_LOCAL_RESPONSE_REPLAY_PAIR_RECEIPT_MISMATCH",
    )
    receiver_manifests: dict[str, Any] = {}
    for label in ("A", "B"):
        primary_manifest = relative_file_manifest(primary_root / f"receiver-{label}")
        replay_manifest = relative_file_manifest(replay_root / f"receiver-{label}")
        require(primary_manifest == replay_manifest, f"SST_LOCAL_RESPONSE_REPLAY_RECEIVER_MISMATCH:{label}")
        receiver_manifests[label] = {
            "file_count": len(primary_manifest),
            "directory_sha256": payload_sha256(primary_manifest),
        }
    primary_validation = read_json(primary_root / "local_response_jk_validation.json")
    replay_validation = read_json(replay_root / "local_response_jk_validation.json")
    primary_validation_material = {
        key: value
        for key, value in primary_validation.items()
        if key not in {"pair_receipt_sha256", "validation_sha256"}
    }
    replay_validation_material = {
        key: value
        for key, value in replay_validation.items()
        if key not in {"pair_receipt_sha256", "validation_sha256"}
    }
    require(
        primary_validation_material == replay_validation_material,
        "SST_LOCAL_RESPONSE_REPLAY_VALIDATION_MISMATCH",
    )
    material = {
        "schema": "nanogpt-local-response-jk-independent-replay-v1",
        "status": "PASS",
        "pair_receipt_material_sha256": payload_sha256(_without_path_and_result(primary_receipt)),
        "receiver_manifests": receiver_manifests,
        "primary_evidence_validation_sha256": primary_validation["validation_sha256"],
        "replay_evidence_validation_sha256": replay_validation["validation_sha256"],
        "normalized_validation_material_sha256": payload_sha256(primary_validation_material),
        "byte_exact_receiver_evidence": True,
        "future_information_used": False,
    }
    result = {**material, "replay_validation_sha256": payload_sha256(material)}
    write_json(primary_root / "local_response_jk_independent_replay.json", result)
    return result
