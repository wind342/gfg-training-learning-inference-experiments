from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    read_json,
    require,
)

from .execution import _checked_result, _read_checked


def _inventory(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            relative = path.relative_to(root).as_posix()
            result[relative] = file_sha256(path)
    return dict(sorted(result.items()))


def validate_p2_replay(
    *,
    primary_root: Path,
    replay_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    primary_seal = _read_checked(primary_root / "PRE_TARGET_RESPONSE_SEAL.json", "nanogpt-p2-pre-target-response-seal-v1")
    replay_seal = _read_checked(replay_root / "PRE_TARGET_RESPONSE_SEAL.json", "nanogpt-p2-pre-target-response-seal-v1")
    require(primary_seal == replay_seal, "P2_REPLAY_SEAL_NOT_IDENTICAL")
    primary_validation = _read_checked(primary_root / "p2_response_pre_target_validation.json", "nanogpt-p2-response-pre-target-validation-v1")
    replay_validation = _read_checked(replay_root / "p2_response_pre_target_validation.json", "nanogpt-p2-response-pre-target-validation-v1")
    require(primary_validation == replay_validation, "P2_REPLAY_PRETARGET_VALIDATION_NOT_IDENTICAL")
    primary_inventory = _inventory(primary_root, ("*.json", "*.npy"))
    replay_inventory = _inventory(replay_root, ("*.json", "*.npy"))
    ignored = {
        "p2_response_pre_target_validation.json",
        "p2_response_independent_replay_validation.json",
    }
    primary_scientific = {key: value for key, value in primary_inventory.items() if key not in ignored}
    replay_scientific = {key: value for key, value in replay_inventory.items() if key not in ignored}
    require(primary_scientific == replay_scientific, "P2_REPLAY_SCIENTIFIC_FILE_INVENTORY_MISMATCH")
    return _checked_result(
        output_path,
        {
            "schema": "nanogpt-p2-independent-replay-validation-v1",
            "status": "PASS",
            "protocol_sha256": primary_seal["protocol_sha256"],
            "seal_result_sha256": primary_seal["result_sha256"],
            "pretarget_validation_result_sha256": primary_validation["result_sha256"],
            "scientific_file_count": len(primary_scientific),
            "json_file_count": sum(key.endswith(".json") for key in primary_scientific),
            "tensor_file_count": sum(key.endswith(".npy") for key in primary_scientific),
            "all_relative_paths_identical": True,
            "all_file_sha256_identical": True,
            "independent_replay_exact": True,
            "native_target_content_opened": False,
        },
    )


__all__ = ["validate_p2_replay"]
