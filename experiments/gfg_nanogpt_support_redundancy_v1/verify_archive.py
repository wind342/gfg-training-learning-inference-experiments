from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)

from .runtime import COMPONENTS, COMPONENT_PAIRS, HistoricalRunRuntime, tensor_sha256
from .support_gfg import SupportGFG, validate_support_gfg


def verify_archive(
    archive_root: Path,
    source_gfg_root: Path,
    trainer_root: Path,
    *,
    replay_steps: tuple[int, ...] = (100, 5000, 10000),
    output_path: Path | None = None,
) -> dict[str, Any]:
    archive = read_json(archive_root / "archive_manifest.json")
    rows: list[dict[str, Any]] = []
    total_forwards = 0
    for item in archive["support_bundles"]:
        bundle_id = item["gfg_bundle_id"]
        derived_bundle = archive_root / bundle_id
        source_bundle = source_gfg_root / bundle_id
        structural = validate_support_gfg(
            derived_bundle / "support_gfg.sqlite3",
            source_database_path=source_bundle / "participant_gfg.sqlite3",
            tensor_directory=derived_bundle / "tensor-objects",
        )
        require(structural["status"] == "PASS", "CSRG_REPLAY_STRUCTURAL_VALIDATION_FAILED")
        graph = SupportGFG(derived_bundle / "support_gfg.sqlite3")
        checkpoint_rows = {row["optimizer_step"]: row for row in graph.checkpoints()}
        graph.close()
        runtime = HistoricalRunRuntime.open(source_bundle, trainer_root, device="cuda", reference_step=100)
        checks = 0
        try:
            for step in replay_steps:
                expected = checkpoint_rows[step]
                parameter_rows = runtime.load_checkpoint(step)
                first = runtime.forward()
                second = runtime.forward()
                require(torch.equal(first, second), "CSRG_REPLAY_BASELINE_REPEAT_MISMATCH")
                require(tensor_sha256(first) == expected["current_baseline_logits_sha256"], "CSRG_REPLAY_BASELINE_HASH_MISMATCH")
                checks += 2
                single_hashes = json.loads(expected["single_gate_hashes_json"])
                pair_hashes = json.loads(expected["pair_gate_hashes_json"])
                for component in COMPONENTS:
                    require(tensor_sha256(runtime.forward((component,))) == single_hashes[component], "CSRG_REPLAY_SINGLE_GATE_HASH_MISMATCH")
                    checks += 1
                for pair in COMPONENT_PAIRS:
                    key = pair[0] + "+" + pair[1]
                    require(tensor_sha256(runtime.forward(pair)) == pair_hashes[key], "CSRG_REPLAY_PAIR_GATE_HASH_MISMATCH")
                    checks += 1
                for name, parameter in runtime.model.named_parameters():
                    require(tensor_sha256(parameter) == parameter_rows[name]["content_sha256"], "CSRG_REPLAY_PARAMETER_MUTATION")
                    checks += 1
                total_forwards += 12
        finally:
            runtime.close()
        rows.append(
            {
                "bundle_id": bundle_id,
                "entry_id": item["entry_id"],
                "replay_checks": checks,
                "replay_steps": list(replay_steps),
                "status": "PASS",
                "structural_validation_sha256": structural["validation_sha256"],
            }
        )
        print("CSRG_INDEPENDENT_REPLAY_BUNDLE_PASS " + json.dumps(rows[-1], sort_keys=True), flush=True)
    result = {
        "archive_manifest_sha256": file_sha256(archive_root / "archive_manifest.json"),
        "bundle_count": len(rows),
        "independent_actual_forward_count": total_forwards,
        "replay_rows": rows,
        "schema": "nanogpt-support-redundancy-independent-replay-v1",
        "status": "PASS",
    }
    result["replay_sha256"] = payload_sha256(result)
    if output_path is not None:
        write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source-gfg-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify_archive(
        arguments.archive_root.resolve(),
        arguments.source_gfg_root.resolve(),
        arguments.trainer_root.resolve(),
        output_path=arguments.output.resolve(),
    )
    print("CSRG_INDEPENDENT_REPLAY_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
