from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    write_json,
)

from .builder import build_support_gfg


def build_archive(
    *,
    archive_index: Path,
    source_gfg_root: Path,
    output_root: Path,
    trainer_root: Path,
    contract_path: Path,
    max_checkpoints: int | None = None,
) -> dict[str, Any]:
    index = read_json(archive_index)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for entry in index["entries"]:
        bundle_id = entry["gfg_bundle_id"]
        destination = output_root / bundle_id
        receipt_path = destination / "build_receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt.get("status") != "PASS":
                raise RuntimeError(f"CSRG_ARCHIVE_PRIOR_RECEIPT_INVALID:{bundle_id}")
            print(
                "CSRG_ARCHIVE_BUNDLE_REUSED "
                + json.dumps({"bundle_id": bundle_id, "entry_id": entry["entry_id"]}),
                flush=True,
            )
        else:
            receipt = build_support_gfg(
                source_gfg_root / bundle_id,
                destination,
                trainer_root,
                entry_id=entry["entry_id"],
                contract_path=contract_path,
                max_checkpoints=max_checkpoints,
            )
        manifest = read_json(destination / "manifest.json")
        rows.append(
            {
                "build_receipt_sha256": file_sha256(receipt_path),
                "database_sha256": manifest["database_sha256"],
                "entry_id": entry["entry_id"],
                "gfg_bundle_id": bundle_id,
                "validation_sha256": receipt["validation_sha256"],
            }
        )
        print("CSRG_ARCHIVE_BUNDLE_PASS " + json.dumps(rows[-1], sort_keys=True), flush=True)
    archive = {
        "contract_sha256": file_sha256(contract_path),
        "schema": "nanogpt-support-redundancy-gfg-archive-v1",
        "source_archive_index_sha256": file_sha256(archive_index),
        "status": "PASS",
        "support_bundle_count": len(rows),
        "support_bundles": rows,
    }
    archive["archive_sha256"] = payload_sha256(archive)
    write_json(output_root / "archive_manifest.json", archive)
    experiment_root = Path(__file__).resolve().parent
    shutil.copy2(contract_path, output_root / "capture_contract_v2.json")
    shutil.copy2(experiment_root / "participant_query.py", output_root / "participant_query.py")
    shutil.copy2(experiment_root / "README.md", output_root / "README.md")
    package = {
        "archive_manifest_sha256": file_sha256(output_root / "archive_manifest.json"),
        "capture_contract_sha256": file_sha256(output_root / "capture_contract_v2.json"),
        "participant_query_sha256": file_sha256(output_root / "participant_query.py"),
        "readme_sha256": file_sha256(output_root / "README.md"),
        "schema": "nanogpt-support-redundancy-participant-package-v1",
        "status": "PASS",
        "support_bundle_count": len(rows),
    }
    package["package_sha256"] = payload_sha256(package)
    write_json(output_root / "package_manifest.json", package)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--source-gfg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--max-checkpoints", type=int)
    arguments = parser.parse_args()
    result = build_archive(
        archive_index=arguments.archive_index.resolve(),
        source_gfg_root=arguments.source_gfg_root.resolve(),
        output_root=arguments.output_root.resolve(),
        trainer_root=arguments.trainer_root.resolve(),
        contract_path=arguments.contract.resolve(),
        max_checkpoints=arguments.max_checkpoints,
    )
    print("CSRG_GFG_ARCHIVE_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
