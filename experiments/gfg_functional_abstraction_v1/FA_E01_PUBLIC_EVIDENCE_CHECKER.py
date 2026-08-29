from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.gfg_functional_abstraction_v1.FA_E01_INDEPENDENT_RECOMPUTE import (
    DEFAULT_PROJECTION_ROOT,
    DEFAULT_SOURCE_ROOT,
    independent_recompute,
)


HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def quick_check(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "fa-e01-public-evidence-manifest-v1", "FA_PUBLIC_SCHEMA")
    checked = 0
    for item in manifest["files"]:
        path = HERE / item["name"]
        require(path.is_file(), f"FA_PUBLIC_FILE_MISSING:{item['name']}")
        require(path.stat().st_size == item["bytes"], f"FA_PUBLIC_FILE_SIZE:{item['name']}")
        require(sha256_file(path) == item["sha256"], f"FA_PUBLIC_FILE_HASH:{item['name']}")
        checked += 1
    results = read_json(HERE / "FA_E01_RESULTS.json")
    receipt = read_json(HERE / "FA_E01_INDEPENDENT_RECOMPUTE.json")
    require(results["status"] == "PASS", "FA_PUBLIC_RESULTS_STATUS")
    require(results["eligible_run_count"] == 13, "FA_PUBLIC_ELIGIBLE_RUNS")
    require(results["verdict"] == "INCONCLUSIVE", "FA_PUBLIC_VERDICT")
    require(receipt["status"] == "PASS", "FA_PUBLIC_RECOMPUTE_STATUS")
    require(receipt["verdict"] == results["verdict"], "FA_PUBLIC_RECOMPUTE_VERDICT")
    require(receipt["results_sha256"] == sha256_file(HERE / "FA_E01_RESULTS.json"), "FA_PUBLIC_RESULTS_RECEIPT")
    require(receipt["fingerprints_sha256"] == sha256_file(HERE / "FA_E01_FINGERPRINTS.npz"), "FA_PUBLIC_FINGERPRINT_RECEIPT")
    return {
        "schema": "fa-e01-public-evidence-check-v1",
        "status": "PASS",
        "mode": "quick",
        "checked_manifest_files": checked,
        "eligible_run_count": results["eligible_run_count"],
        "verdict": results["verdict"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "FA_E01_PUBLIC_EVIDENCE_MANIFEST.json")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    result = quick_check(args.manifest.resolve())
    if args.deep:
        deep = independent_recompute(
            args.projection_root.resolve(),
            args.source_root.resolve(),
            (HERE / "FA_E01_RESULTS.json").resolve(),
            (HERE / "FA_E01_FINGERPRINTS.npz").resolve(),
        )
        result["mode"] = "deep"
        result["deep_recompute"] = deep
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
