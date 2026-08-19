from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.gfg_nanogpt_actual_update_boundary_v1.independent import check as check_tl_p01
from experiments.gfg_nanogpt_training_learning_inference_projection_v1.PUBLIC_EVIDENCE_CHECKER import (
    check as check_inf_e01,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_instrument_bundle(path: Path) -> dict[str, Any]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="verify-instruments-") as temporary:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                failures.append(f"zip_crc:{bad}")
            archive.extractall(temporary)
        root = Path(temporary) / "training_learning_instruments_evidence_v2"
        manifest = json.loads((root / "INSTRUMENT_ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            candidate = root / entry["relative_path"]
            if (
                not candidate.is_file()
                or candidate.stat().st_size != entry["bytes"]
                or sha256(candidate) != entry["sha256"]
            ):
                failures.append(entry["relative_path"])
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    manifest = json.loads((root / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    hash_failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            hash_failures.append(name)
    zip_failures = []
    for name in manifest["archives"]:
        with zipfile.ZipFile(root / name) as archive:
            bad = archive.testzip()
            if bad is not None:
                zip_failures.append(f"{name}:{bad}")

    with tempfile.TemporaryDirectory(prefix="verify-public-evidence-") as temporary:
        temporary_root = Path(temporary)
        with zipfile.ZipFile(root / "tl_p01_actual_update_boundary_evidence_v1.zip") as archive:
            archive.extractall(temporary_root / "tl")
        tl = check_tl_p01(temporary_root / "tl" / "tl_p01_actual_update_boundary_v1")
        with zipfile.ZipFile(root / "inf_e01_frozen_inference_gfg_evidence_v1.zip") as archive:
            archive.extractall(temporary_root / "inf")
        inf = check_inf_e01(temporary_root / "inf" / "inf_e01_frozen_inference_projection_v1")

    instruments = verify_instrument_bundle(root / "training_learning_instruments_evidence_v2.zip")
    inf_recomputed = dict(inf.get("recomputed", {}))
    inf_recomputed.pop("rows", None)
    inf_summary = {
        "status": inf["status"],
        "checks": inf["checks"],
        "integrity_failures": inf["integrity_failures"],
        "comparison_failures": inf["comparison_failures"],
        "recomputed": inf_recomputed,
    }
    checks = {
        "top_level_hashes": not hash_failures,
        "zip_crc": not zip_failures,
        "tl_p01": tl["status"] == "PASS",
        "inf_e01": inf["status"] == "PASS",
        "instrument_manifest": instruments["status"] == "PASS",
    }
    result = {
        "schema": "gfg-publication-evidence-v3-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hash_failures": hash_failures,
        "zip_failures": zip_failures,
        "tl_p01": tl,
        "inf_e01": inf_summary,
        "instruments": instruments,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
