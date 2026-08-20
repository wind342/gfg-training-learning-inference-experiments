from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.gfg_cross_system_frozen_inference_projection_v1.INDEPENDENT_CHECKER import (
    check as check_inf_g01,
)
from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.aggregate import (
    aggregate as aggregate_tl_g02,
)
from experiments.gfg_ddpm_cifar_training_learning_generalization_v1.INDEPENDENT_CHECKER import (
    _check_manifest as check_tl_g02_manifest,
    _check_native_records as check_tl_g02_records,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.aggregate import (
    aggregate as aggregate_tl_g01,
)
from experiments.gfg_resnet_cifar_training_learning_generalization_v1.INDEPENDENT_CHECKER import (
    check_manifest as check_tl_g01_manifest,
)


NEW_BUNDLES = {
    "tl_g01_resnet_cross_system_evidence_v1.zip": "tl_g01_resnet_cross_system_evidence_v1",
    "tl_g02_diffusion_cross_system_evidence_v1.zip": "tl_g02_diffusion_cross_system_evidence_v1",
    "inf_g01_cross_system_inference_evidence_v1.zip": "inf_g01_cross_system_inference_evidence_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_members(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], list[str]]:
    members = archive.infolist()
    failures: list[str] = []
    exact: set[str] = set()
    folded: set[str] = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            failures.append(f"UNSAFE_PATH:{name}")
        if name in exact:
            failures.append(f"DUPLICATE_PATH:{name}")
        if name.casefold() in folded:
            failures.append(f"CASE_COLLISION:{name}")
        exact.add(name)
        folded.add(name.casefold())
    return members, failures


def extract_checked(path: Path, destination: Path) -> list[str]:
    failures: list[str] = []
    with zipfile.ZipFile(path) as archive:
        _, path_failures = safe_members(archive)
        failures.extend(path_failures)
        bad = archive.testzip()
        if bad is not None:
            failures.append(f"ZIP_CRC:{path.name}:{bad}")
        if not failures:
            archive.extractall(destination)
    return failures


def verify_bundle_manifest(root: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = root / "BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        return [f"MISSING_BUNDLE_MANIFEST:{root}"]
    manifest = read_json(manifest_path)
    expected_paths = set()
    for entry in manifest["files"]:
        relative = entry["relative_path"]
        expected_paths.add(relative)
        path = root / relative
        if not path.is_file():
            failures.append(f"MISSING:{relative}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"SIZE:{relative}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"SHA256:{relative}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "BUNDLE_MANIFEST.json"
    }
    if actual_paths != expected_paths:
        failures.append("BUNDLE_FILE_SET_MISMATCH")
    if manifest.get("file_count") != len(expected_paths):
        failures.append("BUNDLE_FILE_COUNT_MISMATCH")
    return failures


def check_tl_g01(root: Path) -> dict[str, Any]:
    runtime = root / "runtime"
    errors = verify_bundle_manifest(root)
    errors.extend(check_tl_g01_manifest(runtime))
    recomputed = aggregate_tl_g01(runtime)
    if recomputed != read_json(runtime / "GENERALIZATION_RESULTS.json"):
        errors.append("RESULT_RECOMPUTATION_MISMATCH")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "recomputed_verdict": recomputed["verdict"],
    }


def check_tl_g02(root: Path) -> dict[str, Any]:
    runtime = root / "runtime"
    errors = verify_bundle_manifest(root)
    for path in sorted(runtime.glob("seed_*/MANIFEST.json")):
        errors.extend(check_tl_g02_manifest(path))
    errors.extend(check_tl_g02_manifest(runtime / "FINAL_MANIFEST.json"))
    errors.extend(check_tl_g02_records(runtime))
    recomputed = aggregate_tl_g02(runtime)
    if recomputed != read_json(runtime / "GENERALIZATION_RESULTS.json"):
        errors.append("RESULT_RECOMPUTATION_MISMATCH")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "recomputed_verdict": recomputed["verdict"],
        "response_count": recomputed["mechanism"]["response_count"],
    }


def check_checkpoint_authorities(
    inf_root: Path, extracted: dict[str, Path]
) -> dict[str, Any]:
    failures: list[str] = []
    authorities = read_json(inf_root / "CHECKPOINT_AUTHORITIES.json")
    checked = []
    for row in authorities["authorities"]:
        bundle = row["authority_bundle"]
        bundle_root_name = NEW_BUNDLES.get(bundle)
        if bundle_root_name is None or bundle not in extracted:
            failures.append(f"UNKNOWN_AUTHORITY_BUNDLE:{bundle}")
            continue
        member = PurePosixPath(row["authority_member"])
        expected_prefix = PurePosixPath(bundle_root_name)
        try:
            relative = member.relative_to(expected_prefix)
        except ValueError:
            failures.append(f"AUTHORITY_PREFIX:{row['authority_member']}")
            continue
        path = extracted[bundle] / relative.as_posix()
        passed = (
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and sha256(path) == row["sha256"]
            and row["identity_passed_in_native_execution"] is True
        )
        if not passed:
            failures.append(f"CHECKPOINT_AUTHORITY:{row['system']}:{row['seed']}")
        checked.append(
            {
                "system": row["system"],
                "seed": row["seed"],
                "sha256": row["sha256"],
                "pass": passed,
            }
        )
    if len(checked) != 6:
        failures.append(f"CHECKPOINT_AUTHORITY_COUNT:{len(checked)}")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "checked": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    manifest = read_json(root / "ARCHIVE_MANIFEST.json")

    hash_failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            hash_failures.append(name)

    zip_failures: list[str] = []
    for name in manifest["archives"]:
        path = root / name
        if not path.is_file():
            zip_failures.append(f"MISSING_ZIP:{name}")
            continue
        with zipfile.ZipFile(path) as archive:
            _, failures = safe_members(archive)
            zip_failures.extend(f"{name}:{failure}" for failure in failures)
            bad = archive.testzip()
            if bad is not None:
                zip_failures.append(f"{name}:ZIP_CRC:{bad}")

    with tempfile.TemporaryDirectory(prefix="verify-cross-system-v4-") as temporary:
        temporary_root = Path(temporary)
        extracted: dict[str, Path] = {}
        for name, archive_root_name in NEW_BUNDLES.items():
            destination = temporary_root / name.removesuffix(".zip")
            destination.mkdir()
            zip_failures.extend(extract_checked(root / name, destination))
            extracted[name] = destination / archive_root_name

        tl_g01 = check_tl_g01(extracted["tl_g01_resnet_cross_system_evidence_v1.zip"])
        tl_g02 = check_tl_g02(extracted["tl_g02_diffusion_cross_system_evidence_v1.zip"])
        inf_root = extracted["inf_g01_cross_system_inference_evidence_v1.zip"]
        inf_manifest_failures = verify_bundle_manifest(inf_root)
        inf = check_inf_g01(
            inf_root / "FORMAL_RESULTS.json",
            inf_root / "FORMAL_GFG.json",
            inf_root / "experiment/MODEL_CONTRACT.json",
        )
        inf["bundle_manifest_failures"] = inf_manifest_failures
        authorities = check_checkpoint_authorities(inf_root, extracted)

    checks = {
        "top_level_hashes": not hash_failures,
        "all_zip_paths_and_crc": not zip_failures,
        "tl_g01": tl_g01["status"] == "PASS",
        "tl_g02": tl_g02["status"] == "PASS",
        "inf_g01": inf["status"] == "PASS" and not inf_manifest_failures,
        "cross_bundle_checkpoint_authorities": authorities["status"] == "PASS",
    }
    result = {
        "schema": "gfg-publication-evidence-v4-cross-system-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hash_failures": hash_failures,
        "zip_failures": zip_failures,
        "tl_g01": tl_g01,
        "tl_g02": tl_g02,
        "inf_g01": inf,
        "checkpoint_authorities": authorities,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
