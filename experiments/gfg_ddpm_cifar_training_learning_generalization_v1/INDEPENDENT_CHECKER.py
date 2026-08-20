from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .aggregate import aggregate
from .numeric import sha256_file


EXPECTED_SEEDS = (20260830, 20260831, 20260832)
EXPECTED_EVENTS_PER_RUN = 7
EXPECTED_TARGETS_PER_EVENT = 24
EXPECTED_RESPONSE_COUNT = (
    len(EXPECTED_SEEDS) * EXPECTED_EVENTS_PER_RUN * EXPECTED_TARGETS_PER_EVENT
)
EXPECTED_FEATURE_SPANS = {"F1": [0, 6], "F3": [6, 14], "F5": [14, 39]}
ALLOWED_FEATURE_PREFIXES = ("F1_", "F3_", "F5_")
FORBIDDEN_FEATURE_FRAGMENTS = (
    "post_",
    "future",
    "run_id",
    "absolute_step",
    "alpha_",
    "response_path",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    manifest = _load_json(path)
    base = path.parent
    for name, expected in manifest.get("files", {}).items():
        target = base / name
        if not target.is_file():
            errors.append(f"MISSING:{target}")
            continue
        if target.stat().st_size != expected["bytes"]:
            errors.append(f"SIZE:{target}")
        if sha256_file(target) != expected["sha256"]:
            errors.append(f"SHA256:{target}")
    return errors


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "MAINTAIN_CORRECT"
    if before and not after:
        return "CORRECT_TO_WRONG"
    if not before and after:
        return "WRONG_TO_CORRECT"
    return "MAINTAIN_WRONG"


def _check_record(row: dict[str, Any], location: str) -> list[str]:
    errors: list[str] = []
    names = row.get("feature_names", [])
    values = row.get("features", [])
    if len(names) != 39 or len(values) != 39:
        errors.append(f"FEATURE_LENGTH:{location}")
    if row.get("feature_families") != EXPECTED_FEATURE_SPANS:
        errors.append(f"FEATURE_SPANS:{location}")
    if any(not name.startswith(ALLOWED_FEATURE_PREFIXES) for name in names):
        errors.append(f"FEATURE_FAMILY_NAME:{location}")
    lowered = [name.lower() for name in names]
    if any(fragment in name for name in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
        errors.append(f"FORBIDDEN_FEATURE_NAME:{location}")
    if any(not math.isfinite(float(value)) for value in values):
        errors.append(f"NONFINITE_FEATURE:{location}")
    pre_margin = float(row["pre_margin"])
    post_margin = float(row["post_margin"])
    pre_correct = bool(row["pre_correct"])
    post_correct = bool(row["post_correct"])
    if pre_correct != (pre_margin >= 0.0):
        errors.append(f"PRE_BOUNDARY:{location}")
    if post_correct != (post_margin >= 0.0):
        errors.append(f"POST_BOUNDARY:{location}")
    if row.get("transition") != _transition(pre_correct, post_correct):
        errors.append(f"TRANSITION:{location}")
    path = row.get("margin_path", [])
    if len(path) != 6:
        errors.append(f"PATH_LENGTH:{location}")
    elif abs(float(path[0]) - pre_margin) > 1e-10 or abs(float(path[-1]) - post_margin) > 1e-10:
        errors.append(f"PATH_ENDPOINT:{location}")
    for name, margin in row.get("predicted_margins", {}).items():
        if bool(row["predictions"][name]) != (float(margin) >= 0.0):
            errors.append(f"PREDICTED_BOUNDARY:{name}:{location}")
    if bool(row["predictions"]["unchanged"]) != pre_correct:
        errors.append(f"UNCHANGED_BOUNDARY:{location}")
    return errors


def _check_native_records(root: Path) -> list[str]:
    errors: list[str] = []
    seen_seeds: list[int] = []
    response_count = 0
    identities: set[tuple[int, int, int]] = set()
    for seed in EXPECTED_SEEDS:
        run = root / f"seed_{seed}"
        if not run.is_dir():
            errors.append(f"MISSING_RUN:{seed}")
            continue
        seen_seeds.append(seed)
        summary = _load_json(run / "RUN_SUMMARY.json")
        if summary.get("seed") != seed or summary.get("smoke") is not False:
            errors.append(f"RUN_IDENTITY:{seed}")
        if summary.get("event_count") != EXPECTED_EVENTS_PER_RUN:
            errors.append(f"EVENT_COUNT:{seed}")
        if summary.get("exchange_count") != EXPECTED_EVENTS_PER_RUN - 1:
            errors.append(f"EXCHANGE_COUNT:{seed}")
        if not summary.get("integrity_pass"):
            errors.append(f"RUN_INTEGRITY:{seed}")
        events = _load_json(run / "EVENTS.json")
        if len(events) != EXPECTED_EVENTS_PER_RUN:
            errors.append(f"EVENT_FILE_COUNT:{seed}")
        for event in events:
            event_index = int(event["event_index"])
            rows = event["analysis"]["records"]
            if len(rows) != EXPECTED_TARGETS_PER_EVENT:
                errors.append(f"TARGET_COUNT:{seed}:{event_index}")
            for row_index, row in enumerate(rows):
                location = f"{seed}:{event_index}:{row_index}"
                errors.extend(_check_record(row, location))
                identity = (seed, event_index, int(row["target_identity"]))
                if identity in identities:
                    errors.append(f"DUPLICATE_TARGET_IDENTITY:{location}")
                identities.add(identity)
                response_count += 1
    if tuple(seen_seeds) != EXPECTED_SEEDS:
        errors.append("SEED_SET")
    if response_count != EXPECTED_RESPONSE_COUNT:
        errors.append(f"RESPONSE_COUNT:{response_count}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    for path in sorted(root.glob("seed_*/MANIFEST.json")):
        errors.extend(_check_manifest(path))
    final_manifest = root / "FINAL_MANIFEST.json"
    if final_manifest.is_file():
        errors.extend(_check_manifest(final_manifest))
    else:
        errors.append("MISSING_FINAL_MANIFEST")
    errors.extend(_check_native_records(root))
    recomputed = aggregate(root)
    recorded_path = root / "GENERALIZATION_RESULTS.json"
    if not recorded_path.is_file():
        errors.append("MISSING_GENERALIZATION_RESULTS")
    elif recomputed != _load_json(recorded_path):
        errors.append("RESULT_RECOMPUTATION_MISMATCH")
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    cap_bytes = 25 * 1024**3
    if total_bytes > cap_bytes:
        errors.append(f"STORAGE_CAP:{total_bytes}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "recomputed_verdict": recomputed["verdict"],
        "response_count": recomputed["mechanism"]["response_count"],
        "formal_runtime_bytes": total_bytes,
        "storage_cap_bytes": cap_bytes,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
