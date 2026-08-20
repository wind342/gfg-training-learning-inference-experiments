from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


DOSE_NAMES = ("balanced", "mild", "high", "exclusive")
RECOVERY_NAMES = ("rebalance_recovery", "repair_recovery")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(condition: dict[str, Any], update: int) -> dict[str, Any]:
    matches = [row for row in condition["support_measurements"] if int(row["phase_update"]) == update]
    if len(matches) != 1:
        raise RuntimeError(f"EXPECTED_ONE_CHECKPOINT:{condition['condition']}:{update}:{len(matches)}")
    return matches[0]


def compact_point(row: dict[str, Any], update_key: str) -> dict[str, Any]:
    per_skill = row["evaluation"]["per_skill"]
    return {
        update_key: int(row["phase_update"]),
        "task0_support_share": float(row["support"]["task_support_shares"][0]),
        "unreinforced_accuracy": mean(float(skill["chain_accuracy"]) for skill in per_skill[1:]),
        "unreinforced_mean_margin": mean(float(skill["mean_margin"]) for skill in per_skill[1:]),
        "task0_accuracy": float(per_skill[0]["chain_accuracy"]),
    }


def mean_points(rows: list[dict[str, Any]], update_key: str) -> dict[str, Any]:
    return {
        update_key: rows[0][update_key],
        "task0_support_share": mean(row["task0_support_share"] for row in rows),
        "unreinforced_accuracy": mean(row["unreinforced_accuracy"] for row in rows),
        "unreinforced_mean_margin": mean(row["unreinforced_mean_margin"] for row in rows),
        "task0_accuracy": mean(row["task0_accuracy"] for row in rows),
    }


def compile_results(formal_root: Path, independent_summary: Path | None) -> dict[str, Any]:
    formal = read_json(formal_root / "FORMAL_RESULT.json")
    seeds = [read_json(formal_root / f"seed-{row['seed']}" / "SEED_RESULT.json") for row in formal["per_seed"]]
    endpoint = {}
    doses = {"balanced": 0.25, "mild": 0.5, "high": 0.75, "exclusive": 1.0}
    for name in DOSE_NAMES:
        points = [compact_point(checkpoint(seed["dose_conditions"][name], 3200), "update") for seed in seeds]
        endpoint[name] = {"dose": doses[name], **mean_points(points, "update")}
    exclusive_duration = []
    for update in (100, 400, 800, 1600, 3200):
        points = [compact_point(checkpoint(seed["dose_conditions"]["exclusive"], update), "update") for seed in seeds]
        exclusive_duration.append(mean_points(points, "update"))
    recovery = {}
    for name in RECOVERY_NAMES:
        recovery[name] = []
        for update in (0, 100, 400, 800, 1600, 2400):
            points = [compact_point(checkpoint(seed["recovery_conditions"][name], update), "recovery_update") for seed in seeds]
            recovery[name].append(mean_points(points, "recovery_update"))
    output = {
        "schema": "rl-e06-compiled-results-v1",
        "formal_result_sha256": sha256(formal_root / "FORMAL_RESULT.json"),
        "scientific_status": formal["scientific_status"],
        "seed_count": formal["seed_count"],
        "counts": formal["counts"],
        "means": formal["means"],
        "decision_gates": formal["decision_gates"],
        "dose_endpoint_means": endpoint,
        "exclusive_duration_means": exclusive_duration,
        "recovery_means": recovery,
    }
    if independent_summary is not None:
        independent = read_json(independent_summary)
        output["independent_check"] = {
            "sha256": sha256(independent_summary),
            "status": independent["status"],
            "full_native_replays": independent["full_native_replays"],
            "independent_recalculation": independent["independent_recalculation"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--independent-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output.resolve(),
        compile_results(
            args.formal_root.resolve(),
            args.independent_summary.resolve() if args.independent_summary else None,
        ),
    )


if __name__ == "__main__":
    main()
