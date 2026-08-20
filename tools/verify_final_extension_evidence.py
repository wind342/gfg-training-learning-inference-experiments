from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.gfg_rl_selective_positive_feedback_dose_recovery_v1.independent_checker import (
    _checkpoint_state,
    _recompute_aggregate,
    _skill_mean,
)
from experiments.gfg_rl_selective_positive_feedback_dose_recovery_v1.runner import (
    DOSE_CONDITIONS,
    RECOVERY_CONDITIONS,
)
from experiments.gfg_rl_selective_positive_feedback_dose_recovery_v1.runtime import (
    combined_state_sha256,
    evaluate_policy,
    support_profile,
)
from experiments.gfg_rl_selective_positive_feedback_support_concentration_v1.independent_checker import (
    run as check_rl_e05,
)
from tools.verify_cross_system_evidence_v4 import (
    extract_checked,
    read_json,
    sha256,
    verify_bundle_manifest,
)


RL_BUNDLES = {
    "rl_e05_selective_feedback_evidence_v1.zip": "rl_e05_selective_feedback_evidence_v1",
    "rl_e06_dose_recovery_evidence_v1.zip": "rl_e06_dose_recovery_evidence_v1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_source_manifest(package: Path) -> list[str]:
    failures: list[str] = []
    manifest = read_json(package / "ARTIFACT_MANIFEST.json")
    for name, expected in manifest["files"].items():
        path = package / name
        if not path.is_file():
            failures.append(f"MISSING_SOURCE:{name}")
        elif path.stat().st_size != expected["bytes"]:
            failures.append(f"SOURCE_SIZE:{name}")
        elif sha256(path) != expected["sha256"]:
            failures.append(f"SOURCE_SHA256:{name}")
    return failures


def check_rl_e06_condition(
    *,
    seed_dir: Path,
    name: str,
    recorded: dict[str, Any],
    expected_updates: int,
    expected_batch_size: int,
    contract: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    receipts_path = seed_dir / f"{name}-feedback-receipts.jsonl"
    trajectory_path = seed_dir / f"{name}-boundary-trajectory.jsonl"
    checkpoint_path = seed_dir / f"{name}-final.pt"
    require(sha256(receipts_path) == recorded["feedback_receipts_sha256"], f"RECEIPT_SHA256:{seed_dir.name}:{name}")
    require(sha256(trajectory_path) == recorded["boundary_trajectory_sha256"], f"TRAJECTORY_SHA256:{seed_dir.name}:{name}")
    require(sha256(checkpoint_path) == recorded["final_checkpoint"]["sha256"], f"CHECKPOINT_SHA256:{seed_dir.name}:{name}")
    receipts = read_jsonl(receipts_path)
    trajectory = read_jsonl(trajectory_path)
    require(len(receipts) == expected_updates, f"RECEIPT_COUNT:{seed_dir.name}:{name}")
    require(len(trajectory) == expected_updates + 1, f"TRAJECTORY_COUNT:{seed_dir.name}:{name}")
    require(sum(int(row["episode_count"]) for row in receipts) == expected_updates * expected_batch_size, f"EPISODE_COUNT:{seed_dir.name}:{name}")
    require(all(row["reward_authority"] == "exact_complete_chain_rule" for row in receipts), f"REWARD_AUTHORITY:{seed_dir.name}:{name}")
    model, optimizer, _ = _checkpoint_state(checkpoint_path, contract, device)
    require(combined_state_sha256(model, optimizer) == recorded["final_state_sha256"], f"STATE_SHA256:{seed_dir.name}:{name}")
    require(evaluate_policy(model, device) == recorded["final_evaluation"], f"FINAL_EVALUATION:{seed_dir.name}:{name}")
    require(support_profile(model, device) == recorded["final_support"], f"FINAL_SUPPORT:{seed_dir.name}:{name}")
    return receipts


def check_rl_e06(root: Path) -> dict[str, Any]:
    failures = verify_bundle_manifest(root)
    failures.extend(verify_source_manifest(root / "experiment"))
    if failures:
        return {"status": "FAIL", "failures": failures}
    formal = root / "formal"
    package = root / "experiment"
    source_manifest = read_json(package / "ARTIFACT_MANIFEST.json")
    require(sha256(formal / "FORMAL_RESULT.json") == source_manifest["formal_result_sha256"], "FORMAL_RESULT_AUTHORITY")
    contract = read_json(package / "MODEL_CONTRACT.json")
    formal_result = read_json(formal / "FORMAL_RESULT.json")
    device = torch.device("cpu")
    seeds: list[dict[str, Any]] = []
    checked_conditions = 0
    checked_gfg_facts = 0
    for seed in [int(value) for value in contract["formal_seeds"]]:
        seed_dir = formal / f"seed-{seed}"
        stored = read_json(seed_dir / "SEED_RESULT.json")
        seeds.append(stored)
        receipts: dict[str, list[dict[str, Any]]] = {}
        for name in DOSE_CONDITIONS:
            receipts[name] = check_rl_e06_condition(
                seed_dir=seed_dir,
                name=name,
                recorded=stored["dose_conditions"][name],
                expected_updates=int(contract["feedback"]["updates"]),
                expected_batch_size=int(contract["feedback"]["batch_size"]),
                contract=contract,
                device=device,
            )
            checked_conditions += 1
        for name in RECOVERY_CONDITIONS:
            receipts[name] = check_rl_e06_condition(
                seed_dir=seed_dir,
                name=name,
                recorded=stored["recovery_conditions"][name],
                expected_updates=int(contract["recovery"]["updates"]),
                expected_batch_size=int(contract["feedback"]["batch_size"]),
                contract=contract,
                device=device,
            )
            checked_conditions += 1
        for update in range(int(contract["feedback"]["updates"])):
            require(len({receipts[name][update]["cue_batch_sha256"] for name in DOSE_CONDITIONS}) == 1, f"DOSE_CUE_LEDGER:{seed}:{update}")
            require(len({receipts[name][update]["uniform_batch_sha256"] for name in DOSE_CONDITIONS}) == 1, f"DOSE_UNIFORM_LEDGER:{seed}:{update}")
        for update in range(int(contract["recovery"]["updates"])):
            require(receipts["rebalance_recovery"][update]["cue_batch_sha256"] == receipts["repair_recovery"][update]["cue_batch_sha256"], f"RECOVERY_CUE_LEDGER:{seed}:{update}")
            require(receipts["rebalance_recovery"][update]["uniform_batch_sha256"] == receipts["repair_recovery"][update]["uniform_batch_sha256"], f"RECOVERY_UNIFORM_LEDGER:{seed}:{update}")
        gfg = read_json(seed_dir / "EXPERIMENT_GFG.json")
        require(gfg["validation"]["all_facts_realized_once"], f"GFG_REALIZATION:{seed}")
        require(gfg["validation"]["all_five_coordinates_present"], f"GFG_COORDINATES:{seed}")
        checked_gfg_facts += int(gfg["validation"]["fact_count"])

    recomputed = _recompute_aggregate(seeds, contract)
    require(all(recomputed["gates_independently_pass"].values()), "RECOMPUTED_GATE_FAILURE")
    require(math.isclose(recomputed["mean_exclusive_accuracy_deficit"], formal_result["means"]["exclusive_unreinforced_accuracy_deficit_vs_balanced"], rel_tol=0.0, abs_tol=1e-10), "ACCURACY_MEAN")
    require(math.isclose(recomputed["mean_exclusive_support_excess"], formal_result["means"]["exclusive_task0_support_share_excess_vs_balanced"], rel_tol=0.0, abs_tol=1e-10), "SUPPORT_MEAN")
    require(math.isclose(recomputed["mean_rebalance_accuracy_gain"], formal_result["means"]["rebalance_unreinforced_accuracy_gain_vs_exclusive"], rel_tol=0.0, abs_tol=1e-10), "RECOVERY_COUNTERFACTUAL_MEAN")
    common_fork = sum(
        _skill_mean(
            read_jsonl(formal / f"seed-{seed['seed']}" / "rebalance_recovery-boundary-trajectory.jsonl")[0]["evaluation"],
            "chain_accuracy",
            [1, 2, 3],
        )
        for seed in seeds
    ) / len(seeds)
    rebalance_endpoint = sum(
        _skill_mean(seed["recovery_conditions"]["rebalance_recovery"]["final_evaluation"], "chain_accuracy", [1, 2, 3])
        for seed in seeds
    ) / len(seeds)
    require(math.isclose(rebalance_endpoint - common_fork, 0.29166666666666663, rel_tol=0.0, abs_tol=1e-10), "RECOVERY_FORK_REFERENCE")
    return {
        "status": "PASS",
        "failures": [],
        "seed_count": len(seeds),
        "checked_conditions": checked_conditions,
        "checked_gfg_facts": checked_gfg_facts,
        "recomputed": recomputed,
        "rebalance_gain_from_common_update_800_fork": rebalance_endpoint - common_fork,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()

    prior = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tools/verify_cross_system_evidence_v4.py"), str(root)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if prior.returncode != 0:
        print(prior.stdout)
        print(prior.stderr, file=sys.stderr)
        raise SystemExit(prior.returncode)
    cross_system = json.loads(prior.stdout)

    with tempfile.TemporaryDirectory(prefix="verify-final-extension-") as temporary:
        temporary_root = Path(temporary)
        extracted: dict[str, Path] = {}
        extraction_failures: list[str] = []
        for name, root_name in RL_BUNDLES.items():
            destination = temporary_root / name.removesuffix(".zip")
            destination.mkdir()
            extraction_failures.extend(extract_checked(root / name, destination))
            extracted[name] = destination / root_name
        require(not extraction_failures, f"RL_BUNDLE_EXTRACTION:{extraction_failures}")
        e05_root = extracted["rl_e05_selective_feedback_evidence_v1.zip"]
        e05_manifest_failures = verify_bundle_manifest(e05_root)
        e05_manifest_failures.extend(verify_source_manifest(e05_root / "experiment"))
        require(not e05_manifest_failures, f"RL_E05_MANIFEST:{e05_manifest_failures}")
        e05 = check_rl_e05(e05_root / "formal", "cpu")
        e06 = check_rl_e06(extracted["rl_e06_dose_recovery_evidence_v1.zip"])

    checks = {
        "carried_forward_and_cross_system": cross_system["status"] == "PASS",
        "rl_e05": e05["status"] == "PASS",
        "rl_e06": e06["status"] == "PASS",
    }
    result = {
        "schema": "gfg-publication-evidence-final-extension-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cross_system": cross_system,
        "rl_e05": e05,
        "rl_e06": e06,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
