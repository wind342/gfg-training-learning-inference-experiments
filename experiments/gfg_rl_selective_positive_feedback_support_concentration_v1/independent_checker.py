from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .runner import PACKAGE, aggregate, file_sha256, read_json, write_json
from .runtime import (
    MultiSkillGRUPolicy,
    combined_state_sha256,
    evaluate_policy,
    make_optimizer,
    rollback_components_state,
    support_profile,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: Any, right: Any, tolerance: float = 1e-7, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), f"KEY_MISMATCH:{path}")
        for key in left:
            close(left[key], right[key], tolerance, f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), f"LENGTH_MISMATCH:{path}")
        for index, (a, b) in enumerate(zip(left, right)):
            close(a, b, tolerance, f"{path}[{index}]")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        require(abs(float(left) - float(right)) <= tolerance, f"VALUE_MISMATCH:{path}:{left}:{right}")
    else:
        require(left == right, f"VALUE_MISMATCH:{path}:{left}:{right}")


def load_checkpoint(
    path: Path,
    contract: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], MultiSkillGRUPolicy, torch.optim.AdamW]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["state"]
    model = MultiSkillGRUPolicy(int(contract["model"]["hidden_size"])).to(device)
    model.load_state_dict(state["model"])
    optimizer = make_optimizer(
        model,
        float(contract["model"]["learning_rate"]),
        float(contract["model"]["weight_decay"]),
    )
    optimizer.load_state_dict(state["optimizer"])
    require(combined_state_sha256(model, optimizer) == state["combined_state_sha256"], f"STATE_HASH_MISMATCH:{path}")
    return payload, model, optimizer


def check_gfg(path: Path, expected_seed: int, expected_updates: int) -> dict[str, Any]:
    gfg = read_json(path)
    require(gfg["seed"] == expected_seed, f"GFG_SEED_MISMATCH:{expected_seed}")
    facts = gfg["atomic_generation_facts"]
    occurrences = gfg["concrete_occurrences"]
    require(len(facts) == len(occurrences), f"GFG_INCIDENCE_COUNT_MISMATCH:{expected_seed}")
    fact_ids = {row["fact_id"] for row in facts}
    realized = [identifier for row in occurrences for identifier in row["realizes_fact"]]
    require(len(realized) == len(set(realized)), f"GFG_DUPLICATE_REALIZATION:{expected_seed}")
    require(set(realized) == fact_ids, f"GFG_REALIZATION_COVERAGE:{expected_seed}")
    require(all(all(row.get(key) not in (None, "") for key in ("u", "tau", "omega", "z", "rho")) for row in facts), f"GFG_COORDINATE_MISSING:{expected_seed}")
    update_facts = [row for row in facts if ":feedback-update-" in row["omega"]]
    require(len(update_facts) == 3 * expected_updates, f"GFG_UPDATE_COUNT:{expected_seed}:{len(update_facts)}")
    return {"fact_count": len(facts), "update_fact_count": len(update_facts), "sha256": file_sha256(path)}


def check_receipts(seed_dir: Path, expected_updates: int) -> None:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in ("selective", "balanced", "frozen"):
        path = seed_dir / f"{condition}-feedback-receipts.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        require(len(rows) == expected_updates, f"RECEIPT_COUNT:{seed_dir.name}:{condition}")
        require(all(row["reward_authority"] == "exact_complete_chain_rule" for row in rows), f"REWARD_AUTHORITY:{seed_dir.name}:{condition}")
        by_condition[condition] = rows
    for selective, frozen in zip(by_condition["selective"], by_condition["frozen"]):
        require(selective["cue_batch_sha256"] == frozen["cue_batch_sha256"], f"SELECTIVE_FROZEN_CUE_MISMATCH:{seed_dir.name}")
        require(selective["uniform_batch_sha256"] == frozen["uniform_batch_sha256"], f"SELECTIVE_FROZEN_UNIFORM_MISMATCH:{seed_dir.name}")
        require(frozen["pre_state_sha256"] == frozen["post_state_sha256"], f"FROZEN_UPDATE_OCCURRED:{seed_dir.name}")


def check_seed(seed: int, artifact_root: Path, contract: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_dir = artifact_root / f"seed-{seed}"
    summary = read_json(seed_dir / "SEED_RESULT.json")
    require(summary["seed"] == seed, f"SEED_RESULT_ID:{seed}")
    _, baseline_model, baseline_optimizer = load_checkpoint(seed_dir / "baseline-seal.pt", contract, device)
    close(evaluate_policy(baseline_model, device), summary["baseline"]["evaluation"])
    close(support_profile(baseline_model, device), summary["baseline"]["support"])
    require(combined_state_sha256(baseline_model, baseline_optimizer) == summary["baseline"]["state_sha256"], f"BASELINE_HASH:{seed}")
    final_models: dict[str, MultiSkillGRUPolicy] = {}
    for condition in ("selective", "balanced", "frozen"):
        _, model, optimizer = load_checkpoint(seed_dir / f"{condition}-final.pt", contract, device)
        final_models[condition] = model
        result = summary["conditions"][condition]
        require(combined_state_sha256(model, optimizer) == result["final_state_sha256"], f"FINAL_HASH:{seed}:{condition}")
        close(evaluate_policy(model, device), result["final_evaluation"])
        close(support_profile(model, device), result["final_support"])
    require(summary["conditions"]["frozen"]["final_state_sha256"] == summary["baseline"]["state_sha256"], f"FROZEN_NOT_EXACT:{seed}")
    masks = set()
    baseline_state = baseline_model.state_dict()
    for row in summary["rollbacks"]["rows"]:
        mask = int(row["mask"])
        masks.add(mask)
        rollback = rollback_components_state(final_models["selective"], baseline_state, row["components"])
        close(evaluate_policy(rollback, device), row["evaluation"])
        close(support_profile(rollback, device), row["support"])
    require(masks == set(range(1, 16)), f"ROLLBACK_MASK_COVERAGE:{seed}")
    check_receipts(seed_dir, int(contract["feedback"]["updates"]))
    gfg_result = check_gfg(seed_dir / "EXPERIMENT_GFG.json", seed, int(contract["feedback"]["updates"]))
    return summary, gfg_result


def run(artifact_root: Path, device_name: str) -> dict[str, Any]:
    contract = read_json(PACKAGE / "MODEL_CONTRACT.json")
    freeze = read_json(PACKAGE / "CONTRACT_FREEZE.json")
    for name, expected in freeze["files"].items():
        require(file_sha256(PACKAGE / name) == expected, f"FROZEN_SOURCE_CHANGED:{name}")
    device = torch.device(device_name)
    summaries = []
    gfg_rows = []
    for seed in contract["formal_seeds"]:
        summary, gfg_result = check_seed(int(seed), artifact_root, contract, device)
        summaries.append(summary)
        gfg_rows.append({"seed": int(seed), **gfg_result})
    recomputed = aggregate(summaries, contract)
    recorded = read_json(artifact_root / "FORMAL_RESULT.json")
    close(recomputed, {key: recorded[key] for key in recomputed})
    report = {
        "schema": "rl-e05-independent-check-v1",
        "status": "PASS",
        "seed_count": len(summaries),
        "formal_result_sha256": file_sha256(artifact_root / "FORMAL_RESULT.json"),
        "scientific_status": recomputed["scientific_status"],
        "decision_gates": recomputed["decision_gates"],
        "gfgs": gfg_rows,
    }
    write_json(artifact_root / "INDEPENDENT_CHECK.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_root.resolve(), args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
