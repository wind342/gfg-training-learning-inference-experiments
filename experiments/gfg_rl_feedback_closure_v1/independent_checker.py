from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .evidence import file_sha256, validate_graph, validate_serialized_snapshot, write_json
from .runtime import object_sha256


PACKAGE = Path(__file__).parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_receipt(receipt: dict[str, Any], condition: str) -> None:
    episodes = receipt["episodes"]
    batch_size = len(episodes)
    require([row["episode_index"] for row in episodes] == list(range(batch_size)), "RL_EPISODE_INDEX_INVALID")
    actions = [row["actions"] for row in episodes]
    physical = [row["physical_consequences"] for row in episodes]
    assigned = [row["assigned_consequences"] for row in episodes]
    sources = [row["assigned_source_episode_indices"] for row in episodes]
    credits = [row["credited_to_action_stages"] for row in episodes]
    require(object_sha256(actions) == receipt["action_ledger_sha256"], "RL_ACTION_LEDGER_HASH_MISMATCH")
    require(object_sha256(physical) == receipt["physical_consequence_sha256"], "RL_PHYSICAL_HASH_MISMATCH")
    require(object_sha256(assigned) == receipt["assigned_consequence_sha256"], "RL_ASSIGNED_HASH_MISMATCH")
    require(object_sha256(sources) == receipt["binding_source_sha256"], "RL_BINDING_SOURCE_HASH_MISMATCH")
    require(object_sha256(credits) == receipt["credit_target_sha256"], "RL_CREDIT_TARGET_HASH_MISMATCH")
    if condition == "A":
        require(physical == assigned, "RL_A_REWARD_CHANGED")
        require(all(row == [index, index] for index, row in enumerate(sources)), "RL_A_BINDING_INVALID")
        require(all(row == [0, 1] for row in credits), "RL_A_CREDIT_INVALID")
    elif condition == "B":
        for stage in range(2):
            require(sorted(row[stage] for row in physical) == sorted(row[stage] for row in assigned), "RL_B_MULTISET_CHANGED")
            indices = [row[stage] for row in sources]
            require(sorted(indices) == list(range(batch_size)), "RL_B_SOURCE_NOT_PERMUTATION")
            require(all(assigned[index][stage] == physical[source][stage] for index, source in enumerate(indices)), "RL_B_SOURCE_VALUE_MISMATCH")
        require(all(row == [0, 1] for row in credits), "RL_B_CREDIT_INVALID")
    elif condition == "C":
        require(physical == assigned, "RL_C_REWARD_CHANGED")
        require(all(row == [index, index] for index, row in enumerate(sources)), "RL_C_BINDING_INVALID")
        require(all(row == [1, 0] for row in credits), "RL_C_CREDIT_NOT_SWAPPED")
    else:
        raise RuntimeError("RL_UNKNOWN_CONDITION")


def _verify_seed(seed_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    seed = read_json(seed_path)
    conditions = seed["conditions"]
    initial = {conditions[name]["initial_state_sha256"] for name in ("A", "B", "C")}
    require(len(initial) == 1 and next(iter(initial)) == seed["clone_state_sha256"], "RL_CLONE_IDENTITY_FAILED")
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for condition in ("A", "B", "C"):
        result = conditions[condition]
        ledger_path = Path(result["ledger"])
        require(file_sha256(ledger_path) == result["ledger_sha256"], "RL_LEDGER_FILE_HASH_MISMATCH")
        rows = read_jsonl(ledger_path)
        require(len(rows) == int(contract["formal"]["reversal_updates"]), "RL_UPDATE_COUNT_MISMATCH")
        previous = result["initial_state_sha256"]
        for expected_update, receipt in enumerate(rows):
            require(receipt["update"] == expected_update, "RL_UPDATE_SEQUENCE_INVALID")
            require(receipt["condition"] == condition, "RL_RECEIPT_CONDITION_MISMATCH")
            require(receipt["pre_state_sha256"] == previous, "RL_PARAMETER_STATE_CHAIN_BROKEN")
            _verify_receipt(receipt, condition)
            previous = receipt["post_state_sha256"]
        require(previous == result["final_state_sha256"], "RL_FINAL_STATE_MISMATCH")
        require(rows[-1]["post_chain_accuracy"] == result["final_evaluation"]["chain_accuracy"], "RL_FINAL_METRIC_MISMATCH")
        require(abs(mean(row["chain_accuracy"] for row in result["curve"]) - result["auc"]) < 1e-15, "RL_AUC_MISMATCH")
        ledgers[condition] = rows
    for update in range(len(ledgers["A"])):
        cue_hashes = {ledgers[name][update]["cue_batch_sha256"] for name in ("A", "B", "C")}
        uniform_hashes = {ledgers[name][update]["uniform_batch_sha256"] for name in ("A", "B", "C")}
        require(len(cue_hashes) == 1, "RL_CONDITION_CUE_BATCH_DIFFERED")
        require(len(uniform_hashes) == 1, "RL_CONDITION_RANDOM_STREAM_DIFFERED")
    fork_path = Path(seed["forks"])
    require(file_sha256(fork_path) == seed["forks_sha256"], "RL_FORK_FILE_HASH_MISMATCH")
    forks = read_json(fork_path)
    for fork in forks:
        values = fork["conditions"]
        require({row["pre_state_sha256"] for row in values.values()} == {fork["common_receiving_state_sha256"]}, "RL_FORK_RECEIVING_STATE_DIFFERED")
        require(len({row["cue_batch_sha256"] for row in values.values()}) == 1, "RL_FORK_CUE_DIFFERED")
        require(len({row["uniform_batch_sha256"] for row in values.values()}) == 1, "RL_FORK_UNIFORM_DIFFERED")
    return {
        "seed": seed["seed"],
        "status": "PASS",
        "updates_per_condition": len(ledgers["A"]),
        "fork_count": len(forks),
        "final_chain_accuracy": {
            name: conditions[name]["final_evaluation"]["chain_accuracy"] for name in ("A", "B", "C")
        },
    }


def check(aggregate_path: Path, evidence_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    contract = read_json(PACKAGE / "EXPERIMENT_CONTRACT.json")
    freeze = read_json(PACKAGE / "CONTRACT_FREEZE.json")
    require(file_sha256(PACKAGE / "EXPERIMENT_CONTRACT.json") == freeze["experiment_contract_sha256"], "RL_FROZEN_CONTRACT_CHANGED")
    require(file_sha256(PACKAGE / "PROTOCOL_FREEZE.md") == freeze["protocol_sha256"], "RL_FROZEN_PROTOCOL_CHANGED")
    aggregate = read_json(aggregate_path)
    require(aggregate["mode"] == "formal", "RL_NOT_FORMAL_RESULT")
    require(aggregate["contract_sha256"] == freeze["experiment_contract_sha256"], "RL_FORMAL_CONTRACT_HASH_MISMATCH")
    require(aggregate["protocol_sha256"] == freeze["protocol_sha256"], "RL_FORMAL_PROTOCOL_HASH_MISMATCH")
    seed_results = [_verify_seed(Path(path), contract) for path in aggregate["seed_result_paths"]]
    require(len(seed_results) == len(contract["formal"]["seeds"]), "RL_FORMAL_SEED_COUNT_MISMATCH")
    evidence = read_json(evidence_manifest_path)
    require(evidence["status"] == "PASS", "RL_EVIDENCE_MANIFEST_FAILED")
    require(evidence["aggregate_result_sha256"] == file_sha256(aggregate_path), "RL_EVIDENCE_AGGREGATE_HASH_MISMATCH")
    evidence_results = []
    for entry in evidence["entries"]:
        snapshot_path = Path(entry["snapshot"])
        graph_path = Path(entry["graph"])
        require(file_sha256(snapshot_path) == entry["snapshot_sha256"], "RL_SNAPSHOT_FILE_HASH_MISMATCH")
        require(file_sha256(graph_path) == entry["graph_file_sha256"], "RL_GRAPH_FILE_HASH_MISMATCH")
        snapshot_id = validate_serialized_snapshot(read_json(snapshot_path))
        graph_validation = validate_graph(read_json(graph_path))
        require(snapshot_id == entry["snapshot_id"], "RL_SNAPSHOT_ID_MISMATCH")
        require(graph_validation["status"] == "PASS", "RL_GRAPH_VALIDATION_FAILED")
        evidence_results.append({"seed": entry["seed"], "snapshot_id": snapshot_id, **graph_validation})
    gates = aggregate["scientific_gates"]
    result = {
        "schema": "rl-feedback-closure-independent-check-v1",
        "status": "PASS",
        "scientific_status": "PASS" if all(gates.values()) else "FAIL",
        "scientific_gates": gates,
        "aggregate_sha256": file_sha256(aggregate_path),
        "evidence_manifest_sha256": file_sha256(evidence_manifest_path),
        "seed_results": seed_results,
        "evidence_results": evidence_results,
        "checks": {
            "frozen_contract": "PASS",
            "initial_clone_identity": "PASS",
            "same_batches_and_random_streams": "PASS",
            "reward_binding_invariants": "PASS",
            "temporal_credit_invariants": "PASS",
            "parameter_state_chains": "PASS",
            "metric_recomputation": "PASS",
            "core_v3_validation": "PASS",
            "gfg_validation": "PASS",
        },
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.aggregate.resolve(), args.evidence_manifest.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
