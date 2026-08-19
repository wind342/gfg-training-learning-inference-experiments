from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runner import aggregate, file_sha256, read_json, require, write_json
from .runtime import (
    CreditPolicy,
    HORIZON,
    METHODS,
    build_base_gfg,
    deterministic_behavior_actions,
    evaluate_credit_policy,
    execute_episode,
    fit_trace_decomposition,
    make_episode_spec,
    terminal_consequence_only,
    validate_base_gfg,
)


PACKAGE = Path(__file__).parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _same_float_lists(left: list[float], right: list[float], tolerance: float = 1e-12) -> bool:
    return len(left) == len(right) and bool(np.allclose(left, right, rtol=0.0, atol=tolerance))


def check_seed(seed_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    seed = read_json(seed_path)
    seed_id = int(seed["seed"])
    mode = seed["mode"]
    config = contract[mode]
    ranges = seed["dataset_ranges"]
    flattened = []
    for low, high in ranges.values():
        require(low <= high, "TCD_DATASET_RANGE_INVALID")
        flattened.extend(range(int(low), int(high) + 1))
    require(len(flattened) == len(set(flattened)), "TCD_DATASET_SPLIT_OVERLAP")

    trace_path = Path(seed["trace_decomposition_model"])
    require(file_sha256(trace_path) == seed["trace_decomposition_model_sha256"], "TCD_TRACE_MODEL_HASH_MISMATCH")
    trace_payload = read_json(trace_path)
    trace_rows = []
    for episode_id in range(ranges["trace_fit"][0], ranges["trace_fit"][1] + 1):
        spec = make_episode_spec(seed_id, episode_id)
        actions = deterministic_behavior_actions(seed_id, episode_id)
        trace_rows.append((spec, actions, execute_episode(spec, actions)))
    trace_model = fit_trace_decomposition(trace_rows, float(contract["trace_decomposition_ridge"]))
    require(_same_float_lists(list(trace_model.coefficients), trace_payload["coefficients"]), "TCD_TRACE_MODEL_REPLAY_MISMATCH")
    require(abs(trace_model.intercept - trace_payload["intercept"]) <= 1e-12, "TCD_TRACE_INTERCEPT_REPLAY_MISMATCH")

    graph_path = Path(seed["base_gfg_samples"])
    require(file_sha256(graph_path) == seed["base_gfg_samples_sha256"], "TCD_GFG_LEDGER_HASH_MISMATCH")
    graphs = read_jsonl(graph_path)
    expected_graph_count = int(config["candidate_test_episodes"])
    require(len(graphs) == expected_graph_count, "TCD_GFG_COUNT_MISMATCH")
    for offset, graph in enumerate(graphs):
        validate_base_gfg(graph)
        episode_id = int(ranges["candidate_test"][0]) + offset
        spec = make_episode_spec(seed_id, episode_id)
        actions = deterministic_behavior_actions(seed_id, episode_id)
        result = execute_episode(spec, actions)
        require(terminal_consequence_only(spec, actions) == result.consequence, "TCD_FAST_REPLAY_SEMANTICS_MISMATCH")
        require(graph == build_base_gfg(spec, actions, result), "TCD_GFG_RECONSTRUCTION_MISMATCH")

    detail_path = Path(seed["candidate_and_credit_details"])
    require(file_sha256(detail_path) == seed["candidate_and_credit_details_sha256"], "TCD_DETAIL_HASH_MISMATCH")
    details = read_jsonl(detail_path)
    require(len(details) == expected_graph_count, "TCD_DETAIL_COUNT_MISMATCH")
    for detail, graph in zip(details, graphs):
        require(detail["graph_sha256"] == graph["graph_sha256"], "TCD_DETAIL_GRAPH_ID_MISMATCH")

    policy_specs = [
        make_episode_spec(seed_id, episode_id)
        for episode_id in range(ranges["policy_test"][0], ranges["policy_test"][1] + 1)
    ]
    for method in METHODS:
        method_result = seed["methods"][method]
        checkpoint_path = Path(method_result["checkpoint"])
        require(file_sha256(checkpoint_path) == method_result["checkpoint_sha256"], "TCD_POLICY_HASH_MISMATCH")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(checkpoint["seed"] == seed_id and checkpoint["method"] == method, "TCD_POLICY_IDENTITY_MISMATCH")
        policy = CreditPolicy(hidden_size=int(contract["policy"]["hidden_size"]))
        policy.load_state_dict(checkpoint["state_dict"])
        replay = evaluate_credit_policy(policy, policy_specs)
        expected = method_result["evaluation"]
        for key in ("mean_terminal_consequence", "terminal_success_rate", "functional_action_accuracy"):
            require(abs(replay[key] - expected[key]) <= 1e-15, f"TCD_POLICY_REPLAY_MISMATCH:{method}:{key}")
        require(replay["per_episode_consequence"] == expected["per_episode_consequence"], "TCD_POLICY_CONSEQUENCE_LEDGER_MISMATCH")
        require(replay["per_episode_success"] == expected["per_episode_success"], "TCD_POLICY_SUCCESS_LEDGER_MISMATCH")
    return {
        "seed": seed_id,
        "status": "PASS",
        "gfg_count": len(graphs),
        "policy_count": len(METHODS),
    }


def check(aggregate_path: Path, output_path: Path) -> dict[str, Any]:
    contract_path = PACKAGE / "EXPERIMENT_CONTRACT.json"
    protocol_path = PACKAGE / "PROTOCOL_FREEZE.md"
    contract = read_json(contract_path)
    aggregate_result = read_json(aggregate_path)
    require(aggregate_result["mode"] == "formal", "TCD_CHECK_REQUIRES_FORMAL")
    freeze = read_json(PACKAGE / "CONTRACT_FREEZE.json")
    require(file_sha256(contract_path) == freeze["experiment_contract_sha256"], "TCD_FROZEN_CONTRACT_CHANGED")
    require(file_sha256(protocol_path) == freeze["protocol_sha256"], "TCD_FROZEN_PROTOCOL_CHANGED")
    for name, expected in freeze["source_hashes"].items():
        require(file_sha256(PACKAGE / name) == expected, f"TCD_FROZEN_SOURCE_CHANGED:{name}")
    seed_rows = [read_json(Path(path)) for path in aggregate_result["seed_result_paths"]]
    seed_checks = [check_seed(Path(path), contract) for path in aggregate_result["seed_result_paths"]]
    replay_aggregate = aggregate("formal", seed_rows, contract)
    for method in METHODS:
        for metric in ("mean_terminal_consequence", "terminal_success_rate", "functional_action_accuracy", "candidate_recall"):
            expected = aggregate_result["methods"][method][metric]
            actual = replay_aggregate["methods"][method][metric]
            require(expected == actual, f"TCD_AGGREGATE_REPLAY_MISMATCH:{method}:{metric}")
    require(aggregate_result["scientific_gates"] == replay_aggregate["scientific_gates"], "TCD_GATE_REPLAY_MISMATCH")
    result = {
        "schema": "gfg-temporal-credit-independent-check-v1",
        "status": "PASS",
        "scientific_status": aggregate_result["scientific_status"],
        "aggregate_sha256": file_sha256(aggregate_path),
        "seed_checks": seed_checks,
        "checks": {
            "frozen_contract_and_sources": "PASS",
            "dataset_split_isolation": "PASS",
            "trace_model_reconstruction": "PASS",
            "base_gfg_hash_and_reconstruction": "PASS",
            "forbidden_credit_edge_absence": "PASS",
            "compact_replay_semantics": "PASS",
            "policy_checkpoint_replay": "PASS",
            "aggregate_metric_recomputation": "PASS"
        }
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.aggregate.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
