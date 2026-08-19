from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.gfg_nanogpt_adjacent_response_transport_v1.inventory import (
    file_sha256,
    load_array,
    load_named_array,
    read_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.nanogpt_adapter import _load_model_module
from experiments.gfg_nanogpt_competitor_handoff_outcome_v1.analysis import (
    _final_truth,
    _metrics,
    compile_competitor_coordinates,
    compile_response_dataset,
)
from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import CONFIRMATION_RUNS, DEVELOPMENT_RUNS
from experiments.gfg_nanogpt_native_prebranch_left_history_v1.analysis import IDENTITY_MATERIAL, RESPONSE_ROOT, RobustSpace
from experiments.gfg_nanogpt_target_support_branch_v1.analysis import _remainder_mask


DEFAULT_TRAINER_ROOT = Path(r"D:\codex\dependencies\nanoGPT-3adf61e")
K = 64
COMPONENTS: dict[str, tuple[str, ...]] = {
    "embedding_readout": ("transformer.wte.",),
    "position_embedding": ("transformer.wpe.",),
    "h0_norm": ("transformer.h.0.ln_",),
    "h0_attn": ("transformer.h.0.attn.",),
    "h0_mlp": ("transformer.h.0.mlp.",),
    "h1_norm": ("transformer.h.1.ln_",),
    "h1_attn": ("transformer.h.1.attn.",),
    "h1_mlp": ("transformer.h.1.mlp.",),
    "final_norm": ("transformer.ln_f.",),
}
METHODS = (
    "f1_f3_f5_full_retrieval",
    "full_network_total_jvp",
    "full_network_component_jvp",
    "full_network_total_jvp_plus_gaps",
    "full_network_component_jvp_plus_gaps",
)
BASELINE = METHODS[0]
PRIMARY = METHODS[-1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model(trainer_root: Path) -> torch.nn.Module:
    module = _load_model_module(trainer_root.resolve())
    config = module.GPTConfig(
        block_size=3,
        vocab_size=24,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )
    return module.GPT(config).cpu().eval()


def _component_for_parameter(name: str) -> str:
    matches = [component for component, prefixes in COMPONENTS.items() if any(name.startswith(prefix) for prefix in prefixes)]
    require(len(matches) == 1, f"PARAMETER_COMPONENT_PARTITION_INVALID:{name}:{matches}")
    return matches[0]


def _section_jvp(
    model: torch.nn.Module,
    section: dict[str, Any],
    first_section: bool,
    central_direction_gate: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    state_path = Path(str(section["receiver_state_path"]))
    transition_path = Path(str(section["transition_path"]))
    entry_root = state_path.parents[3]
    state_doc = read_json(state_path)
    transition_doc = read_json(transition_path)
    state = load_named_array(entry_root, state_doc["state"]["parameters"])
    update = load_named_array(entry_root, transition_doc["step"]["parameter_update"])
    named = tuple(name for name, _parameter in model.named_parameters())
    require(set(state) == set(update) == set(named), f"PARAMETER_NAME_SET_INVALID:{section['section_id']}")
    partition = {name: _component_for_parameter(name) for name in named}
    params = {name: torch.from_numpy(np.ascontiguousarray(state[name]).copy()) for name in named}
    tangent = {name: torch.from_numpy(np.ascontiguousarray(update[name]).copy()) for name in named}
    inputs_np = np.load(str(section["evaluation_input_path"]), allow_pickle=False)
    require(inputs_np.shape == (212, 3) and inputs_np.dtype == np.int64, "EVALUATION_INPUT_INVALID")
    inputs = torch.from_numpy(inputs_np)

    def forward(values: dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _loss = functional_call(model, values, (inputs,), tie_weights=True)
        return logits[:, -1, :]

    component_values: list[torch.Tensor] = []
    primal: torch.Tensor | None = None
    complete: torch.Tensor | None = None
    with sdpa_kernel(SDPBackend.MATH):
        for component in COMPONENTS:
            child_tangent = {
                name: tangent[name] if partition[name] == component else torch.zeros_like(tangent[name])
                for name in named
            }
            child_primal, child = jvp(forward, (params,), (child_tangent,))
            if primal is None:
                primal = child_primal
            else:
                require(torch.equal(primal, child_primal), f"JVP_PRIMAL_DRIFT:{section['section_id']}:{component}")
            component_values.append(child)
        if first_section:
            _full_primal, complete = jvp(forward, (params,), (tangent,))

    require(primal is not None, "JVP_PRIMAL_MISSING")
    components = torch.stack(component_values, dim=-1)
    total = torch.sum(components, dim=-1)
    full_sum_max_abs = None
    if complete is not None:
        full_sum_max_abs = float(torch.max(torch.abs(total - complete)))
        require(full_sum_max_abs <= 1e-5, f"COMPONENT_SUM_MISMATCH:{full_sum_max_abs}")

    response_path = RESPONSE_ROOT / "sections" / f"{section['section_id']}.npz"
    require(response_path.is_file(), f"RESPONSE_SECTION_MISSING:{response_path}")
    with np.load(response_path, allow_pickle=False) as payload:
        alphas = np.asarray(payload["alphas"], dtype=np.float64)
        zero = int(np.flatnonzero(np.isclose(alphas, 0.0, rtol=0.0, atol=1e-12))[0])
        minus = int(np.flatnonzero(np.isclose(alphas, -0.125, rtol=0.0, atol=1e-12))[0])
        plus = int(np.flatnonzero(np.isclose(alphas, 0.125, rtol=0.0, atol=1e-12))[0])
        recorded = np.asarray(payload["all_logits"][zero, 0], dtype=np.float32)
        central = (
            np.asarray(payload["all_logits"][plus, 0], dtype=np.float64)
            - np.asarray(payload["all_logits"][minus, 0], dtype=np.float64)
        ) / 0.25
        groups = np.asarray(payload["groups"], dtype=np.int64)
    primal_np = primal.detach().numpy().astype(np.float32, copy=False)
    total_np = total.detach().numpy().astype(np.float64, copy=False)
    primal_max_abs = float(np.max(np.abs(primal_np.astype(np.float64) - recorded.astype(np.float64))))
    require(primal_max_abs <= 5e-5, f"ALPHA_ZERO_FORWARD_MISMATCH:{section['section_id']}:{primal_max_abs}")
    left = total_np.reshape(-1)
    right = central.reshape(-1)
    correlation = float(np.corrcoef(left, right)[0, 1])
    require(np.isfinite(correlation), f"JVP_CENTRAL_DIRECTION_NONFINITE:{section['section_id']}")
    if central_direction_gate:
        require(correlation >= 0.98, f"JVP_CENTRAL_DIRECTION_GATE_FAILED:{section['section_id']}:{correlation}")
    audit = {
        "section_id": str(section["section_id"]),
        "entry_id": str(section["entry_id_audit_only"]),
        "optimizer_step": int(state_doc["optimizer_step"]),
        "receiver_state_sha256": file_sha256(state_path),
        "actual_update_sha256": file_sha256(transition_path),
        "evaluation_input_sha256": file_sha256(Path(str(section["evaluation_input_path"]))),
        "recorded_response_section_sha256": file_sha256(response_path),
        "alpha_values_used_for_coordinate": [0.0],
        "alpha_values_used_for_validation_only": [-0.125, 0.125],
        "alpha_zero_forward_max_abs": primal_max_abs,
        "jvp_central_difference_correlation": correlation,
        "component_sum_vs_full_jvp_max_abs": full_sum_max_abs,
        "parameter_partition": partition,
    }
    return components.detach().numpy().astype(np.float32, copy=False), groups, audit


def compile_receptive_coordinates(
    response: dict[str, Any], trainer_root: Path, central_direction_gate: bool = True
) -> dict[str, Any]:
    inventory = read_json(RESPONSE_ROOT / "RESOLVED_INVENTORY.json")
    by_section = {str(row["section_id"]): row for row in inventory["sections"]}
    identities = read_json(IDENTITY_MATERIAL)["entries"]
    records = response["records"]
    record_by_section: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        record_by_section[str(record["section_id"])].append(index)
    component_gap = np.full((len(records), 23, len(COMPONENTS)), np.nan, dtype=np.float32)
    total_gap = np.full((len(records), 23), np.nan, dtype=np.float32)
    coordinate_rows: list[dict[str, Any]] = [{} for _ in records]
    audits: list[dict[str, Any]] = []
    model = _model(trainer_root)
    for ordinal, section_id in enumerate(sorted(record_by_section), start=1):
        section = by_section[section_id]
        logit_jvp, groups, audit = _section_jvp(
            model,
            section,
            first_section=ordinal == 1,
            central_direction_gate=central_direction_gate,
        )
        audits.append(audit)
        entry_id = str(section["entry_id_audit_only"])
        identity_index = {
            str(row["evaluation_unit_id"]): row_index
            for row_index, row in enumerate(identities[entry_id])
        }
        response_path = RESPONSE_ROOT / "sections" / f"{section_id}.npz"
        with np.load(response_path, allow_pickle=False) as payload:
            alphas = np.asarray(payload["alphas"], dtype=np.float64)
            zero = int(np.flatnonzero(np.isclose(alphas, 0.0, rtol=0.0, atol=1e-12))[0])
            logits = np.asarray(payload["all_logits"][zero, 0], dtype=np.float64)
        for record_index in record_by_section[section_id]:
            record = records[record_index]
            row_index = identity_index[str(record["evaluation_unit_id"])]
            target = int(groups[row_index])
            require(target == int(record["target_group"]), f"TARGET_IDENTITY_MISMATCH:{record['record_id']}")
            order = np.argsort(logits[row_index])[::-1]
            competitors = order[order != target]
            require(len(competitors) == 23, "COMPETITOR_COUNT_INVALID")
            values = logit_jvp[row_index, target][None, :] - logit_jvp[row_index, competitors]
            component_gap[record_index] = values
            total_gap[record_index] = np.sum(values, axis=1)
            coordinate_rows[record_index] = {
                "row_index": record_index,
                "record_id": str(record["record_id"]),
                "entry_id": entry_id,
                "section_id": section_id,
                "evaluation_unit_id": str(record["evaluation_unit_id"]),
                "evaluation_row_index": row_index,
                "target_group": target,
            }
        print(f"JVP_SECTION {ordinal}/72 {section_id}", flush=True)
    require(bool(np.all(np.isfinite(component_gap))) and bool(np.all(np.isfinite(total_gap))), "RECEPTIVE_COORDINATE_INCOMPLETE")
    return {
        "component_gap_jvp": component_gap,
        "total_gap_jvp": total_gap,
        "coordinate_rows": coordinate_rows,
        "section_audits": audits,
    }


def _scaled(values: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    scaler = RobustSpace().fit(flat[train])
    train_values = scaler.transform(flat[train])
    test_values = scaler.transform(flat[test])
    divisor = math.sqrt(train_values.shape[1])
    return train_values / divisor, test_values / divisor


def _spaces(
    response: dict[str, Any], competitor: dict[str, Any], receptive: dict[str, Any], train: np.ndarray, test: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base_train, base_test = _scaled(response["spaces"]["X0"], train, test)
    gap_train, gap_test = _scaled(competitor["gaps"], train, test)
    total_train, total_test = _scaled(receptive["total_gap_jvp"], train, test)
    component_train, component_test = _scaled(receptive["component_gap_jvp"], train, test)
    return {
        METHODS[0]: (base_train, base_test),
        METHODS[1]: (np.concatenate([base_train, total_train], axis=1), np.concatenate([base_test, total_test], axis=1)),
        METHODS[2]: (np.concatenate([base_train, component_train], axis=1), np.concatenate([base_test, component_test], axis=1)),
        METHODS[3]: (np.concatenate([base_train, gap_train, total_train], axis=1), np.concatenate([base_test, gap_test, total_test], axis=1)),
        METHODS[4]: (np.concatenate([base_train, gap_train, component_train], axis=1), np.concatenate([base_test, gap_test, component_test], axis=1)),
    }


def _predict_fold(
    response: dict[str, Any], competitor: dict[str, Any], receptive: dict[str, Any], labels: np.ndarray,
    train: np.ndarray, test: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probabilities: dict[str, np.ndarray] = {}
    neighbors: dict[str, np.ndarray] = {}
    for method, (train_space, test_space) in _spaces(response, competitor, receptive, train, test).items():
        distances, local = cKDTree(train_space).query(test_space, k=K, workers=-1)
        weights = 1.0 / np.maximum(np.asarray(distances, dtype=np.float64), 1e-9)
        weights /= np.sum(weights, axis=1, keepdims=True)
        global_neighbors = train[np.asarray(local, dtype=np.int64)]
        probabilities[method] = np.sum(labels[global_neighbors] * weights, axis=1)
        neighbors[method] = global_neighbors.astype(np.int32, copy=False)
    return probabilities, neighbors


def _repair(truth: np.ndarray, baseline: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    fixed = int(np.sum(mask & (baseline != truth) & (prediction == truth)))
    broken = int(np.sum(mask & (baseline == truth) & (prediction != truth)))
    return {"fixed_baseline_errors": fixed, "newly_broken_baseline_answers": broken, "net_repairs": fixed - broken}


def run_analysis(
    trainer_root: Path = DEFAULT_TRAINER_ROOT,
    central_direction_gate: bool = True,
) -> dict[str, Any]:
    response, response_audit, response_sources = compile_response_dataset()
    competitor = compile_competitor_coordinates(response)
    receptive = compile_receptive_coordinates(
        response,
        trainer_root.resolve(),
        central_direction_gate=central_direction_gate,
    )
    remainder, _ = _remainder_mask(response)
    start_correct, truth, boundary = _final_truth(response)
    labels = truth.astype(np.float64)
    entries = np.asarray(response["entries"], dtype=object)
    development = np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object))
    confirmation = np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object))
    probabilities = {method: np.full(len(truth), np.nan, dtype=np.float64) for method in METHODS}
    neighbors = {method: np.full((len(truth), K), -1, dtype=np.int32) for method in METHODS}
    for run in DEVELOPMENT_RUNS:
        test = np.flatnonzero(entries == run)
        train = np.flatnonzero(development & (entries != run))
        fold, chosen = _predict_fold(response, competitor, receptive, labels, train, test)
        for method in METHODS:
            probabilities[method][test] = fold[method]
            neighbors[method][test] = chosen[method]
    test = np.flatnonzero(confirmation)
    train = np.flatnonzero(development)
    fold, chosen = _predict_fold(response, competitor, receptive, labels, train, test)
    for method in METHODS:
        probabilities[method][test] = fold[method]
        neighbors[method][test] = chosen[method]
    predictions = {method: probabilities[method] >= 0.5 for method in METHODS}
    for method in METHODS:
        require(bool(np.all(np.isfinite(probabilities[method]))), f"PROBABILITY_INCOMPLETE:{method}")
        require(bool(np.all(neighbors[method] >= 0)), f"NEIGHBOR_INCOMPLETE:{method}")
        require(all(entries[i] not in set(entries[neighbors[method][i]]) for i in range(len(entries))), f"SAME_RUN_NEIGHBOR:{method}")

    severe = np.asarray(response["labels"]["severe_conflict"], dtype=bool)
    split_masks = {"development": development, "confirmation": confirmation, "all_runs": np.ones(len(truth), dtype=bool)}
    subset_masks = {"overall": np.ones(len(truth), dtype=bool), "severe_conflict": severe, "group_level_remainder_311": remainder}
    metrics: dict[str, Any] = {}
    repairs: dict[str, Any] = {}
    baseline = predictions[BASELINE]
    for split_name, split_mask in split_masks.items():
        metrics[split_name] = {}
        repairs[split_name] = {}
        for method in METHODS:
            metrics[split_name][method] = {}
            repairs[split_name][method] = {}
            for subset_name, subset_mask in subset_masks.items():
                mask = split_mask & subset_mask
                metrics[split_name][method][subset_name] = _metrics(truth, predictions[method], start_correct, boundary, mask)
                repairs[split_name][method][subset_name] = _repair(truth, baseline, predictions[method], mask)
    runwise: dict[str, Any] = {}
    for run in (*DEVELOPMENT_RUNS, *CONFIRMATION_RUNS):
        run_mask = entries == run
        runwise[run] = {
            method: {
                subset: {
                    "metrics": _metrics(truth, predictions[method], start_correct, boundary, run_mask & subset_mask),
                    "repairs": _repair(truth, baseline, predictions[method], run_mask & subset_mask),
                }
                for subset, subset_mask in subset_masks.items()
            }
            for method in METHODS
        }
    hard = repairs["all_runs"][PRIMARY]["group_level_remainder_311"]["net_repairs"]
    hard_development = repairs["development"][PRIMARY]["group_level_remainder_311"]["net_repairs"]
    hard_confirmation = repairs["confirmation"][PRIMARY]["group_level_remainder_311"]["net_repairs"]
    if hard > 0 and hard_development > 0 and hard_confirmation > 0:
        verdict = "FULL_NETWORK_RECEPTIVE_STATE_DIAGNOSTIC_IMPROVES_HARD_BRANCH"
    elif hard > 0:
        verdict = "FULL_NETWORK_RECEPTIVE_STATE_SIGNAL_NOT_SPLIT_STABLE"
    else:
        verdict = "FULL_NETWORK_RECEPTIVE_STATE_DIAGNOSTIC_DOES_NOT_IMPROVE_HARD_BRANCH"
    ledger = [
        {
            "row_index": index,
            "record_id": str(record["record_id"]),
            "entry_id": str(record["entry_id"]),
            "truth_final_correct": bool(truth[index]),
            "truth_boundary": str(boundary[index]),
            "severe_conflict": bool(severe[index]),
            "group_level_remainder_311": bool(remainder[index]),
            "probability_final_correct": {method: float(probabilities[method][index]) for method in METHODS},
            "predicted_final_correct": {method: bool(predictions[method][index]) for method in METHODS},
        }
        for index, record in enumerate(response["records"])
    ]
    return {
        "response": response,
        "response_audit": response_audit,
        "response_sources": response_sources,
        "competitor": competitor,
        "receptive": receptive,
        "neighbors": neighbors,
        "prediction_ledger": ledger,
        "metrics": metrics,
        "repairs": repairs,
        "runwise": runwise,
        "decision": {
            "schema": "nanogpt-full-network-receptive-state-decision-v1",
            "status": "PASS",
            "verdict": verdict,
            "evidence_status": "POST_HOC_MECHANISM_DIAGNOSTIC_ONLY",
            "jvp_allowed_as_formal_predictor_input": False,
            "primary_method": PRIMARY,
            "primary_hard_net_repairs": {"development": hard_development, "confirmation": hard_confirmation, "all_runs": hard},
            "curve_metrics_used_for_decision": False,
            "post_outcome_material_used_as_coordinate": False,
        },
        "feature_manifest": {
            "schema": "nanogpt-full-network-receptive-state-feature-manifest-v1",
            "status": "PASS",
            "component_ids": list(COMPONENTS),
            "component_count": len(COMPONENTS),
            "competitor_count": 23,
            "total_jvp_dimension": 23,
            "component_jvp_dimension": 23 * len(COMPONENTS),
            "coordinate_alpha_values": [0.0],
            "validation_only_alpha_values": [-0.125, 0.125],
            "diagnostic_only": True,
            "formal_predictor_input": False,
            "trainer_model_py_sha256": file_sha256(trainer_root / "model.py"),
        },
    }
