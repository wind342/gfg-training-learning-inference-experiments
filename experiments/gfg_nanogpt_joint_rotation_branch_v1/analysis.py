from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.gfg_nanogpt_adjacent_response_transport_v1.inventory import (
    file_sha256,
    load_named_array,
    read_json,
)
from experiments.gfg_nanogpt_competitor_handoff_outcome_v1.analysis import (
    _final_truth,
    _metrics,
    compile_competitor_coordinates,
    compile_response_dataset,
)
from experiments.gfg_nanogpt_full_network_receptive_state_v1.analysis import (
    DEFAULT_TRAINER_ROOT,
    _model,
)
from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
)
from experiments.gfg_nanogpt_native_prebranch_left_history_v1.analysis import (
    IDENTITY_MATERIAL,
    RESPONSE_ROOT,
)
from experiments.gfg_nanogpt_target_support_branch_v1.analysis import _remainder_mask


FULL_NETWORK_REPORT = (
    Path(__file__).parents[1]
    / "gfg_nanogpt_cumulative_scientist_v1"
    / "reports"
    / "full_network_receptive_state_v2"
)
METHODS = (
    "linear",
    "joint_rotation",
    "hidden_curvature",
    "quadratic_complete",
)
BASELINE = METHODS[0]
PRIMARY = METHODS[-1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class HiddenStateModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.gpt = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _batch, length = idx.size()
        positions = torch.arange(length, dtype=torch.long, device=idx.device)
        hidden = self.gpt.transformer.drop(
            self.gpt.transformer.wte(idx) + self.gpt.transformer.wpe(positions)
        )
        for block in self.gpt.transformer.h:
            hidden = block(hidden)
        return self.gpt.transformer.ln_f(hidden)[:, -1, :]


def _hidden_derivatives(
    wrapper: HiddenStateModel,
    params: dict[str, torch.Tensor],
    tangent: dict[str, torch.Tensor],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def forward(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(wrapper, values, (inputs,), tie_weights=True)

    def first(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return jvp(forward, (values,), (tangent,))[1]

    with sdpa_kernel(SDPBackend.MATH):
        hidden, hidden_first = jvp(forward, (params,), (tangent,))
        _first_primal, hidden_second = jvp(first, (params,), (tangent,))
    return hidden, hidden_first, hidden_second


def compile_coordinates(
    response: dict[str, Any],
    competitor: dict[str, Any],
    trainer_root: Path = DEFAULT_TRAINER_ROOT,
) -> dict[str, Any]:
    require((FULL_NETWORK_REPORT / "READY").is_file(), "FULL_NETWORK_REPORT_NOT_READY")
    with np.load(FULL_NETWORK_REPORT / "RECEPTIVE_COORDINATES.npz", allow_pickle=False) as payload:
        expected_ids = np.asarray(payload["record_ids"], dtype=object)
        expected_first = np.asarray(payload["total_gap_jvp"], dtype=np.float64)
    records = response["records"]
    require(
        list(expected_ids) == [str(row["record_id"]) for row in records],
        "FULL_NETWORK_RECORD_ORDER_MISMATCH",
    )
    inventory = read_json(RESPONSE_ROOT / "RESOLVED_INVENTORY.json")
    by_inventory = {str(row["section_id"]): row for row in inventory["sections"]}
    identities = read_json(IDENTITY_MATERIAL)["entries"]
    by_section: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_section[str(record["section_id"])].append(index)

    count = len(records)
    first_gap = np.full((count, 23), np.nan, dtype=np.float32)
    joint = np.full((count, 23), np.nan, dtype=np.float32)
    hidden_curve = np.full((count, 23), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = [{} for _ in records]
    audits: list[dict[str, Any]] = []
    wrapper = HiddenStateModel(_model(trainer_root.resolve())).cpu().eval()

    for ordinal, section_id in enumerate(sorted(by_section), start=1):
        section = by_inventory[section_id]
        state_path = Path(str(section["receiver_state_path"]))
        transition_path = Path(str(section["transition_path"]))
        entry_root = state_path.parents[3]
        state_doc = read_json(state_path)
        transition_doc = read_json(transition_path)
        state = load_named_array(entry_root, state_doc["state"]["parameters"])
        update = load_named_array(entry_root, transition_doc["step"]["parameter_update"])
        named = tuple(name for name, _parameter in wrapper.gpt.named_parameters())
        require(set(state) == set(update) == set(named), f"PARAMETER_SET_MISMATCH:{section_id}")
        params = {
            f"gpt.{name}": torch.from_numpy(np.ascontiguousarray(state[name]).copy())
            for name in named
        }
        tangent = {
            f"gpt.{name}": torch.from_numpy(np.ascontiguousarray(update[name]).copy())
            for name in named
        }
        inputs_np = np.load(str(section["evaluation_input_path"]), allow_pickle=False)
        require(inputs_np.shape == (212, 3), f"EVALUATION_INPUT_SHAPE_INVALID:{section_id}")
        hidden, hidden_first, hidden_second = _hidden_derivatives(
            wrapper, params, tangent, torch.from_numpy(inputs_np)
        )
        hidden_np = hidden.detach().numpy().astype(np.float64, copy=False)
        hidden_first_np = hidden_first.detach().numpy().astype(np.float64, copy=False)
        hidden_second_np = hidden_second.detach().numpy().astype(np.float64, copy=False)
        weight = np.asarray(state["transformer.wte.weight"], dtype=np.float64)
        weight_update = np.asarray(update["transformer.wte.weight"], dtype=np.float64)
        logits = hidden_np @ weight.T

        response_path = RESPONSE_ROOT / "sections" / f"{section_id}.npz"
        with np.load(response_path, allow_pickle=False) as payload:
            alphas = np.asarray(payload["alphas"], dtype=np.float64)
            zero = int(np.flatnonzero(np.isclose(alphas, 0.0, rtol=0.0, atol=1e-12))[0])
            recorded = np.asarray(payload["all_logits"][zero, 0], dtype=np.float64)
            groups = np.asarray(payload["groups"], dtype=np.int64)
        logit_error = float(np.max(np.abs(logits - recorded)))
        require(logit_error <= 5e-5, f"HIDDEN_LOGIT_RECONSTRUCTION_FAILED:{section_id}:{logit_error}")
        identity_index = {
            str(row["evaluation_unit_id"]): row_index
            for row_index, row in enumerate(identities[str(section["entry_id_audit_only"])])
        }
        section_gap_error = 0.0
        section_first_error = 0.0
        for record_index in by_section[section_id]:
            record = records[record_index]
            row_index = identity_index[str(record["evaluation_unit_id"])]
            target = int(groups[row_index])
            require(target == int(record["target_group"]), f"TARGET_MISMATCH:{record['record_id']}")
            # Preserve the exact frozen competitor identity order.  Equal recorded
            # logits can be reordered by tiny reconstruction round-off even when
            # their gaps are identical, while their directional responses differ.
            order = np.argsort(recorded[row_index])[::-1]
            competitors = order[order != target]
            boundary = weight[target][None, :] - weight[competitors]
            boundary_update = weight_update[target][None, :] - weight_update[competitors]
            reconstructed_gap = boundary @ hidden_np[row_index]
            reconstructed_first = (
                boundary_update @ hidden_np[row_index]
                + boundary @ hidden_first_np[row_index]
            )
            joint_value = boundary_update @ hidden_first_np[row_index]
            hidden_value = 0.5 * (boundary @ hidden_second_np[row_index])
            section_gap_error = max(
                section_gap_error,
                float(np.max(np.abs(reconstructed_gap - competitor["gaps"][record_index]))),
            )
            section_first_error = max(
                section_first_error,
                float(np.max(np.abs(reconstructed_first - expected_first[record_index]))),
            )
            first_gap[record_index] = reconstructed_first
            joint[record_index] = joint_value
            hidden_curve[record_index] = hidden_value
            rows[record_index] = {
                "row_index": record_index,
                "record_id": str(record["record_id"]),
                "entry_id": str(record["entry_id"]),
                "section_id": section_id,
                "evaluation_unit_id": str(record["evaluation_unit_id"]),
                "evaluation_row_index": row_index,
                "target_group": target,
            }
        # A gap subtracts two reconstructed logits, each admitted at 5e-5.
        require(section_gap_error <= 1e-4, f"GAP_RECONSTRUCTION_FAILED:{section_id}:{section_gap_error}")
        require(section_first_error <= 5e-4, f"FIRST_JVP_RECONSTRUCTION_FAILED:{section_id}:{section_first_error}")
        audits.append(
            {
                "section_id": section_id,
                "entry_id": str(section["entry_id_audit_only"]),
                "optimizer_step": int(state_doc["optimizer_step"]),
                "receiver_state_sha256": file_sha256(state_path),
                "actual_update_sha256": file_sha256(transition_path),
                "evaluation_input_sha256": file_sha256(Path(str(section["evaluation_input_path"]))),
                "hidden_logit_reconstruction_max_abs": logit_error,
                "gap_reconstruction_max_abs": section_gap_error,
                "first_jvp_reconstruction_max_abs": section_first_error,
                "coordinate_time_boundary": "PRE_UPDATE_STATE_PLUS_FORMED_ACTUAL_UPDATE",
            }
        )
        print(f"JOINT_ROTATION_SECTION {ordinal}/72 {section_id}", flush=True)

    require(
        bool(np.all(np.isfinite(first_gap)))
        and bool(np.all(np.isfinite(joint)))
        and bool(np.all(np.isfinite(hidden_curve))),
        "COORDINATE_NONFINITE",
    )
    return {
        "first_gap": first_gap,
        "joint_rotation": joint,
        "hidden_curvature": hidden_curve,
        "coordinate_rows": rows,
        "section_audits": audits,
    }


def _repair(
    truth: np.ndarray, baseline: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, int]:
    fixed = int(np.sum(mask & (baseline != truth) & (prediction == truth)))
    broken = int(np.sum(mask & (baseline == truth) & (prediction != truth)))
    return {
        "fixed_linear_errors": fixed,
        "newly_broken_linear_answers": broken,
        "net_repairs": fixed - broken,
    }


def run_analysis(trainer_root: Path = DEFAULT_TRAINER_ROOT) -> dict[str, Any]:
    response, response_audit, response_sources = compile_response_dataset()
    competitor = compile_competitor_coordinates(response)
    coordinates = compile_coordinates(response, competitor, trainer_root.resolve())
    gaps = np.asarray(competitor["gaps"], dtype=np.float64)
    first = np.asarray(coordinates["first_gap"], dtype=np.float64)
    joint = np.asarray(coordinates["joint_rotation"], dtype=np.float64)
    hidden = np.asarray(coordinates["hidden_curvature"], dtype=np.float64)
    endpoint = {
        "linear": gaps + first,
        "joint_rotation": gaps + first + joint,
        "hidden_curvature": gaps + first + hidden,
        "quadratic_complete": gaps + first + joint + hidden,
    }
    predictions = {name: np.all(values > 0.0, axis=1) for name, values in endpoint.items()}
    start, truth, boundary = _final_truth(response)
    remainder, _details = _remainder_mask(response)
    severe = np.asarray(response["labels"]["severe_conflict"], dtype=bool)
    entries = np.asarray(response["entries"], dtype=object)
    development = np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object))
    confirmation = np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object))
    splits = {
        "development": development,
        "confirmation": confirmation,
        "all_runs": np.ones(len(truth), dtype=bool),
    }
    subsets = {
        "overall": np.ones(len(truth), dtype=bool),
        "severe_conflict": severe,
        "group_level_remainder_311": remainder,
    }
    metrics: dict[str, Any] = {}
    repairs: dict[str, Any] = {}
    for split_name, split_mask in splits.items():
        metrics[split_name] = {}
        repairs[split_name] = {}
        for method in METHODS:
            metrics[split_name][method] = {}
            repairs[split_name][method] = {}
            for subset_name, subset_mask in subsets.items():
                mask = split_mask & subset_mask
                metrics[split_name][method][subset_name] = _metrics(
                    truth, predictions[method], start, boundary, mask
                )
                repairs[split_name][method][subset_name] = _repair(
                    truth, predictions[BASELINE], predictions[method], mask
                )
    primary_hard = {
        split: repairs[split][PRIMARY]["group_level_remainder_311"]["net_repairs"]
        for split in splits
    }
    if primary_hard["development"] > 0 and primary_hard["confirmation"] > 0:
        verdict = "JOINT_ROTATION_AND_HIDDEN_CURVATURE_REPRODUCIBLY_IMPROVE_HARD_BRANCH"
    elif primary_hard["all_runs"] > 0:
        verdict = "JOINT_ROTATION_AND_HIDDEN_CURVATURE_SIGNAL_NOT_SPLIT_STABLE"
    else:
        verdict = "JOINT_ROTATION_AND_HIDDEN_CURVATURE_DO_NOT_IMPROVE_HARD_BRANCH"
    ledger = [
        {
            "row_index": index,
            "record_id": str(record["record_id"]),
            "entry_id": str(record["entry_id"]),
            "truth_final_correct": bool(truth[index]),
            "truth_boundary": str(boundary[index]),
            "group_level_remainder_311": bool(remainder[index]),
            "predicted_final_correct": {
                method: bool(predictions[method][index]) for method in METHODS
            },
            "minimum_predicted_endpoint_gap": {
                method: float(np.min(endpoint[method][index])) for method in METHODS
            },
        }
        for index, record in enumerate(response["records"])
    ]
    return {
        "response": response,
        "response_audit": response_audit,
        "response_sources": response_sources,
        "competitor": competitor,
        "coordinates": coordinates,
        "endpoint": endpoint,
        "metrics": metrics,
        "repairs": repairs,
        "ledger": ledger,
        "decision": {
            "schema": "nanogpt-joint-rotation-branch-decision-v1",
            "status": "PASS",
            "verdict": verdict,
            "primary_method": PRIMARY,
            "primary_hard_net_repairs": primary_hard,
            "evidence_status": "POST_HOC_MECHANISM_DIAGNOSTIC_ONLY",
            "future_response_used_as_coordinate": False,
            "alpha_positive_probe_used_as_coordinate": False,
        },
    }
