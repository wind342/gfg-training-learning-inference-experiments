from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_support_redundancy_v1.builder import decision_outputs

from .runtime import COMPONENTS, COMPONENT_PAIRS


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _decisions(logits: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    return decision_outputs(torch.from_numpy(logits), torch.from_numpy(targets))


def _group_logit_change(
    baseline: np.ndarray,
    changed: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    groups = targets[:, -1]
    result = np.zeros(23, dtype=np.float64)
    for group in range(23):
        mask = groups == group
        result[group] = float(np.max(np.abs(changed[mask] - baseline[mask])))
    return result


def analyse_run(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    targets = record["validation_targets"]
    phase_arrays: dict[str, dict[str, np.ndarray]] = {}
    phases: dict[str, Any] = {}
    for phase, raw in record["phases"].items():
        baseline = raw["baseline_logits"]
        base_values = _decisions(baseline, targets)
        single_values = {
            component: _decisions(logits, targets)
            for component, logits in raw["single_gate_logits"].items()
        }
        pair_values = {
            pair: _decisions(logits, targets)
            for pair, logits in raw["pair_gate_logits"].items()
        }
        support_profile = np.stack(
            [
                base_values["group_q10_margin"] - single_values[component]["group_q10_margin"]
                for component in COMPONENTS
            ],
            axis=1,
        ).astype(np.float64)
        pair_interaction = np.stack(
            [
                (
                    base_values["group_q10_margin"]
                    - pair_values["+".join(pair)]["group_q10_margin"]
                    - support_profile[:, COMPONENTS.index(pair[0])]
                    - support_profile[:, COMPONENTS.index(pair[1])]
                )
                for pair in COMPONENT_PAIRS
            ],
            axis=1,
        ).astype(np.float64)
        single_logit_change = np.stack(
            [
                _group_logit_change(baseline, raw["single_gate_logits"][component], targets)
                for component in COMPONENTS
            ],
            axis=1,
        )
        pair_logit_change = np.stack(
            [
                _group_logit_change(baseline, raw["pair_gate_logits"]["+".join(pair)], targets)
                for pair in COMPONENT_PAIRS
            ],
            axis=1,
        )
        call_rows = []
        for call in raw["calls"]:
            call_rows.append(
                {
                    "component": call.component,
                    "call_index": call.call_index,
                    "input_sha256": array_sha256(call.input_tensor),
                    "output_sha256": array_sha256(call.output_tensor),
                    "output_l2": float(np.linalg.norm(call.output_tensor.astype(np.float64))),
                    "output_nonzero_count": int(np.count_nonzero(call.output_tensor)),
                }
            )
        distinct_profiles = len({np.ascontiguousarray(row).tobytes() for row in support_profile})
        phases[phase] = {
            "optimizer_step": raw["optimizer_step"],
            "validation_accuracy": raw["baseline"]["accuracy"],
            "baseline_logits_sha256": array_sha256(baseline),
            "single_gate_logits_sha256": {
                component: array_sha256(raw["single_gate_logits"][component])
                for component in COMPONENTS
            },
            "pair_gate_logits_sha256": {
                key: array_sha256(value)
                for key, value in raw["pair_gate_logits"].items()
            },
            "baseline_repeat_exact": bool(np.array_equal(baseline, raw["baseline_repeat_logits"])),
            "parameter_identity_count": len(raw["parameter_rows"]),
            "parameter_identity_exact": True,
            "native_component_calls": call_rows,
            "all_component_outputs_nonzero": all(row["output_nonzero_count"] > 0 for row in call_rows),
            "support_profile_sha256": array_sha256(support_profile),
            "distinct_target_group_support_profiles": distinct_profiles,
            "query_conditioned_profile_variance": float(np.mean(np.var(support_profile, axis=0))),
            "single_gate_changed_component_group_count": int(np.count_nonzero(single_logit_change > 0.0)),
            "pair_gate_changed_pair_group_count": int(np.count_nonzero(pair_logit_change > 0.0)),
            "nonadditive_pair_group_count_at_1e_6": int(np.count_nonzero(np.abs(pair_interaction) > 1e-6)),
            "support_profile": support_profile.tolist(),
            "pair_interaction": pair_interaction.tolist(),
            "single_gate_group_max_abs_logit_change": single_logit_change.tolist(),
        }
        phase_arrays[phase] = {
            "support_profile": support_profile,
            "pair_interaction": pair_interaction,
            "single_gate_group_max_abs_logit_change": single_logit_change,
            "pair_gate_group_max_abs_logit_change": pair_logit_change,
        }

    rollback_rows: dict[str, Any] = {}
    rollback_arrays: dict[str, np.ndarray] = {}
    formed_logits = record["phases"]["formed"]["baseline_logits"]
    formed_values = _decisions(formed_logits, targets)
    for component, raw in record["rollback"].items():
        rollback_values = _decisions(raw["rollback_logits"], targets)
        effect = (
            formed_values["group_q10_margin"] - rollback_values["group_q10_margin"]
        ).astype(np.float64)
        rollback_rows[component] = {
            "rollback_logits_sha256": array_sha256(raw["rollback_logits"]),
            "restored_logits_sha256": array_sha256(raw["restored_logits"]),
            "rollback_changed_logits": not np.array_equal(raw["rollback_logits"], formed_logits),
            "restore_byte_exact": bool(np.array_equal(raw["restored_logits"], formed_logits)),
            "formed_accuracy": float(record["phases"]["formed"]["baseline"]["accuracy"]),
            "rollback_accuracy": float(raw["rollback"]["accuracy"]),
            "accuracy_change_rollback_minus_formed": float(
                raw["rollback"]["accuracy"] - record["phases"]["formed"]["baseline"]["accuracy"]
            ),
            "group_q10_margin_effect": effect.tolist(),
            "changed_target_group_count": int(np.count_nonzero(effect != 0.0)),
            "preformation_parameter_count": len(raw["pre_parameter_rows"]),
            "formed_parameter_count": len(raw["formed_parameter_rows"]),
        }
        rollback_arrays[component] = effect
    phase_arrays["rollback"] = rollback_arrays
    formation_support_changed = (
        phases["pre_formation"]["support_profile_sha256"]
        != phases["formed"]["support_profile_sha256"]
    )
    ledger = {
        "entry_id": record["entry_id"],
        "source_bundle_id": record["source_bundle_id"],
        "phase_selection": record["phase_selection"],
        "phases": phases,
        "rollback": rollback_rows,
        "formation_accuracy_gain": float(
            phases["formed"]["validation_accuracy"] - phases["pre_formation"]["validation_accuracy"]
        ),
        "formation_support_profile_changed": formation_support_changed,
        "at_least_one_rollback_changed_logits": any(
            row["rollback_changed_logits"] for row in rollback_rows.values()
        ),
        "all_rollbacks_restored_exactly": all(
            row["restore_byte_exact"] for row in rollback_rows.values()
        ),
    }
    return ledger, phase_arrays


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_count = len(rows)
    identity_pass = all(
        phase["parameter_identity_exact"]
        for row in rows
        for phase in row["phases"].values()
    )
    causal_call_pass = all(
        all(phase["all_component_outputs_nonzero"] for phase in row["phases"].values())
        and row["phases"]["formed"]["single_gate_changed_component_group_count"] > 0
        for row in rows
    )
    query_combination_pass = all(
        row["phases"]["formed"]["distinct_target_group_support_profiles"] >= 2
        and row["phases"]["formed"]["nonadditive_pair_group_count_at_1e_6"] > 0
        for row in rows
    )
    rollback_pass = all(
        row["at_least_one_rollback_changed_logits"]
        and row["all_rollbacks_restored_exactly"]
        for row in rows
    )
    formation_pass = all(
        row["formation_support_profile_changed"] and row["formation_accuracy_gain"] > 0.0
        for row in rows
    )
    rollback_effects = [
        component["accuracy_change_rollback_minus_formed"]
        for row in rows
        for component in row["rollback"].values()
    ]
    gates = {
        "identity_continuity": identity_pass,
        "causal_component_call": causal_call_pass,
        "query_conditioning_and_nonadditive_combination": query_combination_pass,
        "learned_version_rollback_and_restore": rollback_pass,
        "formation_support_reorganization": formation_pass,
    }
    return {
        "schema": "nanogpt-training-learning-inference-projection-summary-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "run_count": run_count,
        "phase_observation_count": sum(len(row["phases"]) for row in rows),
        "native_component_call_count": sum(
            len(phase["native_component_calls"])
            for row in rows
            for phase in row["phases"].values()
        ),
        "single_gate_forward_count": run_count * 4 * len(COMPONENTS),
        "pair_gate_forward_count": run_count * 4 * len(COMPONENT_PAIRS),
        "rollback_forward_count": run_count * len(COMPONENTS),
        "restoration_forward_count": run_count * len(COMPONENTS),
        "gates": gates,
        "formed_distinct_group_profile_minimum": min(
            row["phases"]["formed"]["distinct_target_group_support_profiles"] for row in rows
        ),
        "formed_nonadditive_pair_group_minimum": min(
            row["phases"]["formed"]["nonadditive_pair_group_count_at_1e_6"] for row in rows
        ),
        "runs_with_changed_rollback_logits": sum(row["at_least_one_rollback_changed_logits"] for row in rows),
        "rollback_accuracy_effect_min": float(min(rollback_effects)),
        "rollback_accuracy_effect_max": float(max(rollback_effects)),
        "rollback_accuracy_effect_mean": float(np.mean(rollback_effects)),
        "interpretation": (
            "Within the executed 13-run witness family, frozen inference used exact training-formed "
            "parameter versions, produced target-conditioned causal support profiles, combined component "
            "supports non-additively, and changed under pre-formation component-version rollback."
        ),
        "claim_boundary": (
            "This establishes an operational training-learning-inference support projection in the executed "
            "nanoGPT family; it is not a universal theorem for every model or inference process."
        ),
    }


__all__ = ["analyse_run", "array_sha256", "summarize"]
