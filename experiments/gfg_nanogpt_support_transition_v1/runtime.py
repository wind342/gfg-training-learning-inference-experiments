from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    require,
)
from experiments.gfg_nanogpt_support_redundancy_v1.builder import decision_outputs
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import (
    COMPONENTS,
    COMPONENT_PAIRS,
    HistoricalRunRuntime,
    load_tensor,
    objects_for_stage,
    tensor_sha256,
    unique_role_objects,
)


COMPONENT_PREFIXES = {
    "h0.attn": "transformer.h.0.attn.",
    "h0.mlp": "transformer.h.0.mlp.",
    "h1.attn": "transformer.h.1.attn.",
    "h1.mlp": "transformer.h.1.mlp.",
}


def _clone_tensor_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().contiguous().cpu().clone() for name, value in values.items()}


def _optimizer_clone(
    values: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {name: _clone_tensor_map(state) for name, state in values.items()}


def _state_hashes(values: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(values.items())}


def _rms(values: Iterable[torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for value in values:
        child = value.detach().to(torch.float64)
        total += float(torch.sum(child * child))
        count += int(child.numel())
    require(count > 0, "CST_RMS_EMPTY")
    return math.sqrt(total / count)


@dataclass
class StateSnapshot:
    parameters: dict[str, torch.Tensor]
    optimizer: dict[str, dict[str, torch.Tensor]]

    def clone(self) -> StateSnapshot:
        return StateSnapshot(_clone_tensor_map(self.parameters), _optimizer_clone(self.optimizer))

    def commitment(self) -> dict[str, Any]:
        optimizer_hashes = {
            f"{name}.{key}": tensor_sha256(value)
            for name, state in sorted(self.optimizer.items())
            for key, value in sorted(state.items())
        }
        material = {
            "optimizer": optimizer_hashes,
            "parameters": _state_hashes(self.parameters),
        }
        return {
            **material,
            "state_sha256": hashlib.sha256(canonical_bytes(material)).hexdigest(),
        }


@dataclass
class StepEvidence:
    execute_optimizer: bool
    loss: float
    raw_gradients: dict[str, torch.Tensor]
    clipped_gradients: dict[str, torch.Tensor]
    parameter_updates: dict[str, torch.Tensor]
    total_gradient_norm: float
    rng_before: dict[str, Any]
    rng_after: dict[str, Any]


def assert_snapshot_isolation(values: Iterable[StateSnapshot]) -> None:
    snapshots = list(values)
    for left_index, left in enumerate(snapshots):
        for right in snapshots[left_index + 1 :]:
            for name in left.parameters:
                require(
                    left.parameters[name].untyped_storage().data_ptr()
                    != right.parameters[name].untyped_storage().data_ptr(),
                    f"CST_SHARED_PARAMETER_STORAGE:{name}",
                )
            for name in left.optimizer:
                for key in left.optimizer[name]:
                    require(
                        left.optimizer[name][key].untyped_storage().data_ptr()
                        != right.optimizer[name][key].untyped_storage().data_ptr(),
                        f"CST_SHARED_OPTIMIZER_STORAGE:{name}.{key}",
                    )


class CausalTrainingRuntime:
    def __init__(
        self,
        bundle: Path,
        trainer_root: Path,
        *,
        device: str = "cuda",
    ) -> None:
        self.bundle = bundle.resolve()
        self.trainer_root = trainer_root.resolve()
        self.device = device
        self.base = HistoricalRunRuntime.open(
            self.bundle,
            self.trainer_root,
            device=device,
            reference_step=100,
        )
        self.model = self.base.model
        self.named_parameters = dict(self.model.named_parameters())
        self.optimizer = self.model.configure_optimizers(
            1.0,
            0.003,
            (0.9, 0.98),
            device,
        )
        before_batch = objects_for_stage(self.base.graph, 0, "before_batch")
        input_rows = [row for row in before_batch if row["role"] == "training_batch_inputs"]
        target_rows = [row for row in before_batch if row["role"] == "training_batch_targets"]
        require(len(input_rows) == len(target_rows) == 1, "CST_TRAINING_BATCH_NOT_UNIQUE")
        self.training_input_row = input_rows[0]
        self.training_target_row = target_rows[0]
        self.training_inputs = load_tensor(self.bundle, input_rows[0]).to(device)
        self.training_targets = load_tensor(self.bundle, target_rows[0]).to(device)
        require(self.training_inputs.shape == self.training_targets.shape, "CST_BATCH_SHAPE_MISMATCH")
        self._initial_hook_counts = self._hook_counts()

    def close(self) -> None:
        self.base.close()

    def _hook_counts(self) -> dict[str, int]:
        return {
            name: len(module._forward_hooks)
            for name, module in self.base.component_modules().items()
        }

    def source_rows(
        self,
        optimizer_step: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        rows = objects_for_stage(self.base.graph, optimizer_step, "after_optimizer_step")
        parameters = unique_role_objects(rows, "parameter_version")
        optimizer = unique_role_objects(rows, "optimizer_state")
        require(set(parameters) == set(self.named_parameters), "CST_SOURCE_PARAMETER_SET_MISMATCH")
        require(
            set(optimizer)
            == {
                f"{name}.{key}"
                for name in self.named_parameters
                for key in ("step", "exp_avg", "exp_avg_sq")
            },
            "CST_SOURCE_OPTIMIZER_SET_MISMATCH",
        )
        return parameters, optimizer

    def load_source_state(self, optimizer_step: int) -> tuple[StateSnapshot, dict[str, Any]]:
        parameter_rows, optimizer_rows = self.source_rows(optimizer_step)
        parameters = {
            name: load_tensor(self.bundle, row)
            for name, row in parameter_rows.items()
        }
        optimizer = {
            name: {
                key: load_tensor(self.bundle, optimizer_rows[f"{name}.{key}"])
                for key in ("step", "exp_avg", "exp_avg_sq")
            }
            for name in self.named_parameters
        }
        state = StateSnapshot(parameters, optimizer)
        self.restore(state)
        actual = self.snapshot().commitment()
        expected_parameters = {name: row["content_sha256"] for name, row in parameter_rows.items()}
        expected_optimizer = {name: row["content_sha256"] for name, row in optimizer_rows.items()}
        require(actual["parameters"] == expected_parameters, "CST_SOURCE_PARAMETER_RESTORE_HASH_MISMATCH")
        require(actual["optimizer"] == expected_optimizer, "CST_SOURCE_OPTIMIZER_RESTORE_HASH_MISMATCH")
        return state, {
            "optimizer_object_ids": {name: row["object_id"] for name, row in optimizer_rows.items()},
            "optimizer_sha256": expected_optimizer,
            "parameter_object_ids": {name: row["object_id"] for name, row in parameter_rows.items()},
            "parameter_sha256": expected_parameters,
        }

    def snapshot(self) -> StateSnapshot:
        parameters = _clone_tensor_map(self.named_parameters)
        optimizer: dict[str, dict[str, torch.Tensor]] = {}
        for name, parameter in self.named_parameters.items():
            source = self.optimizer.state.get(parameter)
            require(source is not None, f"CST_OPTIMIZER_STATE_MISSING:{name}")
            optimizer[name] = {
                key: source[key].detach().contiguous().cpu().clone()
                for key in ("step", "exp_avg", "exp_avg_sq")
            }
        return StateSnapshot(parameters, optimizer)

    def restore(self, state: StateSnapshot) -> None:
        require(set(state.parameters) == set(self.named_parameters), "CST_RESTORE_PARAMETER_SET_MISMATCH")
        with torch.no_grad():
            for name, parameter in self.named_parameters.items():
                parameter.copy_(state.parameters[name].to(self.device))
        self.optimizer.state.clear()
        for name, parameter in self.named_parameters.items():
            child = state.optimizer[name]
            self.optimizer.state[parameter] = {
                key: child[key].detach().clone().to(self.device)
                for key in ("step", "exp_avg", "exp_avg_sq")
            }

    @staticmethod
    def derive_seed(contract_sha256: str, entry_id: str, optimizer_step: int, opportunity: int) -> int:
        raw = hashlib.sha256(
            f"{contract_sha256}\0{entry_id}\0{optimizer_step}\0{opportunity}".encode("utf-8")
        ).digest()
        return int.from_bytes(raw[:8], "big") % (2**31 - 1)

    def set_rng(self, seed: int) -> dict[str, Any]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return self.rng_commitment(seed)

    def rng_commitment(self, seed: int | None = None) -> dict[str, Any]:
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        numpy_state = np.random.get_state()
        material = {
            "cpu": tensor_sha256(cpu),
            "cuda": [tensor_sha256(value) for value in cuda],
            "numpy": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
            "numpy_position": int(numpy_state[2]),
            "python": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
            "seed": seed,
        }
        return {**material, "rng_sha256": hashlib.sha256(canonical_bytes(material)).hexdigest()}

    def train_step(self, *, execute_optimizer: bool, seed: int) -> StepEvidence:
        self.model.train()
        rng_before = self.set_rng(seed)
        pre = _clone_tensor_map(self.named_parameters)
        self.optimizer.zero_grad(set_to_none=True)
        _logits, loss = self.model(self.training_inputs, self.training_targets)
        loss.backward()
        raw = {
            name: parameter.grad.detach().contiguous().cpu().clone()
            for name, parameter in self.named_parameters.items()
        }
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        clipped = {
            name: parameter.grad.detach().contiguous().cpu().clone()
            for name, parameter in self.named_parameters.items()
        }
        if execute_optimizer:
            self.optimizer.step()
        updates = {
            name: parameter.detach().contiguous().cpu() - pre[name]
            for name, parameter in self.named_parameters.items()
        }
        rng_after = self.rng_commitment(seed)
        require(rng_before["rng_sha256"] == rng_after["rng_sha256"], "CST_UNDECLARED_STOCHASTIC_OPERATOR")
        if not execute_optimizer:
            require(all(bool(torch.count_nonzero(value) == 0) for value in updates.values()), "CST_SKIP_MUTATED_PARAMETER")
        return StepEvidence(
            execute_optimizer=execute_optimizer,
            loss=float(loss.detach().cpu()),
            raw_gradients=raw,
            clipped_gradients=clipped,
            parameter_updates=updates,
            total_gradient_norm=float(total_norm.detach().cpu()),
            rng_before=rng_before,
            rng_after=rng_after,
        )

    def compare_historical_next(self, state: StateSnapshot, optimizer_step: int) -> dict[str, Any]:
        if optimizer_step > 10000:
            return {
                "all_available_object_hashes_exact": None,
                "disposition": "HISTORICAL_NEXT_STATE_OUTSIDE_CAPTURED_10000_STEP_SCOPE",
                "optimizer_exact_count": 0,
                "optimizer_object_count": 0,
                "parameter_exact_count": 0,
                "parameter_object_count": 0,
                "source_payload_materialized": False,
            }
        parameter_rows, optimizer_rows = self.source_rows(optimizer_step)
        actual = state.commitment()
        expected_parameters = {name: row["content_sha256"] for name, row in parameter_rows.items()}
        expected_optimizer = {name: row["content_sha256"] for name, row in optimizer_rows.items()}
        parameter_matches = {
            name: actual["parameters"][name] == expected_parameters[name]
            for name in expected_parameters
        }
        optimizer_matches = {
            name: actual["optimizer"][name] == expected_optimizer[name]
            for name in expected_optimizer
        }
        return {
            "all_available_object_hashes_exact": all(parameter_matches.values()) and all(optimizer_matches.values()),
            "optimizer_exact_count": sum(optimizer_matches.values()),
            "optimizer_object_count": len(optimizer_matches),
            "optimizer_per_object_exact": optimizer_matches,
            "parameter_exact_count": sum(parameter_matches.values()),
            "parameter_object_count": len(parameter_matches),
            "parameter_per_object_exact": parameter_matches,
            "source_payload_materialized": all(row["materialized"] for row in [*parameter_rows.values(), *optimizer_rows.values()]),
        }

    def ungated_baseline_repeat(self) -> tuple[torch.Tensor, torch.Tensor]:
        previous_mode = self.model.training
        pre = self.snapshot().commitment()
        pre_rng = self.rng_commitment()
        self.model.eval()
        try:
            first = self.base.forward()
            second = self.base.forward()
            require(torch.equal(first, second), "CST_PRESTATE_BASELINE_NOT_BYTE_EXACT")
            return first, second
        finally:
            self.model.train(previous_mode)
            require(self._hook_counts() == self._initial_hook_counts, "CST_PRESTATE_HOOK_LEAK")
            require(self.snapshot().commitment() == pre, "CST_PRESTATE_BASELINE_MUTATED_STATE")
            require(self.rng_commitment()["rng_sha256"] == pre_rng["rng_sha256"], "CST_PRESTATE_BASELINE_MUTATED_RNG")

    def support_probe(self, evidence: StepEvidence) -> dict[str, Any]:
        previous_mode = self.model.training
        pre = self.snapshot().commitment()
        pre_rng = self.rng_commitment()
        self.model.eval()
        try:
            baseline_1 = self.base.forward()
            baseline_2 = self.base.forward()
            require(torch.equal(baseline_1, baseline_2), "CST_CSRG_BASELINE_NOT_BYTE_EXACT")
            forward_rows: list[dict[str, Any]] = []
            for gate, logits in [
                ((), baseline_1),
                ((), baseline_2),
                *[((component,), self.base.forward((component,))) for component in COMPONENTS],
                *[(pair, self.base.forward(pair)) for pair in COMPONENT_PAIRS],
            ]:
                outputs = decision_outputs(logits, self.base.validation_targets)
                forward_rows.append({"gate_components": list(gate), "logits": logits, **outputs})
            baseline_group = forward_rows[0]["group_q10_margin"]
            single = {
                component: forward_rows[2 + index]["group_q10_margin"]
                for index, component in enumerate(COMPONENTS)
            }
            pair = {
                pair_name: forward_rows[2 + len(COMPONENTS) + index]["group_q10_margin"]
                for index, pair_name in enumerate(COMPONENT_PAIRS)
            }
            necessity = np.stack(
                [np.maximum(0.0, baseline_group - single[name]) for name in COMPONENTS]
            )
            backup = np.stack(
                [
                    np.maximum(
                        0.0,
                        baseline_group
                        - pair[pair_name]
                        - necessity[COMPONENTS.index(pair_name[0])]
                        - necessity[COMPONENTS.index(pair_name[1])],
                    )
                    for pair_name in COMPONENT_PAIRS
                ]
            )
            single_slack = np.min(np.stack([single[name] for name in COMPONENTS]), axis=0)
            double_slack = np.min(np.stack([pair[name] for name in COMPONENT_PAIRS]), axis=0)
            total = necessity.sum(axis=0)
            defined = total > 0.0
            allocation = np.zeros_like(necessity, dtype=np.float64)
            allocation[:, defined] = necessity[:, defined] / total[defined]
            concentration = np.sum(allocation * allocation, axis=0)
            effective = np.full(23, np.nan, dtype=np.float64)
            effective[defined] = 1.0 / concentration[defined]
            loads: dict[str, Any] = {}
            snapshot = self.snapshot()
            for component, prefix in COMPONENT_PREFIXES.items():
                names = [name for name in self.named_parameters if name.startswith(prefix)]
                require(bool(names), f"CST_COMPONENT_EMPTY:{component}")
                preconditioned = [
                    snapshot.optimizer[name]["exp_avg"]
                    / (torch.sqrt(snapshot.optimizer[name]["exp_avg_sq"]) + 1e-8)
                    for name in names
                ]
                loads[component] = {
                    "clipped_gradient_rms": _rms(evidence.clipped_gradients[name] for name in names),
                    "exp_avg_rms": _rms(snapshot.optimizer[name]["exp_avg"] for name in names),
                    "exp_avg_sq_sqrt_mean": math.sqrt(
                        sum(float(snapshot.optimizer[name]["exp_avg_sq"].to(torch.float64).sum()) for name in names)
                        / sum(snapshot.optimizer[name]["exp_avg_sq"].numel() for name in names)
                    ),
                    "parameter_rms": _rms(snapshot.parameters[name] for name in names),
                    "preconditioned_rms": _rms(preconditioned),
                    "raw_gradient_rms": _rms(evidence.raw_gradients[name] for name in names),
                }
            predictions = forward_rows[0]["predictions"]
            expected = self.base.validation_targets[:, -1].numpy()
            result = {
                "actual_forward_count": 12,
                "baseline_byte_exact": True,
                "capability_accuracy": float(np.mean(predictions == expected)),
                "component_loads": loads,
                "double_failure_slack": double_slack,
                "effective_support": effective,
                "forward_rows": forward_rows,
                "necessity": necessity,
                "pair_backup": backup,
                "single_failure_slack": single_slack,
                "support_allocation": allocation,
                "support_concentration": concentration,
                "undefined_effective_support_groups": np.flatnonzero(~defined),
            }
            return result
        finally:
            self.model.train(previous_mode)
            require(self._hook_counts() == self._initial_hook_counts, "CST_CSRG_HOOK_LEAK")
            require(self.snapshot().commitment() == pre, "CST_CSRG_MUTATED_TRAINING_STATE")
            require(self.rng_commitment()["rng_sha256"] == pre_rng["rng_sha256"], "CST_CSRG_MUTATED_RNG")
