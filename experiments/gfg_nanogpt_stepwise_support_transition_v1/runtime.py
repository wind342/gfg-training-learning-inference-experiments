from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import require
from experiments.gfg_nanogpt_support_redundancy_v1.builder import decision_outputs
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import (
    load_tensor,
    objects_for_stage,
)
from experiments.gfg_nanogpt_support_transition_v1.runtime import (
    CausalTrainingRuntime,
    StateSnapshot,
)

from .contracts import (
    ComponentRegistry,
    ProbeContract,
    component_parameter_names,
    resolve_module,
)


def _clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().contiguous().cpu().clone() for name, value in values.items()}


def _rms(values) -> float:
    total = 0.0
    count = 0
    for value in values:
        child = value.detach().to(torch.float64)
        total += float(torch.sum(child * child))
        count += int(child.numel())
    require(count > 0, "SST_RMS_EMPTY")
    return math.sqrt(total / count)


@dataclass
class BatchEvidence:
    optimizer_step: int
    inputs: torch.Tensor
    targets: torch.Tensor
    selection_order: torch.Tensor | None
    selection_order_disposition: dict[str, Any] | None
    source_rows: dict[str, dict[str, Any]]


@dataclass
class StepEvidence:
    optimizer_step: int
    execute_optimizer: bool
    training_logits: torch.Tensor
    activation_outputs: dict[str, torch.Tensor]
    loss: float
    raw_gradients: dict[str, torch.Tensor]
    clipped_gradients: dict[str, torch.Tensor]
    total_gradient_norm: float
    parameter_updates: dict[str, torch.Tensor]
    nominal_weight_decay_updates: dict[str, torch.Tensor]
    adaptive_update_residuals: dict[str, torch.Tensor]
    exp_avg_deltas: dict[str, torch.Tensor]
    exp_avg_sq_deltas: dict[str, torch.Tensor]
    adam_step_deltas: dict[str, torch.Tensor]
    post_preconditioned_directions: dict[str, torch.Tensor]
    rng_before: dict[str, Any]
    rng_after: dict[str, Any]
    optimizer_config: dict[str, Any]


class StepwiseTrainingRuntime(CausalTrainingRuntime):
    def __init__(
        self,
        bundle: Path,
        trainer_root: Path,
        registry: ComponentRegistry,
        probe_contract: ProbeContract,
        *,
        device: str = "cuda",
    ) -> None:
        super().__init__(bundle, trainer_root, device=device)
        self.registry = registry
        self.probe_contract = probe_contract
        self.component_modules = {
            row.component_id: resolve_module(self.model, row.module_path)
            for row in registry.components
        }
        self.component_parameters = component_parameter_names(self.named_parameters, registry)
        self._registry_hook_counts = {
            name: len(module._forward_hooks) for name, module in self.component_modules.items()
        }

    def load_batch(self, optimizer_step: int) -> BatchEvidence:
        rows = objects_for_stage(self.base.graph, optimizer_step, "before_batch")
        selected: dict[str, dict[str, Any]] = {}
        for role in ("training_batch_inputs", "training_batch_targets"):
            matches = [row for row in rows if row["role"] == role]
            require(len(matches) == 1, f"SST_BATCH_ROLE_NOT_UNIQUE:{optimizer_step}:{role}")
            selected[role] = matches[0]
        order_matches = [row for row in rows if row["role"] == "batch_selection_order"]
        require(len(order_matches) <= 1, f"SST_BATCH_ROLE_NOT_UNIQUE:{optimizer_step}:batch_selection_order")
        if order_matches:
            selected["batch_selection_order"] = order_matches[0]
        inputs = load_tensor(self.bundle, selected["training_batch_inputs"])
        targets = load_tensor(self.bundle, selected["training_batch_targets"])
        require(inputs.shape == targets.shape, "SST_BATCH_SHAPE_MISMATCH")
        order: torch.Tensor | None = None
        disposition: dict[str, Any] | None = None
        if "batch_selection_order" in selected:
            order = load_tensor(self.bundle, selected["batch_selection_order"])
            require(int(inputs.shape[0]) == int(order.numel()), "SST_BATCH_ORDER_LENGTH_MISMATCH")
        else:
            disposition = {
                "outcome_kind": "ExplicitDisposition",
                "disposition": "SOURCE_BATCH_SELECTION_ORDER_NOT_CAPTURED_IN_PARTICIPANT_GFG",
                "optimizer_step": optimizer_step,
                "reconstruction_or_guess_used": False,
            }
        return BatchEvidence(
            optimizer_step=optimizer_step,
            inputs=inputs,
            targets=targets,
            selection_order=order,
            selection_order_disposition=disposition,
            source_rows=selected,
        )

    def compare_historical_state(self, state: StateSnapshot, optimizer_step: int) -> dict[str, Any]:
        parameter_rows, optimizer_rows = self.source_rows(optimizer_step)
        parameter_errors: dict[str, float] = {}
        optimizer_errors: dict[str, float] = {}
        parameter_exact: dict[str, bool] = {}
        optimizer_exact: dict[str, bool] = {}
        for name, row in parameter_rows.items():
            expected = load_tensor(self.bundle, row)
            actual = state.parameters[name]
            parameter_exact[name] = bool(torch.equal(actual, expected))
            parameter_errors[name] = float(torch.max(torch.abs(actual.to(torch.float64) - expected.to(torch.float64))))
        for semantic_name, row in optimizer_rows.items():
            parameter_name, key = semantic_name.rsplit(".", 1)
            expected = load_tensor(self.bundle, row)
            actual = state.optimizer[parameter_name][key]
            optimizer_exact[semantic_name] = bool(torch.equal(actual, expected))
            optimizer_errors[semantic_name] = float(torch.max(torch.abs(actual.to(torch.float64) - expected.to(torch.float64))))
        all_exact = all(parameter_exact.values()) and all(optimizer_exact.values())
        return {
            "historical_stepwise_replay_exact": all_exact,
            "runtime_deterministic_reexecution": True,
            "parameter_exact_count": sum(parameter_exact.values()),
            "parameter_object_count": len(parameter_exact),
            "optimizer_exact_count": sum(optimizer_exact.values()),
            "optimizer_object_count": len(optimizer_exact),
            "parameter_max_abs_error": max(parameter_errors.values(), default=0.0),
            "optimizer_max_abs_error": max(optimizer_errors.values(), default=0.0),
            "parameter_per_object_exact": parameter_exact,
            "optimizer_per_object_exact": optimizer_exact,
            "parameter_per_object_max_abs_error": parameter_errors,
            "optimizer_per_object_max_abs_error": optimizer_errors,
            "tolerance_repair_used": False,
        }

    def state_summary(self, state: StateSnapshot | None = None) -> dict[str, Any]:
        snapshot = state or self.snapshot()
        global_preconditioned = [
            snapshot.optimizer[name]["exp_avg"]
            / (torch.sqrt(snapshot.optimizer[name]["exp_avg_sq"]) + 1e-8)
            for name in snapshot.optimizer
        ]
        adam_steps = [float(snapshot.optimizer[name]["step"].reshape(-1)[0]) for name in snapshot.optimizer]
        component_loads: dict[str, Any] = {}
        for component_id, names in self.component_parameters.items():
            component_preconditioned = [
                snapshot.optimizer[name]["exp_avg"]
                / (torch.sqrt(snapshot.optimizer[name]["exp_avg_sq"]) + 1e-8)
                for name in names
            ]
            component_loads[component_id] = {
                "parameter_rms": _rms(snapshot.parameters[name] for name in names),
                "exp_avg_rms": _rms(snapshot.optimizer[name]["exp_avg"] for name in names),
                "exp_avg_sq_sqrt_mean": math.sqrt(
                    sum(float(snapshot.optimizer[name]["exp_avg_sq"].to(torch.float64).sum()) for name in names)
                    / sum(snapshot.optimizer[name]["exp_avg_sq"].numel() for name in names)
                ),
                "preconditioned_rms": _rms(component_preconditioned),
            }
        group = self.optimizer.param_groups[0]
        return {
            "parameter_rms": _rms(snapshot.parameters.values()),
            "exp_avg_rms": _rms(snapshot.optimizer[name]["exp_avg"] for name in snapshot.optimizer),
            "exp_avg_sq_sqrt_mean": math.sqrt(
                sum(float(snapshot.optimizer[name]["exp_avg_sq"].to(torch.float64).sum()) for name in snapshot.optimizer)
                / sum(snapshot.optimizer[name]["exp_avg_sq"].numel() for name in snapshot.optimizer)
            ),
            "preconditioned_rms": _rms(global_preconditioned),
            "adam_step_min": min(adam_steps),
            "adam_step_max": max(adam_steps),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "betas": [float(value) for value in group["betas"]],
            "gradient_clip": 1.0,
            "component_loads": component_loads,
        }

    def train_actual_step(
        self,
        batch: BatchEvidence,
        *,
        execute_optimizer: bool,
        seed: int,
    ) -> StepEvidence:
        self.model.train()
        rng_before = self.set_rng(seed)
        pre = self.snapshot()
        self.optimizer.zero_grad(set_to_none=True)
        activations: dict[str, torch.Tensor] = {}
        handles = []
        for component_id, module in self.component_modules.items():
            def capture(_module, _inputs, output, *, key=component_id):
                require(isinstance(output, torch.Tensor), f"SST_COMPONENT_ACTIVATION_NOT_TENSOR:{key}")
                activations[key] = output.detach().contiguous().cpu().clone()
                return None

            handles.append(module.register_forward_hook(capture))
        try:
            logits, loss = self.model(batch.inputs.to(self.device), batch.targets.to(self.device))
        finally:
            for handle in handles:
                handle.remove()
        require(set(activations) == set(self.component_modules), "SST_COMPONENT_ACTIVATION_INCOMPLETE")
        require(
            {name: len(module._forward_hooks) for name, module in self.component_modules.items()}
            == self._registry_hook_counts,
            "SST_TRAINING_ACTIVATION_HOOK_LEAK",
        )
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
        post = self.snapshot()
        updates = {name: post.parameters[name] - pre.parameters[name] for name in pre.parameters}
        group = self.optimizer.param_groups[0]
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        nominal_decay = (
            {name: -learning_rate * weight_decay * pre.parameters[name] for name in pre.parameters}
            if execute_optimizer
            else {}
        )
        adaptive_residual = (
            {name: updates[name] - nominal_decay[name] for name in updates}
            if execute_optimizer
            else {}
        )
        exp_avg = {name: post.optimizer[name]["exp_avg"] - pre.optimizer[name]["exp_avg"] for name in pre.optimizer}
        exp_avg_sq = {name: post.optimizer[name]["exp_avg_sq"] - pre.optimizer[name]["exp_avg_sq"] for name in pre.optimizer}
        step_delta = {name: post.optimizer[name]["step"] - pre.optimizer[name]["step"] for name in pre.optimizer}
        preconditioned = {
            name: post.optimizer[name]["exp_avg"] / (torch.sqrt(post.optimizer[name]["exp_avg_sq"]) + 1e-8)
            for name in post.optimizer
        }
        rng_after = self.rng_commitment(seed)
        require(rng_before["rng_sha256"] == rng_after["rng_sha256"], "SST_UNDECLARED_STOCHASTIC_OPERATOR")
        if not execute_optimizer:
            require(all(bool(torch.count_nonzero(value) == 0) for value in updates.values()), "SST_SKIP_MUTATED_PARAMETER")
            require(all(bool(torch.count_nonzero(value) == 0) for value in exp_avg.values()), "SST_SKIP_MUTATED_EXP_AVG")
            require(all(bool(torch.count_nonzero(value) == 0) for value in exp_avg_sq.values()), "SST_SKIP_MUTATED_EXP_AVG_SQ")
            require(all(bool(torch.count_nonzero(value) == 0) for value in step_delta.values()), "SST_SKIP_MUTATED_ADAM_STEP")
        return StepEvidence(
            optimizer_step=batch.optimizer_step,
            execute_optimizer=execute_optimizer,
            training_logits=logits.detach().contiguous().cpu(),
            activation_outputs=activations,
            loss=float(loss.detach().cpu()),
            raw_gradients=raw,
            clipped_gradients=clipped,
            total_gradient_norm=float(total_norm.detach().cpu()),
            parameter_updates=updates,
            nominal_weight_decay_updates=nominal_decay,
            adaptive_update_residuals=adaptive_residual,
            exp_avg_deltas=exp_avg,
            exp_avg_sq_deltas=exp_avg_sq,
            adam_step_deltas=step_delta,
            post_preconditioned_directions=preconditioned,
            rng_before=rng_before,
            rng_after=rng_after,
            optimizer_config={
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "gradient_clip": 1.0,
            },
        )

    @torch.no_grad()
    def _forward_with_gates(self, gate_components: tuple[str, ...]) -> torch.Tensor:
        unknown = set(gate_components) - set(self.component_modules)
        require(not unknown, f"SST_UNKNOWN_COMPONENT_GATE:{sorted(unknown)}")
        handles = [
            self.component_modules[name].register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(output)
            )
            for name in gate_components
        ]
        try:
            logits, _loss = self.model(
                self.base.validation_inputs.to(self.device),
                self.base.validation_targets.to(self.device),
            )
            result = logits[:, -1, :].detach().cpu()
            if self.device == "cuda":
                torch.cuda.synchronize()
            return result
        finally:
            for handle in handles:
                handle.remove()

    def support_probe(self) -> dict[str, Any]:
        previous_mode = self.model.training
        pre = self.snapshot().commitment()
        pre_rng = self.rng_commitment()
        self.model.eval()
        try:
            forward_rows: list[dict[str, Any]] = []
            plans = [()] * self.probe_contract.baseline_repetitions + list(self.probe_contract.gate_sets)
            for gate in plans:
                logits = self._forward_with_gates(tuple(gate))
                outputs = decision_outputs(logits, self.base.validation_targets)
                forward_rows.append({"gate_components": list(gate), "logits": logits, **outputs})
            baseline_rows = forward_rows[: self.probe_contract.baseline_repetitions]
            require(all(torch.equal(baseline_rows[0]["logits"], row["logits"]) for row in baseline_rows[1:]), "SST_CSRG_BASELINE_NOT_BYTE_EXACT")
            by_gate = {
                tuple(row["gate_components"]): row
                for row in forward_rows[self.probe_contract.baseline_repetitions :]
            }
            baseline_group = baseline_rows[0]["group_q10_margin"]
            singles = self.probe_contract.single_gates()
            pairs = self.probe_contract.pair_gates()
            necessity = np.stack([np.maximum(0.0, baseline_group - by_gate[gate]["group_q10_margin"]) for gate in singles])
            single_index = {gate[0]: index for index, gate in enumerate(singles)}
            backup = np.stack(
                [
                    np.maximum(
                        0.0,
                        baseline_group
                        - by_gate[gate]["group_q10_margin"]
                        - necessity[single_index[gate[0]]]
                        - necessity[single_index[gate[1]]],
                    )
                    for gate in pairs
                ]
            )
            single_slack = np.min(np.stack([by_gate[gate]["group_q10_margin"] for gate in singles]), axis=0)
            double_slack = np.min(np.stack([by_gate[gate]["group_q10_margin"] for gate in pairs]), axis=0)
            total = necessity.sum(axis=0)
            defined = total > 0.0
            allocation = np.zeros_like(necessity, dtype=np.float64)
            allocation[:, defined] = necessity[:, defined] / total[defined]
            concentration = np.sum(allocation * allocation, axis=0)
            effective = np.full(self.probe_contract.target_group_count, np.nan, dtype=np.float64)
            effective[defined] = 1.0 / concentration[defined]
            snapshot = self.snapshot()
            loads: dict[str, Any] = {}
            for component_id, names in self.component_parameters.items():
                preconditioned = [
                    snapshot.optimizer[name]["exp_avg"]
                    / (torch.sqrt(snapshot.optimizer[name]["exp_avg_sq"]) + 1e-8)
                    for name in names
                ]
                loads[component_id] = {
                    "exp_avg_rms": _rms(snapshot.optimizer[name]["exp_avg"] for name in names),
                    "exp_avg_sq_sqrt_mean": math.sqrt(
                        sum(float(snapshot.optimizer[name]["exp_avg_sq"].to(torch.float64).sum()) for name in names)
                        / sum(snapshot.optimizer[name]["exp_avg_sq"].numel() for name in names)
                    ),
                    "parameter_rms": _rms(snapshot.parameters[name] for name in names),
                    "preconditioned_rms": _rms(preconditioned),
                }
            predictions = baseline_rows[0]["predictions"]
            expected = self.base.validation_targets[:, -1].numpy()
            return {
                "probe_contract_id": self.probe_contract.probe_contract_id,
                "probe_contract_sha256": self.probe_contract.source_sha256,
                "component_registry_id": self.registry.registry_id,
                "component_registry_sha256": self.registry.source_sha256,
                "component_ids": list(self.registry.component_ids),
                "pair_ids": [list(value) for value in pairs],
                "actual_forward_count": len(plans),
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
        finally:
            self.model.train(previous_mode)
            require(
                {name: len(module._forward_hooks) for name, module in self.component_modules.items()}
                == self._registry_hook_counts,
                "SST_CSRG_HOOK_LEAK",
            )
            require(self.snapshot().commitment() == pre, "SST_CSRG_MUTATED_TRAINING_STATE")
            require(self.rng_commitment()["rng_sha256"] == pre_rng["rng_sha256"], "SST_CSRG_MUTATED_RNG")
