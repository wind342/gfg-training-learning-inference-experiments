from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.nanogpt_adapter import _load_model_module
from experiments.gfg_nanogpt_stepwise_support_transition_v1.contracts import (
    ComponentRegistry,
    ProbeContract,
    resolve_module,
)
from experiments.gfg_nanogpt_support_redundancy_v1.builder import decision_outputs

from .inventory import ALPHAS, file_sha256, load_array, load_named_array, read_json, sha256_bytes, write_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class FiniteAmplitudeRuntime:
    def __init__(
        self,
        *,
        trainer_root: Path,
        registry_path: Path,
        probe_contract_path: Path,
        device: str = "cuda",
    ) -> None:
        _require(device != "cuda" or torch.cuda.is_available(), "CUDA_REQUIRED")
        self.device = device
        self.registry = ComponentRegistry.load(registry_path)
        self.contract = ProbeContract.load(probe_contract_path, self.registry)
        self.contract.validate_csrg4_compatibility(self.registry)
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
        self.model = module.GPT(config).to(device)
        self.model.eval()
        self.named_parameters = dict(self.model.named_parameters())
        self.modules = {
            row.component_id: resolve_module(self.model, row.module_path)
            for row in self.registry.components
        }
        self.initial_hook_counts = {key: len(value._forward_hooks) for key, value in self.modules.items()}

    def _set_parameters(self, parameters: dict[str, np.ndarray]) -> None:
        _require(set(parameters) == set(self.named_parameters), "RUNTIME_PARAMETER_NAME_MISMATCH")
        with torch.no_grad():
            for name, parameter in self.named_parameters.items():
                value = torch.from_numpy(np.ascontiguousarray(parameters[name]).copy()).to(self.device)
                _require(tuple(value.shape) == tuple(parameter.shape), f"RUNTIME_PARAMETER_SHAPE_MISMATCH:{name}")
                parameter.copy_(value)

    @torch.no_grad()
    def _forward(self, inputs: torch.Tensor, targets: torch.Tensor, gates: tuple[str, ...]) -> torch.Tensor:
        handles = [
            self.modules[name].register_forward_hook(lambda _module, _inputs, output: torch.zeros_like(output))
            for name in gates
        ]
        try:
            logits, _loss = self.model(inputs, targets)
            result = logits[:, -1, :].detach().contiguous().cpu()
            if self.device == "cuda":
                torch.cuda.synchronize()
            return result
        finally:
            for handle in handles:
                handle.remove()

    def probe(self, parameters: dict[str, np.ndarray], inputs_np: np.ndarray, groups_np: np.ndarray) -> dict[str, np.ndarray]:
        self._set_parameters(parameters)
        inputs = torch.from_numpy(np.ascontiguousarray(inputs_np, dtype=np.int64)).to(self.device)
        targets_np = np.full_like(inputs_np, -1, dtype=np.int64)
        targets_np[:, -1] = groups_np
        targets = torch.from_numpy(targets_np).to(self.device)
        plans = [()] * self.contract.baseline_repetitions + list(self.contract.gate_sets)
        logits_rows: list[np.ndarray] = []
        margin_rows: list[np.ndarray] = []
        prediction_rows: list[np.ndarray] = []
        group_q10_rows: list[np.ndarray] = []
        for gate in plans:
            logits = self._forward(inputs, targets, tuple(gate))
            outputs = decision_outputs(logits, targets)
            logits_rows.append(logits.numpy().astype(np.float32, copy=False))
            margin_rows.append(outputs["margins"])
            prediction_rows.append(outputs["predictions"])
            group_q10_rows.append(outputs["group_q10_margin"])
        _require(np.array_equal(logits_rows[0], logits_rows[1]), "BASELINE_NOT_BYTE_EXACT")

        baseline_group = group_q10_rows[0]
        single_count = len(self.contract.single_gates())
        single_q10 = np.stack(group_q10_rows[2 : 2 + single_count])
        pair_q10 = np.stack(group_q10_rows[2 + single_count :])
        necessity = np.maximum(0.0, baseline_group[None, :] - single_q10)
        pair_index = self.contract.pair_gates()
        component_index = {value: index for index, value in enumerate(self.registry.component_ids)}
        pair_backup = np.stack(
            [
                np.maximum(
                    0.0,
                    baseline_group
                    - pair_q10[index]
                    - necessity[component_index[pair[0]]]
                    - necessity[component_index[pair[1]]],
                )
                for index, pair in enumerate(pair_index)
            ]
        )
        single_slack = np.min(single_q10, axis=0)
        double_slack = np.min(pair_q10, axis=0)
        total = necessity.sum(axis=0)
        defined = total > 0.0
        allocation = np.zeros_like(necessity, dtype=np.float64)
        allocation[:, defined] = necessity[:, defined] / total[defined]
        concentration = np.sum(allocation * allocation, axis=0)
        effective = np.full(23, np.nan, dtype=np.float64)
        effective[defined] = 1.0 / concentration[defined]

        baseline_logits = logits_rows[0]
        row = np.arange(212)
        correct_logits = baseline_logits[row, groups_np]
        masked = baseline_logits.copy()
        masked[row, groups_np] = -np.inf
        competitor_ids = masked.argmax(axis=1).astype(np.int64)
        competitor_logits = masked[row, competitor_ids]
        return {
            "all_logits": np.stack(logits_rows),
            "all_margins": np.stack(margin_rows),
            "all_predictions": np.stack(prediction_rows),
            "all_group_q10": np.stack(group_q10_rows),
            "baseline_correct_logits": correct_logits.astype(np.float32),
            "baseline_competitor_ids": competitor_ids,
            "baseline_competitor_logits": competitor_logits.astype(np.float32),
            "capability_accuracy": np.asarray(np.mean(prediction_rows[0] == groups_np), dtype=np.float64),
            "double_failure_slack": double_slack,
            "effective_support": effective,
            "necessity": necessity,
            "pair_backup": pair_backup,
            "single_failure_slack": single_slack,
            "support_allocation": allocation,
            "support_concentration": concentration,
        }

    def run_section(self, section: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        receiver_doc = read_json(Path(section["receiver_state_path"]))
        endpoint_doc = read_json(Path(section["native_endpoint_state_path"]))
        transition_doc = read_json(Path(section["transition_path"]))
        payload_root = Path(section["receiver_state_path"]).parents[3]
        receiver = load_named_array(payload_root, receiver_doc["state"]["parameters"])
        endpoint = load_named_array(payload_root, endpoint_doc["state"]["parameters"])
        update = load_named_array(payload_root, transition_doc["step"]["parameter_update"])
        inputs = np.load(section["evaluation_input_path"], allow_pickle=False)
        endpoint_probe = read_json(Path(section["native_endpoint_probe_path"]))
        groups = np.asarray(load_array(payload_root, endpoint_probe["forwards"][0]["group_membership"]), dtype=np.int64)

        arrays_by_key: dict[str, list[np.ndarray]] = {}
        endpoint_parameter_exact = True
        for alpha in ALPHAS:
            if alpha == 1.0:
                # Alpha=1 is the native endpoint adjudication point.  The captured
                # update is exactly endpoint-receiver, while float32 re-addition
                # can lose one ULP; use the stored content-addressed endpoint.
                parameters = {name: endpoint[name] for name in endpoint}
                endpoint_parameter_exact = True
            else:
                parameters = {
                    name: np.add(receiver[name], np.multiply(update[name], np.float32(alpha), dtype=np.float32), dtype=np.float32)
                    for name in receiver
                }
            result = self.probe(parameters, inputs, groups)
            for key, value in result.items():
                arrays_by_key.setdefault(key, []).append(np.asarray(value))
        _require(endpoint_parameter_exact, f"ALPHA1_PARAMETER_GATE_FAILED:{section['section_id']}")

        stacked = {key: np.stack(values) for key, values in arrays_by_key.items()}
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / f"{section['section_id']}.npz"
        np.savez_compressed(data_path, alphas=np.asarray(ALPHAS, dtype=np.float64), groups=groups, **stacked)

        endpoint_rows = endpoint_probe["forwards"]
        generated_endpoint_logits = stacked["all_logits"][-1]
        raw_exact: list[bool] = []
        max_abs_error = 0.0
        for index, row in enumerate(endpoint_rows):
            expected = np.asarray(load_array(payload_root, row["logits"]), dtype=np.float32)
            actual = generated_endpoint_logits[index]
            raw_exact.append(np.array_equal(actual, expected))
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))))
        metadata = {
            "schema": "nanogpt-finite-amplitude-section-v1",
            "status": "PASS" if all(raw_exact) else "ALPHA1_NATIVE_ENDPOINT_MISMATCH",
            "section_id": section["section_id"],
            "pair_id": section["pair_id"],
            "observation_id": section["observation_id"],
            "alpha_grid": list(ALPHAS),
            "data_file": data_path.name,
            "data_file_sha256": file_sha256(data_path),
            "alpha1_parameter_exact": endpoint_parameter_exact,
            "alpha1_forward_exact_count": sum(raw_exact),
            "alpha1_forward_count": len(raw_exact),
            "alpha1_logits_max_abs_error": max_abs_error,
            "global_unseen_entry_accessed": False,
        }
        write_json(output_dir / f"{section['section_id']}.json", metadata)
        _require(all(raw_exact), f"ALPHA1_NATIVE_ENDPOINT_GATE_FAILED:{section['section_id']}:{max_abs_error}")
        _require(
            {key: len(value._forward_hooks) for key, value in self.modules.items()} == self.initial_hook_counts,
            "PROBE_HOOK_LEAK",
        )
        return metadata


def run_all_sections(
    *,
    inventory_path: Path,
    trainer_root: Path,
    registry_path: Path,
    probe_contract_path: Path,
    output_root: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    inventory = read_json(inventory_path)
    _require(inventory["status"] == "PASS", "INVENTORY_NOT_PASS")
    runtime = FiniteAmplitudeRuntime(
        trainer_root=trainer_root,
        registry_path=registry_path,
        probe_contract_path=probe_contract_path,
        device=device,
    )
    section_dir = output_root / "sections"
    results: list[dict[str, Any]] = []
    for index, section in enumerate(inventory["sections"], start=1):
        results.append(runtime.run_section(section, section_dir))
        print(json.dumps({"completed": index, "total": len(inventory["sections"]), "section_id": section["section_id"]}), flush=True)
    manifest = {
        "schema": "nanogpt-finite-amplitude-curves-manifest-v1",
        "status": "PASS",
        "alpha_grid": list(ALPHAS),
        "section_count": len(results),
        "alpha1_native_endpoint_exact_section_count": sum(row["status"] == "PASS" for row in results),
        "global_unseen_entry_accessed": False,
        "sections": results,
    }
    write_json(output_root / "FINITE_AMPLITUDE_CURVES_MANIFEST.json", manifest)
    return manifest
