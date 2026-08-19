from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import require
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.nanogpt_adapter import (
    _load_model_module,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import (
    TrainingGFG,
)

from .task_mapping import recover_cyclic_target_mapping


COMPONENTS = ("h0.attn", "h0.mlp", "h1.attn", "h1.mlp")
COMPONENT_PAIRS = tuple(
    (left, right)
    for index, left in enumerate(COMPONENTS)
    for right in COMPONENTS[index + 1 :]
)


def tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().contiguous().cpu().numpy()
    else:
        array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def objects_for_stage(
    graph: TrainingGFG,
    optimizer_step: int,
    stage: str,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for _row, block in graph.blocks(
        min_step=optimizer_step,
        max_step=optimizer_step,
        stage=stage,
    ):
        objects.extend(block["objects"])
    return objects


def unique_role_objects(
    objects: Iterable[dict[str, Any]],
    role: str,
) -> dict[str, dict[str, Any]]:
    selected = [row for row in objects if row["role"] == role]
    result = {str(row["name"]): row for row in selected}
    require(len(result) == len(selected), f"CSRG_DUPLICATE_OBJECT_NAME:{role}")
    return result


def load_tensor(bundle: Path, row: dict[str, Any]) -> torch.Tensor:
    require(bool(row["materialized"]), "CSRG_SOURCE_TENSOR_NOT_MATERIALIZED")
    prefix = "objects://"
    locator = str(row["locator"])
    require(locator.startswith(prefix), "CSRG_SOURCE_TENSOR_LOCATOR_INVALID")
    array = np.load(
        bundle / "tensor-objects" / locator[len(prefix) :],
        allow_pickle=False,
    )
    value = torch.from_numpy(array.copy())
    require(
        tensor_sha256(value) == row["content_sha256"],
        "CSRG_SOURCE_TENSOR_HASH_MISMATCH",
    )
    return value


@dataclass
class HistoricalRunRuntime:
    bundle: Path
    graph: TrainingGFG
    model: torch.nn.Module
    device: str
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor
    target_mapping_certificate: dict[str, Any]
    source_training_inputs: dict[str, Any]
    source_training_targets: dict[str, Any]
    source_validation_inputs: dict[str, Any]

    @classmethod
    def open(
        cls,
        bundle: Path,
        trainer_root: Path,
        *,
        device: str = "cuda",
        reference_step: int = 100,
    ) -> HistoricalRunRuntime:
        bundle = bundle.resolve()
        capture_manifest = __import__("json").loads(
            (bundle / "capture_manifest.json").read_text(encoding="utf-8")
        )
        graph = TrainingGFG(bundle / capture_manifest["database"])
        before_batch = objects_for_stage(graph, 0, "before_batch")
        train_input_row = next(
            row for row in before_batch if row["role"] == "training_batch_inputs"
        )
        train_target_row = next(
            row for row in before_batch if row["role"] == "training_batch_targets"
        )
        train_inputs = load_tensor(bundle, train_input_row)
        train_targets = load_tensor(bundle, train_target_row)

        validation_embedding = objects_for_stage(
            graph,
            reference_step,
            "evaluation_validation:token_embedding",
        )
        validation_input_rows = [
            row
            for row in validation_embedding
            if row["role"] == "layer_input"
            and row["name"] == "token_embedding.input.0"
        ]
        require(
            len(validation_input_rows) == 1,
            "CSRG_VALIDATION_INPUT_SOURCE_NOT_UNIQUE",
        )
        validation_input_row = validation_input_rows[0]
        validation_inputs = load_tensor(bundle, validation_input_row)
        recovered_targets, certificate = recover_cyclic_target_mapping(
            train_inputs.numpy(),
            train_targets.numpy(),
            validation_inputs.numpy(),
        )
        validation_targets = torch.from_numpy(recovered_targets.copy())

        evaluation_objects = objects_for_stage(graph, reference_step, "evaluation")
        explicit_targets = [
            row
            for row in evaluation_objects
            if row["role"] == "evaluation_validation_targets"
        ]
        certificate["explicit_native_target_present"] = bool(explicit_targets)
        if explicit_targets:
            require(
                len(explicit_targets) == 1,
                "CSRG_EXPLICIT_VALIDATION_TARGET_NOT_UNIQUE",
            )
            explicit = load_tensor(bundle, explicit_targets[0])
            require(
                torch.equal(explicit, validation_targets),
                "CSRG_DERIVED_TARGET_MISMATCHES_EXPLICIT_NATIVE_TARGET",
            )
            certificate["explicit_native_target_exact"] = True
            certificate["explicit_native_target_object_id"] = explicit_targets[0][
                "object_id"
            ]

        if device == "cuda":
            require(torch.cuda.is_available(), "CSRG_CUDA_REQUIRED")
        module = _load_model_module(trainer_root.resolve())
        model_config = module.GPTConfig(
            block_size=int(train_inputs.shape[1]),
            vocab_size=int(
                max(train_inputs.max().item(), train_targets[train_targets >= 0].max().item())
                + 1
            ),
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
            bias=False,
        )
        model = module.GPT(model_config).to(device)
        model.eval()
        return cls(
            bundle=bundle,
            graph=graph,
            model=model,
            device=device,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            target_mapping_certificate=certificate,
            source_training_inputs=train_input_row,
            source_training_targets=train_target_row,
            source_validation_inputs=validation_input_row,
        )

    def close(self) -> None:
        self.graph.close()

    def load_checkpoint(self, optimizer_step: int) -> dict[str, dict[str, Any]]:
        objects = objects_for_stage(
            self.graph,
            optimizer_step,
            "after_optimizer_step",
        )
        parameter_rows = unique_role_objects(objects, "parameter_version")
        named_parameters = dict(self.model.named_parameters())
        require(
            set(named_parameters) == set(parameter_rows),
            "CSRG_CHECKPOINT_PARAMETER_SET_MISMATCH",
        )
        with torch.no_grad():
            for name, parameter in named_parameters.items():
                parameter.copy_(load_tensor(self.bundle, parameter_rows[name]).to(self.device))
        return parameter_rows

    def expected_evaluation(
        self,
        optimizer_step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        objects = objects_for_stage(self.graph, optimizer_step, "evaluation")
        logits = [row for row in objects if row["role"] == "validation_logits"]
        metrics = [row for row in objects if row["role"] == "capability_evaluation"]
        require(len(logits) == 1, "CSRG_EXPECTED_VALIDATION_LOGITS_NOT_UNIQUE")
        require(len(metrics) == 1, "CSRG_EXPECTED_CAPABILITY_METRIC_NOT_UNIQUE")
        return logits[0], metrics[0]

    def component_modules(self) -> dict[str, torch.nn.Module]:
        blocks = self.model.transformer.h
        return {
            "h0.attn": blocks[0].attn,
            "h0.mlp": blocks[0].mlp,
            "h1.attn": blocks[1].attn,
            "h1.mlp": blocks[1].mlp,
        }

    @torch.no_grad()
    def forward(self, gated_components: Iterable[str] = ()) -> torch.Tensor:
        components = tuple(gated_components)
        unknown = set(components) - set(COMPONENTS)
        require(not unknown, f"CSRG_UNKNOWN_COMPONENT_GATE:{sorted(unknown)}")
        modules = self.component_modules()
        handles = [
            modules[name].register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(output)
            )
            for name in components
        ]
        try:
            logits, _loss = self.model(
                self.validation_inputs.to(self.device),
                self.validation_targets.to(self.device),
            )
            result = logits[:, -1, :].detach().cpu()
            if self.device == "cuda":
                torch.cuda.synchronize()
            return result
        finally:
            for handle in handles:
                handle.remove()

    def accuracy(self, logits: torch.Tensor) -> float:
        predicted = logits.argmax(dim=-1)
        expected = self.validation_targets[:, -1]
        return float((predicted == expected).float().mean())
