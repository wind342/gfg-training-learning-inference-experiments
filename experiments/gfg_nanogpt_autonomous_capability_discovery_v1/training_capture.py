from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import zlib

import numpy as np
import torch

from .common import canonical_bytes, payload_sha256, write_json
from .task_generator import TokenizedTask, participant_description


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_blocks (
    block_ordinal INTEGER PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    optimizer_step INTEGER NOT NULL,
    stage TEXT NOT NULL,
    prior_block_sha256 TEXT,
    payload_sha256 TEXT NOT NULL,
    block_sha256 TEXT NOT NULL UNIQUE,
    object_count INTEGER NOT NULL,
    occurrence_count INTEGER NOT NULL,
    fact_count INTEGER NOT NULL,
    explicit_edge_count INTEGER NOT NULL,
    payload_zlib BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_block_step
    ON graph_blocks(optimizer_step, stage);
CREATE TABLE IF NOT EXISTS evaluations (
    optimizer_step INTEGER PRIMARY KEY,
    parameter_version INTEGER NOT NULL,
    train_accuracy REAL NOT NULL,
    validation_accuracy REAL NOT NULL,
    loss REAL NOT NULL,
    occurrence_id TEXT NOT NULL,
    metric_object_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_index INTEGER PRIMARY KEY,
    start_block_ordinal INTEGER NOT NULL,
    end_block_ordinal INTEGER NOT NULL,
    end_optimizer_step INTEGER NOT NULL,
    prior_chunk_sha256 TEXT,
    chunk_sha256 TEXT NOT NULL,
    row_counts_json TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ObjectRef:
    object_id: str
    content_sha256: str
    semantic_key: str
    role: str


def _json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def _tensor_bytes(tensor: torch.Tensor) -> tuple[np.ndarray, bytes]:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        array = value.view(torch.int16).numpy()
    else:
        array = value.numpy()
    return array, array.tobytes(order="C")


def decode_block(blob: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


class TrainingGFGCapture:
    """Losslessly template-compressed capture of real nanoGPT training.

    Every tensor instance has an identity, exact content hash, dtype, shape,
    role and locator. Repeated graph structure is stored as reversible fact
    blocks rather than millions of redundant SQLite rows. Every fact block has
    exactly one outcome and an outcome-specific ordered source-role list; each
    expanded atomic fact receives a deterministic identity.
    """

    def __init__(
        self,
        *,
        run_id: str,
        run_directory: Path,
        task: TokenizedTask,
        evaluation_interval: int,
        materialization_interval: int,
        chunk_steps: int,
    ) -> None:
        self.run_id = run_id
        self.run_directory = run_directory.resolve()
        self.task = task
        self.evaluation_interval = evaluation_interval
        self.materialization_interval = materialization_interval
        self.chunk_steps = chunk_steps
        self.database_path = self.run_directory / "participant_gfg.sqlite3"
        self.object_root = self.run_directory / "tensor-objects"
        self.object_root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.executescript(SCHEMA)
        self._put_metadata("run_id", run_id)
        self._put_metadata("participant_task", participant_description(task))
        self._put_metadata(
            "capture_profile",
            {
                "actual_batch_composition": True,
                "actual_gradients": True,
                "actual_layer_boundaries": True,
                "actual_optimizer_state": True,
                "actual_parameter_versions": True,
                "compression": ("reversible outcome-specific source-role fact blocks"),
                "content_hash_every_tensor_instance": True,
                "materialization_interval": materialization_interval,
                "no_approximate_temporal_join": True,
                "nonmaterialized_tensor_access": (
                    "deterministic replay from exact initial state, batch "
                    "identity and occurrence identity"
                ),
                "schema": "nanogpt-training-gfg-block-store-v2",
            },
        )
        self.connection.commit()

        self._block_ordinal = 0
        self._occurrence_ordinal = 0
        self._last_block_sha: str | None = None
        self._last_occurrence: str | None = None
        self._chunk_index = 0
        self._chunk_start_block = 0
        self._current_step = 0
        self._stage = "initialization"
        self._group: dict[str, Any] | None = None
        self._model_bound = False
        self._module_handles: list[Any] = []
        self._module_parameter_refs: dict[str, list[ObjectRef]] = {}
        self._parameter_refs: dict[str, ObjectRef] = {}
        self._optimizer_state_refs: dict[str, dict[str, ObjectRef]] = {}
        self._gradient_refs: dict[str, ObjectRef] = {}
        self._gradient_roles: dict[str, str] = {}
        self._pending_optimizer_sources: dict[
            str,
            tuple[
                ObjectRef,
                ObjectRef,
                str,
                dict[str, ObjectRef],
            ],
        ] = {}
        self._batch_x: ObjectRef | None = None
        self._batch_y: ObjectRef | None = None
        self._forward_loss: ObjectRef | None = None
        self._optimizer_configuration: ObjectRef | None = None
        self._train_inputs_source: ObjectRef | None = None
        self._train_targets_source: ObjectRef | None = None
        self._validation_inputs_source: ObjectRef | None = None
        self._validation_targets_source: ObjectRef | None = None
        self._clip_configuration: ObjectRef | None = None

    def _put_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value_json) VALUES (?,?)",
            (key, _json(value)),
        )

    def _start_group(self, stage: str, optimizer_step: int) -> None:
        if self._group is not None:
            raise RuntimeError("CAPTURE_GROUP_ALREADY_OPEN")
        self._group = {
            "edges": [],
            "fact_blocks": [],
            "objects": [],
            "occurrences": [],
            "optimizer_step": optimizer_step,
            "schema": "nanogpt-training-gfg-block-v2",
            "stage": stage,
        }

    def _flush_group(self) -> None:
        if self._group is None:
            raise RuntimeError("CAPTURE_GROUP_NOT_OPEN")
        payload = self._group
        self._group = None
        raw = canonical_bytes(payload)
        payload_sha = hashlib.sha256(raw).hexdigest()
        material = {
            "block_ordinal": self._block_ordinal,
            "optimizer_step": payload["optimizer_step"],
            "payload_sha256": payload_sha,
            "prior_block_sha256": self._last_block_sha,
            "run_id": self.run_id,
            "stage": payload["stage"],
        }
        block_sha = payload_sha256(material)
        block_id = "block_" + block_sha
        fact_count = sum(len(row["sources"]) for row in payload["fact_blocks"])
        self.connection.execute(
            """
            INSERT INTO graph_blocks(
              block_ordinal,block_id,optimizer_step,stage,
              prior_block_sha256,payload_sha256,block_sha256,
              object_count,occurrence_count,fact_count,
              explicit_edge_count,payload_zlib
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                self._block_ordinal,
                block_id,
                payload["optimizer_step"],
                payload["stage"],
                self._last_block_sha,
                payload_sha,
                block_sha,
                len(payload["objects"]),
                len(payload["occurrences"]),
                fact_count,
                len(payload["edges"]),
                sqlite3.Binary(zlib.compress(raw, level=6)),
            ),
        )
        self._last_block_sha = block_sha
        self._block_ordinal += 1

    def _append_object(self, descriptor: dict[str, Any]) -> ObjectRef:
        if self._group is None:
            raise RuntimeError("OBJECT_OUTSIDE_CAPTURE_GROUP")
        content_sha = descriptor["content_sha256"]
        object_id = "obj_" + payload_sha256(
            {
                "content_sha256": content_sha,
                "dtype": descriptor["dtype"],
                "run_id": self.run_id,
                "semantic_key": descriptor["semantic_key"],
                "shape": descriptor["shape"],
            }
        )
        descriptor = {"object_id": object_id, **descriptor}
        self._group["objects"].append(descriptor)
        return ObjectRef(
            object_id,
            content_sha,
            descriptor["semantic_key"],
            descriptor["role"],
        )

    def _literal_object(
        self,
        *,
        semantic_key: str,
        role: str,
        optimizer_step: int,
        name: str,
        payload: dict[str, Any],
    ) -> ObjectRef:
        content_sha = payload_sha256(payload)
        return self._append_object(
            {
                "content_sha256": content_sha,
                "dtype": "json",
                "literal_payload": payload,
                "locator": "inline://literal",
                "materialized": True,
                "name": name,
                "object_kind": "literal",
                "optimizer_step": optimizer_step,
                "payload": {},
                "role": role,
                "semantic_key": semantic_key,
                "shape": [],
            }
        )

    def _materialize_step(self, step: int) -> bool:
        return step == 0 or step % self.materialization_interval == 0

    def _tensor_object(
        self,
        tensor: torch.Tensor,
        *,
        semantic_key: str,
        role: str,
        optimizer_step: int,
        name: str,
        force_materialize: bool = False,
        payload: dict[str, Any] | None = None,
    ) -> ObjectRef:
        array, raw = _tensor_bytes(tensor)
        content_sha = hashlib.sha256(raw).hexdigest()
        materialized = force_materialize or self._materialize_step(optimizer_step)
        if materialized:
            path = self.object_root / f"{content_sha}.npy"
            if not path.exists():
                np.save(path, array, allow_pickle=False)
            locator = f"objects://{path.name}"
        else:
            locator = f"replay://{self.run_id}/step/{optimizer_step}/{semantic_key}"
        return self._append_object(
            {
                "content_sha256": content_sha,
                "dtype": str(tensor.dtype),
                "locator": locator,
                "materialized": materialized,
                "name": name,
                "object_kind": "tensor",
                "optimizer_step": optimizer_step,
                "payload": {
                    "device_at_capture": str(tensor.device),
                    "requires_grad": bool(tensor.requires_grad),
                    "tensor_version": int(tensor._version),
                    **(payload or {}),
                },
                "role": role,
                "semantic_key": semantic_key,
                "shape": list(tensor.shape),
            }
        )

    def _occurrence(
        self,
        *,
        optimizer_step: int,
        occurrence_type: str,
        occurrence_stage: str,
        transform: dict[str, Any],
        payload: dict[str, Any],
    ) -> str:
        if self._group is None:
            raise RuntimeError("OCCURRENCE_OUTSIDE_CAPTURE_GROUP")
        ordinal = self._occurrence_ordinal
        self._occurrence_ordinal += 1
        occurrence_id = "occ_" + payload_sha256(
            {
                "occurrence_type": occurrence_type,
                "optimizer_step": optimizer_step,
                "ordinal": ordinal,
                "run_id": self.run_id,
            }
        )
        self._group["occurrences"].append(
            {
                "occurrence_id": occurrence_id,
                "occurrence_stage": occurrence_stage,
                "occurrence_type": occurrence_type,
                "optimizer_step": optimizer_step,
                "ordinal": ordinal,
                "payload": payload,
                "transform_reference": transform,
            }
        )
        if self._last_occurrence is not None:
            self._edge(
                "program_order",
                self._last_occurrence,
                occurrence_id,
                {"basis": "synchronous_training_control_flow"},
            )
        self._last_occurrence = occurrence_id
        return occurrence_id

    def _edge(
        self,
        relation_type: str,
        source_id: str,
        target_id: str,
        payload: dict[str, Any],
        primitive_or_derived: str = "primitive",
    ) -> None:
        if self._group is None:
            raise RuntimeError("EDGE_OUTSIDE_CAPTURE_GROUP")
        material = {
            "payload": payload,
            "primitive_or_derived": primitive_or_derived,
            "relation_type": relation_type,
            "source_id": source_id,
            "target_id": target_id,
        }
        self._group["edges"].append(
            {"edge_id": "edge_" + payload_sha256(material), **material}
        )

    def _bind(
        self,
        occurrence_id: str,
        sources: Iterable[tuple[ObjectRef, str]],
        outcomes: Iterable[ObjectRef],
        *,
        binding_payload: dict[str, Any] | None = None,
    ) -> None:
        if self._group is None:
            raise RuntimeError("BINDING_OUTSIDE_CAPTURE_GROUP")
        source_rows = [
            {
                "content_sha256": source.content_sha256,
                "object_id": source.object_id,
                "relation_role": role,
            }
            for source, role in sources
        ]
        outcome_rows = [
            {
                "content_sha256": outcome.content_sha256,
                "object_id": outcome.object_id,
                "outcome_role": outcome.role,
            }
            for outcome in outcomes
        ]
        if not source_rows or len(outcome_rows) != 1:
            raise RuntimeError("EMPTY_GENERATION_BINDING_BLOCK")
        material = {
            "domain_scope_id": self.run_id,
            "occurrence_id": occurrence_id,
            "outcomes": outcome_rows,
            "payload": binding_payload or {},
            "sources": source_rows,
        }
        self._group["fact_blocks"].append(
            {
                "fact_block_id": "factblock_" + payload_sha256(material),
                **material,
            }
        )

    def _bind_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self._model_bound:
            return
        self._train_inputs_source = self._tensor_object(
            self.task.train_inputs,
            semantic_key=f"{self.run_id}:dataset:train:inputs",
            role="training_dataset_inputs",
            optimizer_step=0,
            name="training_dataset_inputs",
            force_materialize=True,
        )
        self._train_targets_source = self._tensor_object(
            self.task.train_targets,
            semantic_key=f"{self.run_id}:dataset:train:targets",
            role="training_dataset_targets",
            optimizer_step=0,
            name="training_dataset_targets",
            force_materialize=True,
        )
        self._validation_inputs_source = self._tensor_object(
            self.task.validation_inputs,
            semantic_key=f"{self.run_id}:dataset:validation:inputs",
            role="validation_dataset_inputs",
            optimizer_step=0,
            name="validation_dataset_inputs",
            force_materialize=True,
        )
        self._validation_targets_source = self._tensor_object(
            self.task.validation_targets,
            semantic_key=f"{self.run_id}:dataset:validation:targets",
            role="validation_dataset_targets",
            optimizer_step=0,
            name="validation_dataset_targets",
            force_materialize=True,
        )
        self._optimizer_configuration = self._literal_object(
            semantic_key=f"{self.run_id}:optimizer:configuration",
            role="optimizer_configuration",
            optimizer_step=0,
            name="AdamW",
            payload={
                "class": type(optimizer).__name__,
                "param_groups": [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in optimizer.param_groups
                ],
            },
        )
        self._clip_configuration = self._literal_object(
            semantic_key=f"{self.run_id}:gradient-clip:configuration",
            role="gradient_clip_configuration",
            optimizer_step=0,
            name="gradient_clip_configuration",
            payload={
                "error_if_nonfinite": False,
                "max_norm": 1.0,
                "norm_type": 2.0,
            },
        )
        initial_version = self._current_step
        refs_by_identity: dict[int, ObjectRef] = {}
        for name, parameter in model.named_parameters():
            ref = self._tensor_object(
                parameter,
                semantic_key=(
                    f"{self.run_id}:parameter:{name}:version:{initial_version}"
                ),
                role="parameter_version",
                optimizer_step=initial_version,
                name=name,
                force_materialize=True,
                payload={"parameter_version": initial_version},
            )
            self._parameter_refs[name] = ref
            refs_by_identity[id(parameter)] = ref
            state_refs: dict[str, ObjectRef] = {}
            for state_name, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    state_refs[state_name] = self._tensor_object(
                        value,
                        semantic_key=(
                            f"{self.run_id}:optimizer:{name}:"
                            f"{state_name}:version:{initial_version}"
                        ),
                        role="optimizer_state",
                        optimizer_step=initial_version,
                        name=f"{name}.{state_name}",
                        force_materialize=True,
                        payload={
                            "parameter_version": initial_version,
                            "resume_boundary": initial_version > 0,
                        },
                    )
            self._optimizer_state_refs[name] = state_refs
        named_modules = [
            ("token_embedding", model.transformer.wte),
            ("position_embedding", model.transformer.wpe),
            *[
                (f"transformer_block_{index}", block)
                for index, block in enumerate(model.transformer.h)
            ],
            ("final_layer_norm", model.transformer.ln_f),
            ("language_model_head", model.lm_head),
        ]
        for module_name, module in named_modules:
            self._module_parameter_refs[module_name] = [
                refs_by_identity[id(parameter)]
                for parameter in module.parameters(recurse=True)
                if id(parameter) in refs_by_identity
            ]
            self._module_handles.append(
                module.register_forward_hook(self._module_hook(module_name))
            )
        self._model_bound = True

    def _module_hook(self, module_name: str):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            tensor_inputs = [
                value for value in inputs if isinstance(value, torch.Tensor)
            ]
            if isinstance(output, torch.Tensor):
                tensor_outputs = [output]
            elif isinstance(output, (tuple, list)):
                tensor_outputs = [
                    value for value in output if isinstance(value, torch.Tensor)
                ]
            else:
                tensor_outputs = []
            if not tensor_outputs:
                return
            step = self._current_step
            stage = f"{self._stage}:{module_name}"
            self._start_group(stage, step)
            sources: list[tuple[ObjectRef, str]] = []
            for index, tensor in enumerate(tensor_inputs):
                sources.append(
                    (
                        self._tensor_object(
                            tensor,
                            semantic_key=(
                                f"{self.run_id}:step:{step}:stage:"
                                f"{self._stage}:module:{module_name}:"
                                f"input:{index}"
                            ),
                            role="layer_input",
                            optimizer_step=step,
                            name=f"{module_name}.input.{index}",
                        ),
                        f"layer_input_{index}",
                    )
                )
            sources.extend(
                (ref, "layer_parameter")
                for ref in self._module_parameter_refs[module_name]
            )
            outcomes = [
                self._tensor_object(
                    tensor,
                    semantic_key=(
                        f"{self.run_id}:step:{step}:stage:{self._stage}:"
                        f"module:{module_name}:output:{index}"
                    ),
                    role="layer_activation",
                    optimizer_step=step,
                    name=f"{module_name}.output.{index}",
                )
                for index, tensor in enumerate(tensor_outputs)
            ]
            occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="layer_forward",
                occurrence_stage=self._stage,
                transform={
                    "framework": "PyTorch",
                    "module": type(module).__name__,
                    "module_name": module_name,
                },
                payload={"training": bool(module.training)},
            )
            self._bind(occurrence, sources, outcomes)
            self._flush_group()

        return hook

    def __call__(self, stage: str, payload: dict[str, Any]) -> None:
        if stage == "before_batch":
            step = int(payload["step"])
            if step and step % self.chunk_steps == 0:
                self._close_chunk(end_optimizer_step=step)
            self._current_step = step
            self._stage = "training_forward"
            self._start_group("before_batch", step)
            self._bind_model(payload["model"], payload["optimizer"])
            order = payload["order"].detach().cpu()
            order_ref = self._tensor_object(
                order,
                semantic_key=f"{self.run_id}:step:{step}:batch:order",
                role="batch_selection_order",
                optimizer_step=step,
                name="batch_selection_order",
                force_materialize=True,
            )
            sample_ids = [
                self.task.train_sample_ids[int(index)] for index in order.tolist()
            ]
            self._batch_x = self._tensor_object(
                payload["x"],
                semantic_key=f"{self.run_id}:step:{step}:batch:inputs",
                role="training_batch_inputs",
                optimizer_step=step,
                name="batch_inputs",
                force_materialize=True,
                payload={"sample_ids": sample_ids},
            )
            self._batch_y = self._tensor_object(
                payload["y"],
                semantic_key=f"{self.run_id}:step:{step}:batch:targets",
                role="training_batch_targets",
                optimizer_step=step,
                name="batch_targets",
                force_materialize=True,
                payload={"sample_ids": sample_ids},
            )
            occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="batch_materialization",
                occurrence_stage="before_batch",
                transform={"operation": "frozen_full_batch_selection"},
                payload={"sample_ids": sample_ids},
            )
            self._bind(
                occurrence,
                [
                    (self._train_inputs_source, "selected_dataset_inputs"),
                    (order_ref, "selection_order"),
                ],
                [self._batch_x],
                binding_payload={"selected_component": "inputs"},
            )
            self._bind(
                occurrence,
                [
                    (self._train_targets_source, "selected_dataset_targets"),
                    (order_ref, "selection_order"),
                ],
                [self._batch_y],
                binding_payload={"selected_component": "targets"},
            )
            self._flush_group()
            return

        if stage == "after_forward":
            step = int(payload["step"])
            self._stage = "after_forward"
            self._start_group(stage, step)
            logits = self._tensor_object(
                payload["logits"],
                semantic_key=f"{self.run_id}:step:{step}:forward:logits",
                role="training_logits",
                optimizer_step=step,
                name="training_logits",
            )
            loss = self._tensor_object(
                payload["loss"],
                semantic_key=f"{self.run_id}:step:{step}:forward:loss",
                role="training_loss",
                optimizer_step=step,
                name="training_loss",
            )
            forward_occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="training_forward",
                occurrence_stage=stage,
                transform={
                    "implementation": "frozen nanoGPT forward",
                },
                payload={"parameter_version": step},
            )
            self._bind(
                forward_occurrence,
                [
                    (self._batch_x, "input_tokens"),
                    *[
                        (ref, "parameter_version")
                        for ref in self._parameter_refs.values()
                    ],
                ],
                [logits],
            )
            loss_occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="training_loss",
                occurrence_stage=stage,
                transform={
                    "implementation": "frozen nanoGPT cross entropy",
                    "loss": "cross_entropy",
                },
                payload={"parameter_version": step},
            )
            self._bind(
                loss_occurrence,
                [
                    (logits, "forward_logits"),
                    (self._batch_y, "target_tokens"),
                ],
                [loss],
            )
            self._forward_loss = loss
            self._flush_group()
            return

        if stage == "after_backward":
            step = int(payload["step"])
            self._stage = stage
            self._start_group(stage, step)
            self._gradient_refs = {}
            self._gradient_roles = {}
            occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="autograd_backward",
                occurrence_stage=stage,
                transform={"operation": "PyTorch Autograd backward"},
                payload={},
            )
            for name, parameter in payload["model"].named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"MISSING_GRADIENT:{name}")
                gradient = self._tensor_object(
                    parameter.grad,
                    semantic_key=(f"{self.run_id}:step:{step}:gradient:{name}:raw"),
                    role="parameter_gradient",
                    optimizer_step=step,
                    name=name,
                )
                self._bind(
                    occurrence,
                    [
                        (self._forward_loss, "loss_seed"),
                        (
                            self._parameter_refs[name],
                            "differentiated_parameter",
                        ),
                    ],
                    [gradient],
                    binding_payload={"parameter_name": name},
                )
                self._gradient_refs[name] = gradient
                self._gradient_roles[name] = "unclipped_gradient"
            self._flush_group()
            return

        if stage == "after_gradient_clip":
            step = int(payload["step"])
            self._stage = stage
            self._start_group(stage, step)
            active_gradient_names = [
                name
                for name, parameter in payload["model"].named_parameters()
                if parameter.grad is not None
            ]
            total_norm = self._tensor_object(
                payload["total_norm"],
                semantic_key=f"{self.run_id}:step:{step}:gradient:total-norm",
                role="gradient_total_norm",
                optimizer_step=step,
                name="gradient_total_norm",
                force_materialize=True,
            )
            norm_occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="gradient_global_norm",
                occurrence_stage=stage,
                transform={
                    "norm_type": 2.0,
                    "operation": "torch._foreach_norm/clip_grad_norm_ norm phase",
                },
                payload={"active_gradient_count": len(active_gradient_names)},
            )
            self._bind(
                norm_occurrence,
                [
                    (self._gradient_refs[name], "norm_input_gradient")
                    for name in active_gradient_names
                ],
                [total_norm],
            )
            occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="gradient_clip_application",
                occurrence_stage=stage,
                transform={
                    "max_norm": 1.0,
                    "operation": "torch.nn.utils.clip_grad_norm_ scale phase",
                },
                payload={"total_norm": float(payload["total_norm"].detach().cpu())},
            )
            clipped: dict[str, ObjectRef] = {}
            for name, parameter in payload["model"].named_parameters():
                if parameter.grad is None:
                    disposition = self._literal_object(
                        semantic_key=(
                            f"{self.run_id}:step:{step}:gradient:{name}:"
                            "clip-disposition"
                        ),
                        role="explicit_disposition",
                        optimizer_step=step,
                        name=f"{name}.gradient_clip_disposition",
                        payload={
                            "classification": "NOT_USED",
                            "parameter_name": name,
                            "reason": "gradient_absent_at_native_clip",
                            "stage": stage,
                        },
                    )
                    self._bind(
                        occurrence,
                        [
                            (
                                self._gradient_refs[name],
                                "unclipped_gradient",
                            ),
                            (
                                self._clip_configuration,
                                "clip_configuration",
                            ),
                        ],
                        [disposition],
                        binding_payload={
                            "explicit_disposition": True,
                            "parameter_name": name,
                        },
                    )
                    clipped[name] = disposition
                    self._gradient_roles[name] = "gradient_disposition"
                    continue
                output = self._tensor_object(
                    parameter.grad,
                    semantic_key=(f"{self.run_id}:step:{step}:gradient:{name}:clipped"),
                    role="clipped_parameter_gradient",
                    optimizer_step=step,
                    name=name,
                )
                self._bind(
                    occurrence,
                    [
                        (self._gradient_refs[name], "unclipped_gradient"),
                        (total_norm, "global_gradient_norm"),
                        (
                            self._clip_configuration,
                            "clip_configuration",
                        ),
                    ],
                    [output],
                    binding_payload={"parameter_name": name},
                )
                clipped[name] = output
                self._gradient_roles[name] = "clipped_gradient"
            self._gradient_refs = clipped
            self._flush_group()
            return

        if stage == "before_optimizer_step":
            self._pending_optimizer_sources = {
                name: (
                    self._parameter_refs[name],
                    self._gradient_refs[name],
                    self._gradient_roles[name],
                    self._optimizer_state_refs.get(name, {}),
                )
                for name, _parameter in payload["model"].named_parameters()
            }
            return

        if stage == "after_optimizer_step":
            step = int(payload["step"])
            parameter_version = int(payload["parameter_version"])
            self._current_step = parameter_version
            self._stage = stage
            self._start_group(stage, parameter_version)
            occurrence = self._occurrence(
                optimizer_step=parameter_version,
                occurrence_type="optimizer_parameter_update",
                occurrence_stage=stage,
                transform={
                    "algorithm": type(payload["optimizer"]).__name__,
                    "implementation": "actual optimizer.step",
                },
                payload={"parameter_version": parameter_version},
            )
            next_parameters: dict[str, ObjectRef] = {}
            next_states: dict[str, dict[str, ObjectRef]] = {}
            for name, parameter in payload["model"].named_parameters():
                before_parameter, gradient, gradient_role, before_states = (
                    self._pending_optimizer_sources[name]
                )
                after_parameter = self._tensor_object(
                    parameter,
                    semantic_key=(
                        f"{self.run_id}:parameter:{name}:version:{parameter_version}"
                    ),
                    role="parameter_version",
                    optimizer_step=parameter_version,
                    name=name,
                    payload={"parameter_version": parameter_version},
                )
                after_states: dict[str, ObjectRef] = {}
                for state_name, value in payload["optimizer"].state[parameter].items():
                    if isinstance(value, torch.Tensor):
                        after_states[state_name] = self._tensor_object(
                            value,
                            semantic_key=(
                                f"{self.run_id}:optimizer:{name}:"
                                f"{state_name}:version:{parameter_version}"
                            ),
                            role="optimizer_state",
                            optimizer_step=parameter_version,
                            name=f"{name}.{state_name}",
                        )
                self._bind(
                    occurrence,
                    [
                        (before_parameter, "parameter_before_update"),
                        (gradient, gradient_role),
                        (
                            self._optimizer_configuration,
                            "optimizer_configuration_for_parameter_update",
                        ),
                        *[
                            (value, f"{state_name}_before_parameter_update")
                            for state_name, value in sorted(before_states.items())
                        ],
                    ],
                    [after_parameter],
                    binding_payload={
                        "outcome_component": "parameter",
                        "parameter_name": name,
                    },
                )
                for state_name, after_state in sorted(after_states.items()):
                    state_sources: list[tuple[ObjectRef, str]] = [
                        (
                            self._optimizer_configuration,
                            f"optimizer_configuration_for_{state_name}",
                        )
                    ]
                    before_state = before_states.get(state_name)
                    if state_name == "step":
                        state_sources.append(
                            (before_parameter, "optimizer_state_owner")
                        )
                        if before_state is not None:
                            state_sources.append((before_state, "step_before_update"))
                    elif state_name == "exp_avg":
                        state_sources.append((gradient, gradient_role))
                        if before_state is not None:
                            state_sources.append(
                                (before_state, "exp_avg_before_update")
                            )
                    elif state_name == "exp_avg_sq":
                        state_sources.append((gradient, gradient_role))
                        if before_state is not None:
                            state_sources.append(
                                (before_state, "exp_avg_sq_before_update")
                            )
                    elif state_name == "max_exp_avg_sq":
                        state_sources.append((gradient, gradient_role))
                        for dependency_name in (
                            "exp_avg_sq",
                            "max_exp_avg_sq",
                        ):
                            dependency = before_states.get(dependency_name)
                            if dependency is not None:
                                state_sources.append(
                                    (
                                        dependency,
                                        f"{dependency_name}_before_update",
                                    )
                                )
                    else:
                        state_sources.append((gradient, gradient_role))
                        if before_state is not None:
                            state_sources.append(
                                (
                                    before_state,
                                    f"{state_name}_before_update",
                                )
                            )
                    self._bind(
                        occurrence,
                        state_sources,
                        [after_state],
                        binding_payload={
                            "optimizer_state": state_name,
                            "outcome_component": "optimizer_state",
                            "parameter_name": name,
                        },
                    )
                self._edge(
                    "GeneratedOrigin",
                    before_parameter.object_id,
                    after_parameter.object_id,
                    {
                        "parameter_name": name,
                        "prior_version": parameter_version - 1,
                        "result_version": parameter_version,
                    },
                )
                for state_name, after_state in after_states.items():
                    before_state = before_states.get(state_name)
                    if before_state is not None:
                        self._edge(
                            "GeneratedOrigin",
                            before_state.object_id,
                            after_state.object_id,
                            {
                                "optimizer_state": state_name,
                                "parameter_name": name,
                                "prior_version": parameter_version - 1,
                                "result_version": parameter_version,
                            },
                        )
                next_parameters[name] = after_parameter
                next_states[name] = after_states
            self._parameter_refs = next_parameters
            self._optimizer_state_refs = next_states
            self._flush_group()
            self.connection.commit()
            return

        if stage == "before_evaluation":
            self._current_step = int(payload["step"])
            self._stage = "evaluation"
            return

        if stage == "before_evaluation_split":
            self._current_step = int(payload["step"])
            self._stage = f"evaluation_{payload['split']}"
            return

        if stage == "evaluation":
            step = int(payload["step"])
            self._current_step = step
            self._stage = stage
            self._start_group(stage, step)
            evaluation_sources = {
                "train": (
                    self._tensor_object(
                        payload["train_inputs"],
                        semantic_key=f"{self.run_id}:evaluation:{step}:train:inputs",
                        role="evaluation_train_inputs",
                        optimizer_step=step,
                        name="evaluation_train_inputs",
                        force_materialize=True,
                    ),
                    self._tensor_object(
                        payload["train_targets"],
                        semantic_key=f"{self.run_id}:evaluation:{step}:train:targets",
                        role="evaluation_train_targets",
                        optimizer_step=step,
                        name="evaluation_train_targets",
                        force_materialize=True,
                    ),
                ),
                "validation": (
                    self._tensor_object(
                        payload["validation_inputs"],
                        semantic_key=(
                            f"{self.run_id}:evaluation:{step}:validation:inputs"
                        ),
                        role="evaluation_validation_inputs",
                        optimizer_step=step,
                        name="evaluation_validation_inputs",
                        force_materialize=True,
                    ),
                    self._tensor_object(
                        payload["validation_targets"],
                        semantic_key=(
                            f"{self.run_id}:evaluation:{step}:validation:targets"
                        ),
                        role="evaluation_validation_targets",
                        optimizer_step=step,
                        name="evaluation_validation_targets",
                        force_materialize=True,
                    ),
                ),
            }
            outputs = {
                (split, kind): self._tensor_object(
                    payload[f"{split}_{kind}"],
                    semantic_key=(f"{self.run_id}:evaluation:{step}:{split}:{kind}"),
                    role=f"{split}_{kind}",
                    optimizer_step=step,
                    name=f"{split}_{kind}",
                    force_materialize=True,
                )
                for split in ("train", "validation")
                for kind in ("predictions", "logits")
            }
            metric = payload["metrics"]
            metric_object = self._literal_object(
                semantic_key=f"{self.run_id}:evaluation:{step}:metrics",
                role="capability_evaluation",
                optimizer_step=step,
                name="evaluation_metrics",
                payload=metric,
            )
            for split in ("train", "validation"):
                inputs, _targets = evaluation_sources[split]
                forward_occurrence = self._occurrence(
                    optimizer_step=step,
                    occurrence_type="evaluation_forward",
                    occurrence_stage=f"evaluation_{split}",
                    transform={
                        "operation": "frozen_nanoGPT_evaluation_forward",
                        "split": split,
                    },
                    payload={"parameter_version": step, "split": split},
                )
                self._bind(
                    forward_occurrence,
                    [
                        (inputs, "evaluation_inputs"),
                        *[
                            (ref, "evaluated_parameter_version")
                            for ref in self._parameter_refs.values()
                        ],
                    ],
                    [outputs[(split, "logits")]],
                    binding_payload={"split": split},
                )
                prediction_occurrence = self._occurrence(
                    optimizer_step=step,
                    occurrence_type="evaluation_prediction",
                    occurrence_stage=f"evaluation_{split}",
                    transform={"operation": "last_token_argmax", "split": split},
                    payload={"parameter_version": step, "split": split},
                )
                self._bind(
                    prediction_occurrence,
                    [(outputs[(split, "logits")], "evaluation_logits")],
                    [outputs[(split, "predictions")]],
                    binding_payload={"split": split},
                )
            occurrence = self._occurrence(
                optimizer_step=step,
                occurrence_type="capability_evaluation",
                occurrence_stage=stage,
                transform={"operation": "frozen_train_validation_evaluation"},
                payload={"parameter_version": step},
            )
            self._bind(
                occurrence,
                [
                    (
                        outputs[("train", "predictions")],
                        "train_predictions",
                    ),
                    (evaluation_sources["train"][1], "train_targets"),
                    (
                        outputs[("validation", "predictions")],
                        "validation_predictions",
                    ),
                    (
                        evaluation_sources["validation"][1],
                        "validation_targets",
                    ),
                    (self._forward_loss, "latest_training_loss"),
                ],
                [metric_object],
            )
            self._flush_group()
            self.connection.execute(
                """
                INSERT INTO evaluations(
                  optimizer_step,parameter_version,train_accuracy,
                  validation_accuracy,loss,occurrence_id,metric_object_id
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    step,
                    int(metric["parameter_version"]),
                    float(metric["train_accuracy"]),
                    float(metric["validation_accuracy"]),
                    float(metric["loss"]),
                    occurrence,
                    metric_object.object_id,
                ),
            )
            self.connection.commit()
            return

    def _close_chunk(self, *, end_optimizer_step: int) -> None:
        end_block = self._block_ordinal - 1
        if end_block < self._chunk_start_block:
            return
        rows = list(
            self.connection.execute(
                """
                SELECT block_sha256,object_count,occurrence_count,
                       fact_count,explicit_edge_count
                FROM graph_blocks
                WHERE block_ordinal BETWEEN ? AND ?
                ORDER BY block_ordinal
                """,
                (self._chunk_start_block, end_block),
            )
        )
        counts = {
            "blocks": len(rows),
            "explicit_edges": sum(row[4] for row in rows),
            "facts": sum(row[3] for row in rows),
            "objects": sum(row[1] for row in rows),
            "occurrences": sum(row[2] for row in rows),
        }
        prior = self.connection.execute(
            "SELECT chunk_sha256 FROM chunks WHERE chunk_index=?",
            (self._chunk_index - 1,),
        ).fetchone()
        chunk_sha = payload_sha256(
            {
                "block_hashes": [row[0] for row in rows],
                "chunk_index": self._chunk_index,
                "end_optimizer_step": end_optimizer_step,
                "prior_chunk_sha256": prior[0] if prior else None,
                "run_id": self.run_id,
            }
        )
        self.connection.execute(
            """
            INSERT INTO chunks(
              chunk_index,start_block_ordinal,end_block_ordinal,
              end_optimizer_step,prior_chunk_sha256,chunk_sha256,
              row_counts_json
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                self._chunk_index,
                self._chunk_start_block,
                end_block,
                end_optimizer_step,
                prior[0] if prior else None,
                chunk_sha,
                _json(counts),
            ),
        )
        self._chunk_index += 1
        self._chunk_start_block = end_block + 1
        self.connection.commit()

    def finalize(self, final_step: int) -> dict[str, Any]:
        if self._group is not None:
            raise RuntimeError("CAPTURE_GROUP_LEFT_OPEN")
        self._close_chunk(end_optimizer_step=final_step)
        for handle in self._module_handles:
            handle.remove()
        aggregate = self.connection.execute(
            """
            SELECT COUNT(*),COALESCE(SUM(object_count),0),
                   COALESCE(SUM(occurrence_count),0),
                   COALESCE(SUM(fact_count),0),
                   COALESCE(SUM(explicit_edge_count),0)
            FROM graph_blocks
            """
        ).fetchone()
        counts = {
            "blocks": aggregate[0],
            "explicit_edges": aggregate[4],
            "facts": aggregate[3],
            "objects": aggregate[1],
            "occurrences": aggregate[2],
            "evaluations": self.connection.execute(
                "SELECT COUNT(*) FROM evaluations"
            ).fetchone()[0],
            "chunks": self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[
                0
            ],
        }
        self._put_metadata("final_counts", counts)
        self._put_metadata("final_step", final_step)
        self._put_metadata("final_block_sha256", self._last_block_sha)
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()
        manifest = {
            "chunks": [
                {
                    "chunk_index": row[0],
                    "chunk_sha256": row[1],
                    "end_optimizer_step": row[2],
                }
                for row in sqlite3.connect(self.database_path).execute(
                    """
                    SELECT chunk_index,chunk_sha256,end_optimizer_step
                    FROM chunks ORDER BY chunk_index
                    """
                )
            ],
            "counts": counts,
            "database": self.database_path.name,
            "database_sha256": _file_hash(self.database_path),
            "final_block_sha256": self._last_block_sha,
            "final_step": final_step,
            "object_file_count": len(list(self.object_root.glob("*.npy"))),
            "run_id": self.run_id,
            "schema": "nanogpt-training-gfg-manifest-v1",
            "status": "CAPTURE_CLOSED",
            "task_commitment": self.task.participant_task_commitment,
        }
        manifest["manifest_sha256"] = payload_sha256(manifest)
        write_json(self.run_directory / "capture_manifest.json", manifest)
        return manifest


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
