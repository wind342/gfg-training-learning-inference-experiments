from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Protocol

import torch
from torch.utils.checkpoint import checkpoint


SEED = 424242


class WriteOnlyCollector(Protocol):
    def write(self, event: "RuntimeEvent") -> None: ...


class NativeObserver(Protocol):
    def __call__(self, loss: torch.Tensor) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrainingSpec:
    workload: str
    sample_identity: str = "sample_a"
    evidence_context: str = "environment_a"
    checkpoint_mode: str = "none"
    step_key: str = "step_0"


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    stage: str
    ordinal: int
    step_key: str
    payload: dict[str, Any]
    tensors: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class TrainingRun:
    spec: TrainingSpec
    ordinary_result: dict[str, Any]
    ordinary_bytes: bytes
    native_observation: dict[str, Any] | None


class _Runtime:
    def __init__(self, spec: TrainingSpec, collector: WriteOnlyCollector | None) -> None:
        self.spec = spec
        self.collector = collector
        self.stage = "forward"
        self._next_ordinal = 0
        self._tensors: list[tuple[torch.Tensor, str]] = []

    def reserve_ordinal(self) -> int:
        result = self._next_ordinal
        self._next_ordinal += 1
        return result

    def register(self, tensor: torch.Tensor, stable_ref: str) -> torch.Tensor:
        for existing, existing_ref in self._tensors:
            if existing is tensor:
                if existing_ref != stable_ref:
                    raise RuntimeError("TENSOR_RUNTIME_IDENTITY_CONFLICT")
                return tensor
        self._tensors.append((tensor, stable_ref))
        return tensor

    def reference(self, tensor: torch.Tensor) -> str:
        for existing, stable_ref in self._tensors:
            if existing is tensor:
                return stable_ref
        raise RuntimeError("UNREGISTERED_RUNTIME_TENSOR")

    def emit(
        self,
        kind: str,
        payload: dict[str, Any],
        tensors: tuple[torch.Tensor, ...] = (),
        *,
        ordinal: int | None = None,
        stage: str | None = None,
    ) -> None:
        if self.collector is None:
            return
        before = [_tensor_guard(tensor) for tensor in tensors]
        rng_before = torch.random.get_rng_state().tolist()
        event = RuntimeEvent(
            kind=kind,
            stage=stage or self.stage,
            ordinal=self.reserve_ordinal() if ordinal is None else ordinal,
            step_key=self.spec.step_key,
            payload=payload,
            tensors=tensors,
        )
        callback_result = self.collector.write(event)
        if callback_result is not None:
            raise RuntimeError("COLLECTOR_CALLBACK_MUST_BE_WRITE_ONLY")
        after = [_tensor_guard(tensor) for tensor in tensors]
        if before != after or rng_before != torch.random.get_rng_state().tolist():
            raise RuntimeError("COLLECTOR_MUTATED_TRAINING_STATE")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "requires_grad": bool(tensor.requires_grad),
        "shape": list(tensor.shape),
        "value": detached.tolist(),
    }


def _tensor_guard(tensor: torch.Tensor) -> dict[str, Any]:
    payload = _tensor_payload(tensor)
    payload["value_sha256"] = hashlib.sha256(_canonical_bytes(payload["value"])).hexdigest()
    return payload


def _configure_runtime() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("PYTORCH_INTEROP_THREAD_PROFILE_DRIFT") from exc
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("CUDA_MUST_BE_DISABLED")


def _source(
    runtime: _Runtime,
    stable_ref: str,
    tensor: torch.Tensor,
    *,
    source_identity: str,
    source_role: str,
    version: str,
    selected: bool = True,
) -> torch.Tensor:
    runtime.register(tensor, stable_ref)
    runtime.emit(
        "source",
        {
            "selected": selected,
            "source_identity": source_identity,
            "source_ref": stable_ref,
            "source_role": source_role,
            "tensor": _tensor_payload(tensor),
            "version": version,
        },
        (tensor,),
    )
    return tensor


def _tracked(
    runtime: _Runtime,
    operation_type: str,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    attributes: dict[str, Any] | None = None,
) -> torch.Tensor:
    ordinal = runtime.reserve_ordinal()
    output_ref = f"{runtime.spec.step_key}:{runtime.stage}:tensor:{ordinal}"
    runtime.register(output, output_ref)
    runtime.emit(
        "operation",
        {
            "attributes": attributes or {},
            "input_refs": [runtime.reference(tensor) for tensor in inputs],
            "input_tensors": [_tensor_payload(tensor) for tensor in inputs],
            "operation_type": operation_type,
            "output_ref": output_ref,
            "output_tensor": _tensor_payload(output),
        },
        (*inputs, output),
        ordinal=ordinal,
    )
    return output


def tracked_matmul(runtime: _Runtime, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_matmul", (left, right), torch.matmul(left, right))


def tracked_add(runtime: _Runtime, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_add", (left, right), torch.add(left, right))


def tracked_mul(runtime: _Runtime, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_mul", (left, right), torch.mul(left, right))


def tracked_relu(runtime: _Runtime, value: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_relu", (value,), torch.relu(value))


def tracked_pow(runtime: _Runtime, value: torch.Tensor, exponent: float) -> torch.Tensor:
    return _tracked(
        runtime,
        "tracked_pow",
        (value,),
        torch.pow(value, exponent),
        {"exponent": exponent},
    )


def tracked_sum(runtime: _Runtime, value: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_sum", (value,), torch.sum(value))


def tracked_mean(runtime: _Runtime, value: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_mean", (value,), torch.mean(value))


def tracked_sin(runtime: _Runtime, value: torch.Tensor) -> torch.Tensor:
    return _tracked(runtime, "tracked_sin", (value,), torch.sin(value))


def _standard_workload(
    runtime: _Runtime,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    name = runtime.spec.workload
    inputs: dict[str, torch.Tensor] = {}
    parameters: dict[str, torch.Tensor] = {}
    externals: dict[str, torch.Tensor] = {}

    if name in {"linear_chain", "branch_and_merge"}:
        inputs["x"] = _source(
            runtime,
            "source:sample:x",
            torch.tensor([[0.2, -0.1], [0.5, 0.3]], requires_grad=True),
            source_identity=runtime.spec.sample_identity,
            source_role="training_sample",
            version="input",
        )
        parameters["w"] = _source(
            runtime,
            "source:parameter:w:before",
            torch.tensor([[0.4, -0.2], [0.1, 0.6]], requires_grad=True),
            source_identity="parameter_w",
            source_role="parameter_before_step",
            version="before_step",
        )
        parameters["b"] = _source(
            runtime,
            "source:parameter:b:before",
            torch.tensor([0.05, -0.03], requires_grad=True),
            source_identity="parameter_b",
            source_role="parameter_before_step",
            version="before_step",
        )
        hidden = tracked_relu(
            runtime,
            tracked_add(runtime, tracked_matmul(runtime, inputs["x"], parameters["w"]), parameters["b"]),
        )
        if name == "linear_chain":
            forward_output = tracked_pow(runtime, hidden, 2.0)
            loss = tracked_mean(runtime, forward_output)
        else:
            externals["scale"] = _source(
                runtime,
                "source:external:scale",
                torch.tensor(1.25),
                source_identity="external_scale",
                source_role="external_state",
                version="forward",
            )
            branch_a = tracked_mul(runtime, hidden, externals["scale"])
            branch_b = tracked_pow(runtime, hidden, 2.0)
            forward_output = tracked_add(runtime, branch_a, branch_b)
            loss = tracked_sum(runtime, forward_output)
    elif name == "shared_tensor_reuse":
        inputs["x"] = _source(
            runtime,
            "source:sample:x",
            torch.tensor([0.2, 0.5], requires_grad=True),
            source_identity=runtime.spec.sample_identity,
            source_role="training_sample",
            version="input",
        )
        parameters["w"] = _source(
            runtime,
            "source:parameter:w:before",
            torch.tensor([0.4, 0.1], requires_grad=True),
            source_identity="parameter_w",
            source_role="parameter_before_step",
            version="before_step",
        )
        shared = tracked_mul(runtime, inputs["x"], parameters["w"])
        forward_output = tracked_add(runtime, shared, shared)
        loss = tracked_sum(runtime, forward_output)
    elif name == "duplicate_valued_distinct_sources":
        for key, identity in (("x1", runtime.spec.sample_identity), ("x2", runtime.spec.sample_identity + "_peer")):
            inputs[key] = _source(
                runtime,
                f"source:sample:{key}",
                torch.tensor([0.2, 0.5], requires_grad=True),
                source_identity=identity,
                source_role="training_sample",
                version="input",
            )
        parameters["w"] = _source(
            runtime,
            "source:parameter:w:before",
            torch.tensor([0.4, 0.1], requires_grad=True),
            source_identity="parameter_w",
            source_role="parameter_before_step",
            version="before_step",
        )
        left = tracked_mul(runtime, inputs["x1"], parameters["w"])
        right = tracked_mul(runtime, inputs["x2"], parameters["w"])
        forward_output = tracked_add(runtime, left, right)
        loss = tracked_sum(runtime, forward_output)
    elif name == "zero_gradient_and_unused_sources":
        parameters["p_zero"] = _source(
            runtime,
            "source:parameter:p_zero:before",
            torch.tensor([0.2, 0.5], requires_grad=True),
            source_identity="parameter_p_zero",
            source_role="parameter_before_step",
            version="before_step",
        )
        parameters["p_unused"] = _source(
            runtime,
            "source:parameter:p_unused:before",
            torch.tensor([0.2, 0.5], requires_grad=True),
            source_identity="parameter_p_unused",
            source_role="parameter_before_step",
            version="before_step",
            selected=False,
        )
        externals["zero"] = _source(
            runtime,
            "source:external:zero",
            torch.tensor(0.0),
            source_identity="zero_multiplier",
            source_role="external_state",
            version="forward",
        )
        forward_output = tracked_mul(runtime, parameters["p_zero"], externals["zero"])
        loss = tracked_sum(runtime, forward_output)
    else:
        raise ValueError(f"UNKNOWN_WORKLOAD:{name}")
    return inputs, parameters, externals, forward_output, loss


def _checkpoint_workload(
    runtime: _Runtime,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if runtime.spec.checkpoint_mode not in {"no_checkpoint", "stable", "divergent"}:
        raise ValueError(f"UNKNOWN_CHECKPOINT_MODE:{runtime.spec.checkpoint_mode}")
    sample = _source(
        runtime,
        "source:sample:x",
        torch.tensor([0.25, 0.5]),
        source_identity=runtime.spec.sample_identity,
        source_role="training_sample",
        version="input",
    )
    parameter = _source(
        runtime,
        "source:parameter:p:before",
        torch.tensor([0.7, -0.4], requires_grad=True),
        source_identity="parameter_p",
        source_role="parameter_before_step",
        version="before_step",
    )
    scale_state = {
        "value": _source(
            runtime,
            "source:external:scale:forward",
            torch.tensor(1.0),
            source_identity="external_scale_forward",
            source_role="external_state",
            version="original_forward",
        )
    }

    @contextlib.contextmanager
    def phase_context(stage: str):
        previous = runtime.stage
        runtime.stage = stage
        runtime.emit("phase_enter", {"phase": stage}, stage=stage)
        try:
            yield
        finally:
            runtime.emit("phase_exit", {"phase": stage}, stage=stage)
            runtime.stage = previous

    def context_fn():
        return phase_context("forward_original"), phase_context("backward_recomputation")

    def block(current_parameter: torch.Tensor, current_sample: torch.Tensor) -> torch.Tensor:
        product = tracked_mul(runtime, current_parameter, current_sample)
        scaled = tracked_mul(runtime, product, scale_state["value"])
        return tracked_sin(runtime, scaled)

    if runtime.spec.checkpoint_mode == "no_checkpoint":
        runtime.stage = "forward_original"
        forward_output = block(parameter, sample)
        runtime.stage = "forward"
    else:
        forward_output = checkpoint(
            block,
            parameter,
            sample,
            use_reentrant=False,
            context_fn=context_fn,
            determinism_check="default",
            early_stop=False,
        )
        runtime.register(forward_output, runtime.reference(forward_output))
    loss = tracked_sum(runtime, forward_output)
    externals = {"scale_forward": scale_state["value"]}
    if runtime.spec.checkpoint_mode in {"stable", "divergent"}:
        next_scale = 1.0 if runtime.spec.checkpoint_mode == "stable" else 2.0
        next_identity = "external_scale_recomputation_stable" if next_scale == 1.0 else "external_scale_recomputation_divergent"
        replacement = _source(
            runtime,
            "source:external:scale:recomputation",
            torch.tensor(next_scale),
            source_identity=next_identity,
            source_role="external_state",
            version="backward_recomputation",
        )
        scale_state["value"] = replacement
        externals["scale_recomputation"] = replacement
    return {"x": sample}, {"p": parameter}, externals, forward_output, loss


def _register_gradient_hooks(
    runtime: _Runtime,
    leaves: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    if runtime.collector is None:
        return handles
    for leaf_ref in sorted(leaves):
        leaf = leaves[leaf_ref]
        if not leaf.requires_grad:
            continue
        ordinal = runtime.reserve_ordinal()

        def hook(gradient: torch.Tensor, *, ref: str = leaf_ref, fixed_ordinal: int = ordinal) -> None:
            runtime.emit(
                "gradient",
                {
                    "gradient": _tensor_payload(gradient),
                    "leaf_ref": ref,
                },
                (gradient,),
                ordinal=fixed_ordinal,
                stage="gradient_production",
            )
            return None

        handles.append(leaf.register_hook(hook))
    return handles


def _optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(parameters):
        state = optimizer.state.get(parameters[name], {})
        result[name] = {
            key: _tensor_payload(value) if isinstance(value, torch.Tensor) else value
            for key, value in sorted(state.items())
        }
    return result


def _parameter_values(parameters: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _tensor_payload(parameters[name]) for name in sorted(parameters)}


def _gradient_values(leaves: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: None if leaves[name].grad is None else _tensor_payload(leaves[name].grad)
        for name in sorted(leaves)
    }


def run_training_step(
    spec: TrainingSpec,
    collector: WriteOnlyCollector | None = None,
    native_observer: NativeObserver | None = None,
) -> TrainingRun:
    """Execute the one shared training path for every capture mode."""
    _configure_runtime()
    runtime = _Runtime(spec, collector)
    if spec.workload == "checkpoint_external_state":
        inputs, parameters, externals, forward_output, loss = _checkpoint_workload(runtime)
    else:
        inputs, parameters, externals, forward_output, loss = _standard_workload(runtime)

    native_observation = None
    if native_observer is not None:
        before = _tensor_guard(loss)
        native_observation = native_observer(loss)
        if before != _tensor_guard(loss):
            raise RuntimeError("NATIVE_OBSERVER_MUTATED_TRAINING_STATE")

    parameter_before = _parameter_values(parameters)
    optimizer = torch.optim.SGD(
        list(parameters.values()),
        lr=0.05,
        momentum=0.25,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        foreach=False,
        fused=False,
        maximize=False,
    )
    optimizer_state_before = _optimizer_state(optimizer, parameters)
    runtime.emit(
        "source_metadata",
        {
            "selected": True,
            "source_identity": "optimizer_state",
            "source_ref": "source:optimizer:state:before",
            "source_role": "optimizer_state_before_step",
            "value": optimizer_state_before,
            "version": "before_step",
        },
        stage="optimizer_update",
    )
    all_leaves = {
        **{f"input:{name}": value for name, value in inputs.items()},
        **{f"parameter:{name}": value for name, value in parameters.items()},
    }
    handles = _register_gradient_hooks(runtime, all_leaves)
    runtime.emit(
        "backward_start",
        {"loss_ref": runtime.reference(loss)},
        (loss,),
        stage="backward",
    )
    loss.backward()
    for handle in handles:
        handle.remove()

    for leaf_ref in sorted(all_leaves):
        leaf = all_leaves[leaf_ref]
        if leaf.requires_grad and leaf.grad is None:
            runtime.emit(
                "gradient_absent",
                {"leaf_ref": leaf_ref, "reason_code": "UNUSED_IN_THIS_LOSS_OCCURRENCE"},
                (leaf,),
                stage="gradient_production",
            )

    runtime.emit(
        "optimizer_before",
        {
            "optimizer": "torch.optim.SGD",
            "parameter_refs": [runtime.reference(parameters[name]) for name in sorted(parameters)],
            "gradients": _gradient_values({f"parameter:{name}": value for name, value in parameters.items()}),
            "parameter_values": parameter_before,
            "optimizer_state": optimizer_state_before,
        },
        tuple(parameters[name] for name in sorted(parameters)),
        stage="optimizer_update",
    )
    optimizer.step()
    parameter_after = _parameter_values(parameters)
    optimizer_state_after = _optimizer_state(optimizer, parameters)
    runtime.emit(
        "optimizer_after",
        {
            "optimizer": "torch.optim.SGD",
            "parameter_refs": [runtime.reference(parameters[name]) for name in sorted(parameters)],
            "gradients": _gradient_values({f"parameter:{name}": value for name, value in parameters.items()}),
            "parameter_values": parameter_after,
            "optimizer_state": optimizer_state_after,
        },
        tuple(parameters[name] for name in sorted(parameters)),
        stage="optimizer_update",
    )

    ordinary_result = {
        "device": "cpu",
        "dtype": "torch.float64",
        "exception": None,
        "external_values": {name: _tensor_payload(value) for name, value in sorted(externals.items())},
        "forward_output": _tensor_payload(forward_output),
        "gradients": _gradient_values(all_leaves),
        "input_values": {name: _tensor_payload(value) for name, value in sorted(inputs.items())},
        "loss": _tensor_payload(loss),
        "optimizer_state_after": optimizer_state_after,
        "optimizer_state_before": optimizer_state_before,
        "parameter_after": parameter_after,
        "parameter_before": parameter_before,
        "workload": spec.workload,
    }
    ordinary_bytes = _canonical_bytes(ordinary_result) + b"\n"
    return TrainingRun(spec, ordinary_result, ordinary_bytes, native_observation)
