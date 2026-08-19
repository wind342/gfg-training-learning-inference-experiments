from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, ContextManager, Protocol

import torch
from torch.utils.checkpoint import checkpoint


SEED = 424242


class ExecutionObserver(Protocol):
    def saved_tensor_context(
        self,
        stage_getter: Callable[[], str],
    ) -> ContextManager[None]: ...

    def tensor_registered(
        self,
        tensor: torch.Tensor,
        stable_ref: str,
        stage: str,
    ) -> None: ...

    def source_registered(
        self,
        tensor: torch.Tensor,
        stable_ref: str,
        *,
        source_identity: str,
        source_role: str,
        version: str,
        selected: bool,
    ) -> None: ...

    def stable_ref_for_tensor(self, tensor: torch.Tensor) -> str | None: ...

    def before_backward(
        self,
        loss: torch.Tensor,
        leaves: dict[str, torch.Tensor],
    ) -> None: ...

    def after_backward(self, leaves: dict[str, torch.Tensor]) -> None: ...


@dataclass(frozen=True)
class NativeTrainingSpec:
    workload: str
    sample_identity: str = "sample_a"
    checkpoint_mode: str = "none"
    step_key: str = "step_0"


@dataclass(frozen=True)
class NativeTrainingRun:
    spec: NativeTrainingSpec
    ordinary_result: dict[str, Any]
    ordinary_bytes: bytes


class _Runtime:
    def __init__(
        self,
        spec: NativeTrainingSpec,
        observer: ExecutionObserver | None,
        source_overrides: dict[str, Any] | None,
    ) -> None:
        self.spec = spec
        self.observer = observer
        self.source_overrides = source_overrides or {}
        self.stage = "forward"
        self.next_ordinal = 0
        self.tensors: list[tuple[torch.Tensor, str]] = []

    def reserve(self) -> int:
        result = self.next_ordinal
        self.next_ordinal += 1
        return result

    def register(self, tensor: torch.Tensor, stable_ref: str) -> torch.Tensor:
        for existing, existing_ref in self.tensors:
            if existing is tensor:
                if existing_ref != stable_ref:
                    raise RuntimeError("NATIVE_WORKLOAD_TENSOR_REF_CONFLICT")
                if self.observer is not None:
                    self.observer.tensor_registered(tensor, stable_ref, self.stage)
                return tensor
        self.tensors.append((tensor, stable_ref))
        if self.observer is not None:
            self.observer.tensor_registered(tensor, stable_ref, self.stage)
        return tensor

    def reference(self, tensor: torch.Tensor) -> str:
        for existing, stable_ref in self.tensors:
            if existing is tensor:
                return stable_ref
        if self.observer is not None:
            stable_ref = self.observer.stable_ref_for_tensor(tensor)
            if stable_ref is not None:
                self.tensors.append((tensor, stable_ref))
                return stable_ref
        raise RuntimeError("NATIVE_WORKLOAD_TENSOR_UNREGISTERED")

    def override(self, stable_ref: str, baseline: torch.Tensor) -> torch.Tensor:
        if stable_ref not in self.source_overrides:
            return baseline
        value = torch.as_tensor(
            self.source_overrides[stable_ref],
            dtype=baseline.dtype,
            device=baseline.device,
        ).detach().clone()
        if value.shape != baseline.shape:
            raise ValueError(f"NATIVE_SOURCE_OVERRIDE_SHAPE_MISMATCH:{stable_ref}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"NATIVE_SOURCE_OVERRIDE_NONFINITE:{stable_ref}")
        value.requires_grad_(baseline.requires_grad)
        return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "requires_grad": bool(tensor.requires_grad),
        "shape": list(tensor.shape),
        "value": tensor.detach().cpu().tolist(),
    }


def _configure() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("NATIVE_PYTORCH_INTEROP_THREAD_PROFILE_DRIFT") from exc
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("NATIVE_ORACLE_CUDA_MUST_BE_DISABLED")


def _source(
    runtime: _Runtime,
    stable_ref: str,
    baseline: torch.Tensor,
    *,
    source_identity: str,
    source_role: str,
    version: str,
    selected: bool = True,
) -> torch.Tensor:
    tensor = runtime.override(stable_ref, baseline)
    runtime.reserve()
    runtime.register(tensor, stable_ref)
    if runtime.observer is not None:
        runtime.observer.source_registered(
            tensor,
            stable_ref,
            source_identity=source_identity,
            source_role=source_role,
            version=version,
            selected=selected,
        )
    return tensor


def _result(runtime: _Runtime, tensor: torch.Tensor) -> torch.Tensor:
    ordinal = runtime.reserve()
    return runtime.register(
        tensor,
        f"{runtime.spec.step_key}:{runtime.stage}:tensor:{ordinal}",
    )


def _standard(
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
        product = _result(runtime, torch.matmul(inputs["x"], parameters["w"]))
        shifted = _result(runtime, torch.add(product, parameters["b"]))
        hidden = _result(runtime, torch.relu(shifted))
        if name == "linear_chain":
            output = _result(runtime, torch.pow(hidden, 2.0))
            loss = _result(runtime, torch.mean(output))
        else:
            externals["scale"] = _source(
                runtime,
                "source:external:scale",
                torch.tensor(1.25),
                source_identity="external_scale",
                source_role="external_state",
                version="forward",
            )
            left = _result(runtime, torch.mul(hidden, externals["scale"]))
            right = _result(runtime, torch.pow(hidden, 2.0))
            output = _result(runtime, torch.add(left, right))
            loss = _result(runtime, torch.sum(output))
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
        shared = _result(runtime, torch.mul(inputs["x"], parameters["w"]))
        output = _result(runtime, torch.add(shared, shared))
        loss = _result(runtime, torch.sum(output))
    elif name == "duplicate_valued_distinct_sources":
        for key, identity in (
            ("x1", runtime.spec.sample_identity),
            ("x2", runtime.spec.sample_identity + "_peer"),
        ):
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
        left = _result(runtime, torch.mul(inputs["x1"], parameters["w"]))
        right = _result(runtime, torch.mul(inputs["x2"], parameters["w"]))
        output = _result(runtime, torch.add(left, right))
        loss = _result(runtime, torch.sum(output))
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
        output = _result(runtime, torch.mul(parameters["p_zero"], externals["zero"]))
        loss = _result(runtime, torch.sum(output))
    else:
        raise ValueError(f"NATIVE_WORKLOAD_UNKNOWN:{name}")
    return inputs, parameters, externals, output, loss


def _checkpoint(
    runtime: _Runtime,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    mode = runtime.spec.checkpoint_mode
    if mode not in {
        "no_checkpoint",
        "stable",
        "divergent",
        "replay_stable",
        "replay_divergent",
    }:
        raise ValueError(f"NATIVE_CHECKPOINT_MODE_UNKNOWN:{mode}")
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
    replay = mode.startswith("replay_")
    scale_ref = (
        "source:external:scale:recomputation"
        if replay
        else "source:external:scale:forward"
    )
    scale_state = {"value": _source(
        runtime,
        scale_ref,
        torch.tensor(2.0 if mode == "replay_divergent" else 1.0),
        source_identity=(
            "external_scale_recomputation_divergent"
            if mode == "replay_divergent"
            else (
                "external_scale_recomputation_stable"
                if replay
                else "external_scale_forward"
            )
        ),
        source_role="external_state",
        version="backward_recomputation" if replay else "original_forward",
    )}

    @contextlib.contextmanager
    def phase(stage: str):
        previous = runtime.stage
        runtime.stage = stage
        runtime.reserve()
        try:
            yield
        finally:
            runtime.reserve()
            runtime.stage = previous

    def contexts():
        return phase("forward_original"), phase("backward_recomputation")

    def block(current_parameter: torch.Tensor, current_sample: torch.Tensor) -> torch.Tensor:
        runtime.register(current_parameter, "source:parameter:p:before")
        runtime.register(current_sample, "source:sample:x")
        runtime.register(scale_state["value"], runtime.reference(scale_state["value"]))
        product = _result(runtime, torch.mul(current_parameter, current_sample))
        scaled = _result(runtime, torch.mul(product, scale_state["value"]))
        return _result(runtime, torch.sin(scaled))

    if replay:
        runtime.stage = "backward_recomputation"
        output = block(parameter, sample)
        runtime.stage = "forward"
    elif mode == "no_checkpoint":
        runtime.stage = "forward_original"
        output = block(parameter, sample)
        runtime.stage = "forward"
    else:
        output = checkpoint(
            block,
            parameter,
            sample,
            use_reentrant=False,
            context_fn=contexts,
            determinism_check="default",
            early_stop=False,
        )
        runtime.register(output, runtime.reference(output))
    loss = _result(runtime, torch.sum(output))
    externals = {
        ("scale_recomputation" if replay else "scale_forward"): scale_state["value"]
    }
    if mode in {"stable", "divergent"}:
        value = 1.0 if mode == "stable" else 2.0
        replacement = _source(
            runtime,
            "source:external:scale:recomputation",
            torch.tensor(value),
            source_identity=(
                "external_scale_recomputation_stable"
                if value == 1.0
                else "external_scale_recomputation_divergent"
            ),
            source_role="external_state",
            version="backward_recomputation",
        )
        scale_state["value"] = replacement
        externals["scale_recomputation"] = replacement
    return {"x": sample}, {"p": parameter}, externals, output, loss


def _optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        name: {
            key: _payload(value) if isinstance(value, torch.Tensor) else value
            for key, value in sorted(optimizer.state.get(parameters[name], {}).items())
        }
        for name in sorted(parameters)
    }


def _parameter_values(parameters: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _payload(parameters[name]) for name in sorted(parameters)}


def _gradient_values(leaves: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: None if leaves[name].grad is None else _payload(leaves[name].grad)
        for name in sorted(leaves)
    }


def run_native_oracle_training_step(
    spec: NativeTrainingSpec,
    *,
    observer: ExecutionObserver | None = None,
    source_value_overrides: dict[str, Any] | None = None,
) -> NativeTrainingRun:
    _configure()
    runtime = _Runtime(spec, observer, source_value_overrides)
    context = (
        observer.saved_tensor_context(lambda: runtime.stage)
        if observer is not None
        else contextlib.nullcontext()
    )
    with context:
        if spec.workload == "checkpoint_external_state":
            inputs, parameters, externals, output, loss = _checkpoint(runtime)
        else:
            inputs, parameters, externals, output, loss = _standard(runtime)
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
        runtime.reserve()
        leaves = {
            **{f"input:{name}": tensor for name, tensor in inputs.items()},
            **{f"parameter:{name}": tensor for name, tensor in parameters.items()},
        }
        for leaf in (leaves[key] for key in sorted(leaves)):
            if leaf.requires_grad:
                runtime.reserve()
        runtime.reserve()
        if observer is not None:
            observer.before_backward(loss, leaves)
        try:
            loss.backward()
        finally:
            if observer is not None:
                observer.after_backward(leaves)
    optimizer.step()
    result = {
        "device": "cpu",
        "dtype": "torch.float64",
        "exception": None,
        "external_values": {
            name: _payload(value) for name, value in sorted(externals.items())
        },
        "forward_output": _payload(output),
        "gradients": _gradient_values(leaves),
        "input_values": {name: _payload(value) for name, value in sorted(inputs.items())},
        "loss": _payload(loss),
        "optimizer_state_after": _optimizer_state(optimizer, parameters),
        "optimizer_state_before": optimizer_state_before,
        "parameter_after": _parameter_values(parameters),
        "parameter_before": parameter_before,
        "workload": spec.workload,
    }
    return NativeTrainingRun(spec, result, _canonical(result) + b"\n")
