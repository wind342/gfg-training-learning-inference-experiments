from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch

from .graph_canonicalization import canonicalize_graph
from .native_oracle_workloads import (
    NativeTrainingSpec,
    run_native_oracle_training_step,
)
from .saved_tensor_observer import (
    EventClock,
    SavedTensorObserver,
    canonical_bytes,
    gradient_descriptors,
    tensor_descriptor,
)


def _find_identity(values: list[torch.autograd.graph.Node], target: torch.autograd.graph.Node) -> int | None:
    for index, value in enumerate(values):
        if value is target:
            return index
    return None


def _live_graph(
    loss: torch.Tensor,
) -> tuple[dict[str, Any], list[torch.autograd.graph.Node], dict[int, str]]:
    root = loss.grad_fn
    if root is None:
        raise RuntimeError("NATIVE_BACKWARD_ROOT_MISSING")
    nodes: list[torch.autograd.graph.Node] = [root]
    queue = [root]
    visited: list[torch.autograd.graph.Node] = []
    raw_nodes = []
    raw_edges = []
    paths: dict[int, list[tuple[int, ...]]] = {0: []}

    def visit_paths(index: int, path: tuple[int, ...], active: frozenset[int]) -> None:
        if index in active:
            raise RuntimeError("NATIVE_BACKWARD_GRAPH_CYCLE")
        paths.setdefault(index, []).append(path)
        node = nodes[index]
        for slot, (target, _output_nr) in enumerate(node.next_functions):
            if target is None:
                continue
            target_index = _find_identity(nodes, target)
            if target_index is None:
                nodes.append(target)
                target_index = len(nodes) - 1
            visit_paths(target_index, (*path, slot), active | {index})

    visit_paths(0, (), frozenset())
    while queue:
        node = queue.pop(0)
        if any(node is item for item in visited):
            continue
        visited.append(node)
        source_index = _find_identity(nodes, node)
        if source_index is None:
            raise RuntimeError("NATIVE_BACKWARD_NODE_IDENTITY_LOST")
        raw_nodes.append({"key": f"runtime_node_{source_index}", "node_type": node.name()})
        for slot, (target, output_nr) in enumerate(node.next_functions):
            target_key = None
            if target is not None:
                target_index = _find_identity(nodes, target)
                if target_index is None:
                    nodes.append(target)
                    target_index = len(nodes) - 1
                target_key = f"runtime_node_{target_index}"
                queue.append(target)
            raw_edges.append({
                "output_nr": int(output_nr),
                "slot": slot,
                "source_key": f"runtime_node_{source_index}",
                "target_key": target_key,
            })
    raw = {"edges": raw_edges, "nodes": raw_nodes, "root_key": "runtime_node_0"}
    graph = canonicalize_graph(raw)
    node_ids = {
        index: "agn_" + hashlib.sha256(canonical_bytes({
            "node_type": node.name(),
            "root_paths": [
                "root" if not path else "root/" + "/".join(str(slot) for slot in path)
                for path in sorted(set(paths[index]))
            ],
        })).hexdigest()
        for index, node in enumerate(nodes)
    }
    if {row["node_id"] for row in graph["nodes"]} != set(node_ids.values()):
        raise RuntimeError("NATIVE_BACKWARD_CANONICAL_NODE_ALIGNMENT_FAILED")
    return graph, nodes, node_ids


class NativeBackwardTracer:
    def __init__(
        self,
        loss: torch.Tensor,
        leaves: dict[str, torch.Tensor],
        saved: SavedTensorObserver,
        clock: EventClock,
    ) -> None:
        self.graph, self.nodes, self.node_ids = _live_graph(loss)
        self.leaves = leaves
        self.saved = saved
        self.clock = clock
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.executions: list[dict[str, Any]] = []
        self.gradient_slots: list[dict[str, Any]] = []
        self.leaf_gradients: list[dict[str, Any]] = []
        self._active: dict[str, dict[str, Any]] = {}

    def attach(self) -> None:
        for index, node in enumerate(self.nodes):
            node_id = self.node_ids[index]
            node_type = node.name()

            def prehook(
                grad_outputs: tuple[torch.Tensor | None, ...],
                *,
                fixed_id: str = node_id,
                fixed_type: str = node_type,
            ) -> None:
                if fixed_id in self._active:
                    raise RuntimeError("NATIVE_NODE_PREHOOK_DUPLICATE_ACTIVE")
                ordinal = self.clock.record(
                    "node_prehook",
                    native_node_id=fixed_id,
                    native_node_type=fixed_type,
                )
                self._active[fixed_id] = {
                    "grad_outputs": gradient_descriptors(grad_outputs),
                    "native_node_id": fixed_id,
                    "native_node_type": fixed_type,
                    "prehook_ordinal": ordinal,
                }
                self.saved.enter_node(fixed_id, fixed_type)
                return None

            def posthook(
                grad_inputs: tuple[torch.Tensor | None, ...],
                grad_outputs: tuple[torch.Tensor | None, ...],
                *,
                fixed_id: str = node_id,
                fixed_type: str = node_type,
            ) -> None:
                active = self._active.pop(fixed_id, None)
                if active is None:
                    raise RuntimeError("NATIVE_NODE_POSTHOOK_WITHOUT_PREHOOK")
                ordinal = self.clock.record(
                    "node_posthook",
                    native_node_id=fixed_id,
                    native_node_type=fixed_type,
                )
                self.saved.exit_node(fixed_id)
                execution = {
                    **active,
                    "posthook_ordinal": ordinal,
                }
                self.executions.append(execution)
                self.gradient_slots.append({
                    "grad_inputs": gradient_descriptors(grad_inputs),
                    "grad_outputs": gradient_descriptors(grad_outputs),
                    "native_node_id": fixed_id,
                    "native_node_type": fixed_type,
                    "posthook_ordinal": ordinal,
                    "prehook_ordinal": active["prehook_ordinal"],
                })
                return None

            self.handles.append(node.register_prehook(prehook))
            self.handles.append(node.register_hook(posthook))

        for leaf_ref in sorted(self.leaves):
            leaf = self.leaves[leaf_ref]
            if not leaf.requires_grad:
                continue

            def leaf_hook(
                gradient: torch.Tensor,
                *,
                fixed_ref: str = leaf_ref,
            ) -> None:
                self.leaf_gradients.append({
                    "gradient": tensor_descriptor(gradient),
                    "leaf_ref": fixed_ref,
                    "ordinal": self.clock.record("leaf_gradient_hook", leaf_ref=fixed_ref),
                })
                return None

            self.handles.append(leaf.register_hook(leaf_hook))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def export(self) -> dict[str, Any]:
        pre_counts: dict[str, int] = {}
        post_counts: dict[str, int] = {}
        for row in self.clock.rows:
            node_id = row.get("native_node_id")
            if row["event"] == "node_prehook":
                pre_counts[node_id] = pre_counts.get(node_id, 0) + 1
            elif row["event"] == "node_posthook":
                post_counts[node_id] = post_counts.get(node_id, 0) + 1
        executed = sorted(set(pre_counts) | set(post_counts))
        return {
            "all_executed_nodes_paired": all(
                pre_counts.get(node_id) == post_counts.get(node_id) == 1
                for node_id in executed
            ),
            "execution_order": [row["native_node_id"] for row in sorted(
                self.executions,
                key=lambda row: row["prehook_ordinal"],
            )],
            "executions": sorted(self.executions, key=lambda row: row["prehook_ordinal"]),
            "gradient_slots": sorted(self.gradient_slots, key=lambda row: row["prehook_ordinal"]),
            "leaf_gradient_hooks": sorted(self.leaf_gradients, key=lambda row: row["ordinal"]),
            "native_graph": self.graph,
            "shared_nodes_execute_once": all(
                pre_counts.get(row["node_id"], 0) == 1
                for row in self.graph["nodes"]
                if row["is_shared"]
            ),
        }


class NativeBackwardDependencyController:
    def __init__(
        self,
        *,
        intervention_token: str | None = None,
        perturbation: str | None = None,
        source_replay_ref: str | None = None,
        baseline_token_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.clock = EventClock()
        self.saved = SavedTensorObserver(
            self.clock,
            intervention_token=intervention_token,
            perturbation=perturbation,
            source_replay_ref=source_replay_ref,
            baseline_token_values=baseline_token_values,
        )
        self.tracer: NativeBackwardTracer | None = None
        self._result: dict[str, Any] | None = None

    def saved_tensor_context(
        self,
        stage_getter: Callable[[], str],
    ) -> Any:
        return self.saved.context(stage_getter)

    def tensor_registered(
        self,
        tensor: torch.Tensor,
        stable_ref: str,
        stage: str,
    ) -> None:
        self.saved.tensor_registered(tensor, stable_ref, stage)

    def source_registered(
        self,
        tensor: torch.Tensor,
        stable_ref: str,
        *,
        source_identity: str,
        source_role: str,
        version: str,
        selected: bool,
    ) -> None:
        self.saved.source_registered(
            tensor,
            stable_ref,
            source_identity=source_identity,
            source_role=source_role,
            version=version,
            selected=selected,
        )

    def stable_ref_for_tensor(self, tensor: torch.Tensor) -> str | None:
        return self.saved.stable_ref_for_tensor(tensor)

    def before_backward(
        self,
        loss: torch.Tensor,
        leaves: dict[str, torch.Tensor],
    ) -> None:
        self.tracer = NativeBackwardTracer(loss, leaves, self.saved, self.clock)
        self.tracer.attach()

    def after_backward(self, leaves: dict[str, torch.Tensor]) -> None:
        del leaves
        if self.tracer is None:
            raise RuntimeError("NATIVE_BACKWARD_TRACER_MISSING")
        self.tracer.detach()
        tracer = self.tracer.export()
        saved = self.saved.export()
        assigned = [row for row in saved["unpack_trace"] if row["assigned"]]
        self._result = {
            "backward": tracer,
            "hook_ordering": self.clock.rows,
            "saved_tensors": saved,
            "status": (
                "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED"
                if tracer["all_executed_nodes_paired"]
                and tracer["shared_nodes_execute_once"]
                else "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_NOT_ESTABLISHED"
            ),
            "unassigned_unpack_count": len(saved["unpack_trace"]) - len(assigned),
        }

    def result(self) -> dict[str, Any]:
        if self._result is None:
            raise RuntimeError("NATIVE_BACKWARD_RESULT_NOT_READY")
        return self._result

    def packed_values(self) -> dict[str, torch.Tensor]:
        return self.saved.packed_values()


def observe_native_backward(spec: NativeTrainingSpec) -> dict[str, Any]:
    plain = run_native_oracle_training_step(spec)
    controller = NativeBackwardDependencyController()
    observed = run_native_oracle_training_step(spec, observer=controller)
    result = controller.result()
    result["baseline_ordinary_bytes_exact"] = plain.ordinary_bytes == observed.ordinary_bytes
    result["baseline_gradients_exact"] = (
        plain.ordinary_result["gradients"] == observed.ordinary_result["gradients"]
    )
    result["ordinary_result"] = observed.ordinary_result
    result["status"] = (
        "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED"
        if result["status"] == "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED"
        and result["baseline_ordinary_bytes_exact"]
        and result["baseline_gradients_exact"]
        and result["unassigned_unpack_count"] == 0
        else "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_NOT_ESTABLISHED"
    )
    return result
