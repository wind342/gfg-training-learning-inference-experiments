from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch

from .native_backward_dependency_oracle import NativeBackwardDependencyController
from .native_oracle_workloads import (
    NativeTrainingSpec,
    run_native_oracle_training_step,
)
from .saved_tensor_observer import (
    FROZEN_PERTURBATIONS,
    canonical_bytes,
    frozen_perturbation,
    tensor_descriptor,
)


STANDARD_WORKLOADS = (
    "branch_and_merge",
    "duplicate_valued_distinct_sources",
    "linear_chain",
    "shared_tensor_reuse",
    "zero_gradient_and_unused_sources",
)
CHECKPOINT_MODES = ("divergent", "no_checkpoint", "stable")


def declared_specs() -> list[tuple[str, NativeTrainingSpec]]:
    return [
        *((name, NativeTrainingSpec(name)) for name in STANDARD_WORKLOADS),
        *((
            f"checkpoint:{mode}",
            NativeTrainingSpec("checkpoint_external_state", checkpoint_mode=mode),
        ) for mode in CHECKPOINT_MODES),
    ]


def _gradient_map(ordinary_result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"step_0:gradient:{leaf_ref}": value
        for leaf_ref, value in sorted(ordinary_result["gradients"].items())
    }


def _changed_gradients(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
) -> tuple[list[str], list[str]]:
    keys = sorted(set(baseline) | set(intervention))
    changed = [
        key for key in keys
        if canonical_bytes(baseline.get(key)) != canonical_bytes(intervention.get(key))
    ]
    return changed, [key for key in keys if key not in changed]


def _gradients_finite(gradients: dict[str, Any]) -> bool:
    return all(
        descriptor is None
        or bool(torch.isfinite(torch.as_tensor(descriptor["value"])).all())
        for descriptor in gradients.values()
    )


def _observed_run(
    spec: NativeTrainingSpec,
    *,
    intervention_token: str | None = None,
    perturbation: str | None = None,
    source_replay_ref: str | None = None,
    baseline_token_values: dict[str, torch.Tensor] | None = None,
    source_value_overrides: dict[str, Any] | None = None,
) -> tuple[Any, NativeBackwardDependencyController]:
    controller = NativeBackwardDependencyController(
        intervention_token=intervention_token,
        perturbation=perturbation,
        source_replay_ref=source_replay_ref,
        baseline_token_values=baseline_token_values,
    )
    run = run_native_oracle_training_step(
        spec,
        observer=controller,
        source_value_overrides=source_value_overrides,
    )
    return run, controller


def _baseline(key: str, spec: NativeTrainingSpec) -> dict[str, Any]:
    plain = run_native_oracle_training_step(spec)
    observed, controller = _observed_run(spec)
    observation = controller.result()
    return {
        "baseline_gradients": _gradient_map(plain.ordinary_result),
        "baseline_gradients_exact": (
            plain.ordinary_result["gradients"] == observed.ordinary_result["gradients"]
        ),
        "baseline_ordinary_bytes_exact": plain.ordinary_bytes == observed.ordinary_bytes,
        "graph_sha256": observation["backward"]["native_graph"]["canonical_graph_sha256"],
        "key": key,
        "observation": observation,
        "packed_values": controller.packed_values(),
        "relation_ref_translation": {},
        "spec": spec,
    }


def _intermediate_registration_refs(observation: dict[str, Any]) -> list[str]:
    result = []
    for row in observation["saved_tensors"]["tensor_registrations"]:
        stable_ref = row["stable_tensor_ref"]
        if (
            row["stage"] == "backward_recomputation"
            and stable_ref.startswith("step_0:backward_recomputation:tensor:")
            and stable_ref not in result
        ):
            result.append(stable_ref)
    return result


def _checkpoint_replay_baseline(
    key: str,
    mode: str,
    actual: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = _baseline(
        key,
        NativeTrainingSpec(
            "checkpoint_external_state",
            checkpoint_mode=f"replay_{mode}",
        ),
    )
    actual_refs = _intermediate_registration_refs(actual["observation"])
    replay_refs = _intermediate_registration_refs(replay["observation"])
    if len(actual_refs) != len(replay_refs) or not actual_refs:
        raise RuntimeError("CHECKPOINT_REPLAY_REGISTRATION_ALIGNMENT_FAILED")
    replay["relation_ref_translation"] = dict(zip(replay_refs, actual_refs, strict=True))
    equivalence = {
        "actual_gradient_exact": (
            actual["baseline_gradients"] == replay["baseline_gradients"]
        ),
        "actual_graph_sha256": actual["graph_sha256"],
        "actual_intermediate_refs": actual_refs,
        "graph_topology_exact": actual["graph_sha256"] == replay["graph_sha256"],
        "mode": mode,
        "replay_graph_sha256": replay["graph_sha256"],
        "replay_intermediate_refs": replay_refs,
        "stable_ref_translation": replay["relation_ref_translation"],
        "workload_key": key,
    }
    equivalence["supported"] = all([
        equivalence["actual_gradient_exact"],
        equivalence["graph_topology_exact"],
        len(equivalence["stable_ref_translation"]) == len(actual_refs),
    ])
    return replay, equivalence


def _saved_tensor_interventions(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    relation_witnesses: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, baseline in sorted(baselines.items()):
        spec = baseline["spec"]
        pack_rows = {
            row["token_key"]: row
            for row in baseline["observation"]["saved_tensors"]["pack_trace"]
            if row["unpack_count"] > 0
        }
        baseline_unpacks = baseline["observation"]["saved_tensors"]["unpack_trace"]
        for token_key, pack_row in sorted(pack_rows.items()):
            for perturbation in FROZEN_PERTURBATIONS:
                attempt: dict[str, Any] = {
                    "exception": None,
                    "perturbation": perturbation,
                    "stable_tensor_ref": pack_row["stable_tensor_ref"],
                    "token_key": token_key,
                    "workload_key": key,
                }
                try:
                    run, controller = _observed_run(
                        spec,
                        intervention_token=token_key,
                        perturbation=perturbation,
                    )
                    observation = controller.result()
                    gradients = _gradient_map(run.ordinary_result)
                    changed, unchanged = _changed_gradients(
                        baseline["baseline_gradients"],
                        gradients,
                    )
                    applications = observation["saved_tensors"]["intervention_applications"]
                    graph_sha = observation["backward"]["native_graph"]["canonical_graph_sha256"]
                    attempt.update({
                        "baseline_gradients": baseline["baseline_gradients"],
                        "changed_target_gradients": changed,
                        "finite": _gradients_finite(gradients),
                        "graph_sha256": graph_sha,
                        "graph_topology_exact": graph_sha == baseline["graph_sha256"],
                        "intervention_applications": applications,
                        "intervention_gradients": gradients,
                        "intervention_tensor": (
                            applications[0]["result"] if applications else None
                        ),
                        "only_declared_token_intervened": bool(applications)
                        and all(row["token_key"] == token_key for row in applications),
                        "unchanged_target_gradients": unchanged,
                    })
                    if attempt["graph_topology_exact"] and attempt["only_declared_token_intervened"]:
                        actual_unpacks = [
                            row for row in baseline_unpacks if row["token_key"] == token_key
                        ]
                        dependency_key = baseline["relation_ref_translation"].get(
                            pack_row["stable_tensor_ref"],
                            pack_row["stable_tensor_ref"],
                        )
                        for target in changed:
                            relation_witnesses[(key, dependency_key, target)].append({
                                "actual_unpack_witnesses": actual_unpacks,
                                "perturbation": perturbation,
                                "replay_ref": pack_row["stable_tensor_ref"],
                                "token_key": token_key,
                            })
                except Exception as exc:  # scientific artifact records the exact failure
                    attempt.update({
                        "exception": f"{type(exc).__name__}:{exc}",
                        "finite": False,
                        "graph_topology_exact": False,
                        "only_declared_token_intervened": False,
                    })
                attempts.append(attempt)
    relations = []
    for (workload_key, dependency_key, target_key), witnesses in sorted(relation_witnesses.items()):
        relations.append({
            "actual_unpack_witnesses": [
                row
                for witness in witnesses
                for row in witness["actual_unpack_witnesses"]
            ],
            "dependency_key": dependency_key,
            "dependency_kind": "saved_tensor",
            "graph_sha256": baselines[workload_key]["graph_sha256"],
            "node_execution_witnesses": sorted({
                row["native_node_id"]
                for witness in witnesses
                for row in witness["actual_unpack_witnesses"]
                if row["native_node_id"] is not None
            }),
            "successful_interventions": [
                {"perturbation": row["perturbation"], "token_key": row["token_key"]}
                for row in witnesses
            ],
            "target_gradient_key": target_key,
            "workload_key": workload_key,
        })
    unpacked_tokens = sum(
        row["unpack_count"] > 0
        for baseline in baselines.values()
        for row in baseline["observation"]["saved_tensors"]["pack_trace"]
    )
    expected_attempts = unpacked_tokens * len(FROZEN_PERTURBATIONS)
    all_safe = all(
        row["exception"] is None
        and row["finite"]
        and row["graph_topology_exact"]
        and row["only_declared_token_intervened"]
        for row in attempts
    )
    return {
        "attempt_count": len(attempts),
        "attempts": attempts,
        "expected_attempt_count": expected_attempts,
        "relations": relations,
        "status": (
            "NATIVE_SAVED_TENSOR_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED"
            if len(attempts) == expected_attempts and all_safe
            else "INTERVENTION_ORACLE_NOT_ESTABLISHED"
        ),
        "unpacked_token_count": unpacked_tokens,
    }


def _source_interventions(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    relation_witnesses: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, baseline in sorted(baselines.items()):
        sources = baseline["observation"]["saved_tensors"]["registered_sources"]
        baseline_unpacks = baseline["observation"]["saved_tensors"]["unpack_trace"]
        for source in sources:
            source_ref = source["source_ref"]
            value = torch.tensor(source["tensor"]["value"], dtype=torch.float64).reshape(
                source["tensor"]["shape"]
            )
            for perturbation in FROZEN_PERTURBATIONS:
                replacement = frozen_perturbation(value, perturbation)
                attempt: dict[str, Any] = {
                    "exception": None,
                    "perturbation": perturbation,
                    "replacement": tensor_descriptor(replacement),
                    "source_identity": source["source_identity"],
                    "source_ref": source_ref,
                    "workload_key": key,
                }
                try:
                    run, controller = _observed_run(
                        baseline["spec"],
                        source_replay_ref=source_ref,
                        baseline_token_values=baseline["packed_values"],
                        source_value_overrides={source_ref: replacement.tolist()},
                    )
                    observation = controller.result()
                    gradients = _gradient_map(run.ordinary_result)
                    changed, unchanged = _changed_gradients(
                        baseline["baseline_gradients"], gradients
                    )
                    graph_sha = observation["backward"]["native_graph"]["canonical_graph_sha256"]
                    replayed_unpacks = [
                        row for row in observation["saved_tensors"]["intervention_applications"]
                        if row["stable_tensor_ref"] == source_ref
                    ]
                    registered = next(
                        row
                        for row in observation["saved_tensors"]["registered_sources"]
                        if row["source_ref"] == source_ref
                    )
                    other_saved_tensors_frozen = all(
                        row["stable_tensor_ref"] == source_ref
                        or canonical_bytes(row["result"]["value"])
                        == canonical_bytes(
                            baseline["packed_values"][row["token_key"]].tolist()
                        )
                        for row in observation["saved_tensors"]["unpack_trace"]
                    )
                    attempt.update({
                        "baseline_gradients": baseline["baseline_gradients"],
                        "changed_target_gradients": changed,
                        "finite": _gradients_finite(gradients),
                        "graph_sha256": graph_sha,
                        "graph_topology_exact": graph_sha == baseline["graph_sha256"],
                        "intervention_gradients": gradients,
                        "other_saved_tensors_frozen": other_saved_tensors_frozen,
                        "replayed_source_unpack_count": len(replayed_unpacks),
                        "source_identity_preserved": (
                            registered["source_identity"] == source["source_identity"]
                            and registered["source_ref"] == source_ref
                        ),
                        "unchanged_target_gradients": unchanged,
                    })
                    if attempt["graph_topology_exact"]:
                        actual_unpacks = [
                            row for row in baseline_unpacks
                            if row["stable_tensor_ref"] == source_ref
                        ]
                        for target in changed:
                            relation_witnesses[(key, source_ref, target)].append({
                                "actual_unpack_witnesses": actual_unpacks,
                                "perturbation": perturbation,
                            })
                except Exception as exc:
                    attempt.update({
                        "exception": f"{type(exc).__name__}:{exc}",
                        "finite": False,
                        "graph_topology_exact": False,
                        "other_saved_tensors_frozen": False,
                        "source_identity_preserved": True,
                    })
                attempts.append(attempt)
    relations = []
    for (workload_key, dependency_key, target_key), witnesses in sorted(relation_witnesses.items()):
        relations.append({
            "actual_unpack_witnesses": [
                row
                for witness in witnesses
                for row in witness["actual_unpack_witnesses"]
            ],
            "dependency_key": dependency_key,
            "dependency_kind": "registered_source_replay",
            "graph_sha256": baselines[workload_key]["graph_sha256"],
            "node_execution_witnesses": sorted({
                row["native_node_id"]
                for witness in witnesses
                for row in witness["actual_unpack_witnesses"]
                if row["native_node_id"] is not None
            }),
            "successful_interventions": [
                {"perturbation": row["perturbation"]} for row in witnesses
            ],
            "target_gradient_key": target_key,
            "workload_key": workload_key,
        })
    expected_attempts = sum(
        len(row["observation"]["saved_tensors"]["registered_sources"])
        for row in baselines.values()
    ) * len(FROZEN_PERTURBATIONS)
    all_safe = all(
        row["exception"] is None
        and row["finite"]
        and row["graph_topology_exact"]
        and row["other_saved_tensors_frozen"]
        and row["source_identity_preserved"]
        for row in attempts
    )
    return {
        "attempt_count": len(attempts),
        "attempts": attempts,
        "expected_attempt_count": expected_attempts,
        "relations": relations,
        "status": (
            "NATIVE_SOURCE_REPLAY_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED"
            if len(attempts) == expected_attempts and all_safe
            else "NATIVE_SOURCE_REPLAY_GRADIENT_DEPENDENCY_ORACLE_NOT_ESTABLISHED"
        ),
    }


def _merge_relations(
    saved_relations: list[dict[str, Any]],
    source_relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation in [*saved_relations, *source_relations]:
        key = (
            relation["workload_key"],
            relation["dependency_key"],
            relation["target_gradient_key"],
        )
        if key not in merged:
            merged[key] = deepcopy(relation)
            merged[key]["dependency_kind"] = [relation["dependency_kind"]]
        else:
            row = merged[key]
            row["dependency_kind"] = sorted(set([
                *row["dependency_kind"], relation["dependency_kind"],
            ]))
            row["actual_unpack_witnesses"].extend(relation["actual_unpack_witnesses"])
            row["node_execution_witnesses"] = sorted(set([
                *row["node_execution_witnesses"], *relation["node_execution_witnesses"],
            ]))
            row["successful_interventions"].extend(relation["successful_interventions"])
    return [merged[key] for key in sorted(merged)]


def run_gradient_intervention_oracle() -> dict[str, Any]:
    actual_baselines = {key: _baseline(key, spec) for key, spec in declared_specs()}
    dependency_baselines = dict(actual_baselines)
    replay_equivalence = []
    for mode in ("stable", "divergent"):
        key = f"checkpoint:{mode}"
        replay, equivalence = _checkpoint_replay_baseline(
            key,
            mode,
            actual_baselines[key],
        )
        dependency_baselines[key] = replay
        replay_equivalence.append(equivalence)
    saved = _saved_tensor_interventions(dependency_baselines)
    source = _source_interventions(dependency_baselines)
    relations = _merge_relations(saved["relations"], source["relations"])
    baseline_observations = {
        key: {
            "baseline_gradients": row["baseline_gradients"],
            "baseline_gradients_exact": row["baseline_gradients_exact"],
            "baseline_ordinary_bytes_exact": row["baseline_ordinary_bytes_exact"],
            "graph_sha256": row["graph_sha256"],
            "observation": row["observation"],
        }
        for key, row in sorted(actual_baselines.items())
    }
    baseline_exact = all(
        row["baseline_gradients_exact"]
        and row["baseline_ordinary_bytes_exact"]
        and row["observation"]["status"]
        == "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED"
        for row in baseline_observations.values()
    )
    supported = all([
        baseline_exact,
        saved["status"] == "NATIVE_SAVED_TENSOR_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        source["status"] == "NATIVE_SOURCE_REPLAY_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
    ])
    return {
        "baseline_observations": baseline_observations,
        "checkpoint_recomputation_replay_equivalence": replay_equivalence,
        "native_gradient_dependency_oracle": {
            "relation_count": len(relations),
            "relations": relations,
            "status": (
                "NATIVE_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED"
                if supported
                else "NATIVE_GRADIENT_DEPENDENCY_ORACLE_NOT_ESTABLISHED"
            ),
        },
        "saved_tensor_interventions": saved,
        "source_replay_interventions": source,
    }
