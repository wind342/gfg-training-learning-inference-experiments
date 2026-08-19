from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)

from .local_response_crosscheck_validator import (
    EXACT_H1_PARAMETER_BRANCH,
    _exact_h1_arrays,
    _metrics,
)
from .reciprocal_validator import _load_tensor
from .storage import TensorStore


def _response_record(root: Path, receiver_label: str) -> tuple[Path, dict[str, Any]]:
    receiver_root = root / f"receiver-{receiver_label}"
    record = read_json(receiver_root / "local_response_jk.json")
    require(record["schema"] == "nanogpt-local-response-jk-v1", "SST_RESPONSE_TRANSPORT_SCHEMA_INVALID")
    require(record["receiver_label"] == receiver_label, "SST_RESPONSE_TRANSPORT_RECEIVER_MISMATCH")
    return receiver_root, record


def _numeric_tensor(
    receiver_root: Path,
    record: dict[str, Any],
    role: str,
    field: str,
) -> np.ndarray:
    require(role in record["numeric_responses"], f"SST_RESPONSE_TRANSPORT_NUMERIC_ROLE_MISSING:{role}")
    references = record["numeric_responses"][role]
    require(field in references, f"SST_RESPONSE_TRANSPORT_NUMERIC_FIELD_MISSING:{role}:{field}")
    return _load_tensor(receiver_root, references[field]).astype(np.float64, copy=False)


def _categorical_signature(receiver_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, references in sorted(record["categorical_transitions"].items()):
        result[role] = {
            field: references[field]["raw_tensor_sha256"]
            for field in ("plus", "minus", "plus_changed_mask", "minus_changed_mask")
        }
    return result


def validate_response_transport_cross(
    *,
    reciprocal_root: Path,
    a_skip_root: Path,
    b_skip_root: Path,
    a_native_full_root: Path,
    b_native_full_root: Path,
    a_native_full_protocol_path: Path,
    b_native_full_protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    require(not output_root.exists(), "SST_RESPONSE_TRANSPORT_OUTPUT_ALREADY_EXISTS")
    output_root.mkdir(parents=True)
    store = TensorStore(output_root / "tensor-objects")

    skip_roots = {"A": a_skip_root, "B": b_skip_root}
    native_full_roots = {"A": a_native_full_root, "B": b_native_full_root}
    protocol_paths = {"A": a_native_full_protocol_path, "B": b_native_full_protocol_path}
    protocols = {label: read_json(path) for label, path in protocol_paths.items()}

    for donor_label in ("A", "B"):
        require(
            protocols[donor_label]["donor_update"]["label"] == donor_label,
            f"SST_RESPONSE_TRANSPORT_DONOR_MISMATCH:{donor_label}",
        )
        require(
            protocols[donor_label]["receiver_state_kind"] == "native_full",
            f"SST_RESPONSE_TRANSPORT_STATE_KIND_INVALID:{donor_label}",
        )
        for root, expected_kind in (
            (skip_roots[donor_label], "skip"),
            (native_full_roots[donor_label], "native_full"),
        ):
            validation = read_json(root / "local_response_jk_validation.json")
            require(validation["status"] == "PASS", f"SST_RESPONSE_TRANSPORT_INPUT_NOT_VALIDATED:{root}")
            pair = read_json(root / "local_response_pair_receipt.json")
            actual_kind = str(pair.get("receiver_state_kind", "skip"))
            require(actual_kind == expected_kind, f"SST_RESPONSE_TRANSPORT_INPUT_STATE_KIND_MISMATCH:{root}")

    rows: dict[str, Any] = {}
    strict_contexts = 0
    falsified_contexts = 0
    total_contexts = 0
    check_count = 0

    for donor_label in ("A", "B"):
        donor_rows: dict[str, Any] = {}
        for receiver_label in ("A", "B"):
            skip_receiver_root, skip_record = _response_record(skip_roots[donor_label], receiver_label)
            full_receiver_root, full_record = _response_record(native_full_roots[donor_label], receiver_label)
            require(
                str(full_record.get("receiver_state_kind")) == "native_full",
                f"SST_RESPONSE_TRANSPORT_FULL_RECORD_KIND_INVALID:{donor_label}:{receiver_label}",
            )
            require(
                float(skip_record["epsilon"]) == float(full_record["epsilon"]),
                f"SST_RESPONSE_TRANSPORT_EPSILON_MISMATCH:{donor_label}:{receiver_label}",
            )
            require(
                int(skip_record["horizon"]) == int(full_record["horizon"]) == 1,
                f"SST_RESPONSE_TRANSPORT_HORIZON_MISMATCH:{donor_label}:{receiver_label}",
            )

            exact_branch = EXACT_H1_PARAMETER_BRANCH[donor_label][receiver_label]
            skip_baseline, skip_effect = _exact_h1_arrays(
                reciprocal_root,
                receiver_label=receiver_label,
                branch=exact_branch,
            )
            roles = sorted(
                set(skip_record["numeric_responses"])
                & set(full_record["numeric_responses"])
                & set(skip_baseline)
                & set(skip_effect)
            )
            numeric_rows: dict[str, Any] = {}
            changed_count = 0
            nonzero_count = 0
            full_endpoint_better_count = 0
            skip_endpoint_better_count = 0
            full_endpoint_tie_count = 0
            skip_endpoint_tie_count = 0

            for role in roles:
                if not np.issubdtype(skip_baseline[role].dtype, np.floating):
                    continue
                skip_j = _numeric_tensor(skip_receiver_root, skip_record, role, "j_first_order")
                full_j = _numeric_tensor(full_receiver_root, full_record, role, "j_first_order")
                full_target = _numeric_tensor(full_receiver_root, full_record, role, "full_delta")
                skip_target = (
                    skip_effect[role].astype(np.float64, copy=False)
                    - skip_baseline[role].astype(np.float64, copy=False)
                )
                require(
                    skip_j.shape == full_j.shape == skip_target.shape == full_target.shape,
                    f"SST_RESPONSE_TRANSPORT_SHAPE_MISMATCH:{donor_label}:{receiver_label}:{role}",
                )
                transport = full_j - skip_j
                transport_reference = store.put(
                    transport,
                    representation=(
                        f"response-transport:{donor_label}:{receiver_label}:{role}:"
                        "native_full_j_minus_skip_j"
                    ),
                )
                skip_on_skip = _metrics(skip_j, skip_target)
                full_on_skip = _metrics(full_j, skip_target)
                skip_on_full = _metrics(skip_j, full_target)
                full_on_full = _metrics(full_j, full_target)
                changed = not np.array_equal(skip_j, full_j)
                nonzero = bool(
                    np.linalg.norm(skip_target) > 0.0
                    or np.linalg.norm(full_target) > 0.0
                    or np.linalg.norm(skip_j) > 0.0
                    or np.linalg.norm(full_j) > 0.0
                )
                full_better = full_on_full["residual_rms"] < skip_on_full["residual_rms"]
                skip_better = skip_on_skip["residual_rms"] < full_on_skip["residual_rms"]
                full_tie = full_on_full["residual_rms"] == skip_on_full["residual_rms"]
                skip_tie = skip_on_skip["residual_rms"] == full_on_skip["residual_rms"]
                changed_count += int(changed)
                nonzero_count += int(nonzero)
                if nonzero:
                    full_endpoint_better_count += int(full_better)
                    skip_endpoint_better_count += int(skip_better)
                    full_endpoint_tie_count += int(full_tie)
                    skip_endpoint_tie_count += int(skip_tie)
                numeric_rows[role] = {
                    "changed_across_native_transition": changed,
                    "native_full_j_on_native_full_target": full_on_full,
                    "skip_j_on_native_full_target": skip_on_full,
                    "native_full_j_on_skip_target": full_on_skip,
                    "skip_j_on_skip_target": skip_on_skip,
                    "native_full_j_better_on_native_full_target": full_better,
                    "skip_j_better_on_skip_target": skip_better,
                    "nonzero_role": nonzero,
                    "response_transport": transport_reference,
                    "shape": list(skip_j.shape),
                }
                check_count += 8

            skip_categorical = _categorical_signature(skip_receiver_root, skip_record)
            full_categorical = _categorical_signature(full_receiver_root, full_record)
            categorical_roles = sorted(set(skip_categorical) | set(full_categorical))
            categorical_changed_count = sum(
                1
                for role in categorical_roles
                if skip_categorical.get(role) != full_categorical.get(role)
            )
            check_count += len(categorical_roles)

            strict_support = (
                nonzero_count > 0
                and changed_count >= nonzero_count
                and full_endpoint_better_count == nonzero_count
                and skip_endpoint_better_count == nonzero_count
            )
            byte_identical = changed_count == 0 and categorical_changed_count == 0
            no_registration_improvement = (
                full_endpoint_better_count == 0 and skip_endpoint_better_count == 0
            )
            falsified = byte_identical or no_registration_improvement
            context_outcome = (
                "STRICT_TRANSPORT_SUPPORT"
                if strict_support
                else "TRANSPORT_FALSIFIED"
                if falsified
                else "MIXED_TRANSPORT_EVIDENCE"
            )
            strict_contexts += int(strict_support)
            falsified_contexts += int(falsified)
            total_contexts += 1
            donor_rows[receiver_label] = {
                "exact_skip_h1_branch": exact_branch,
                "numeric_role_count": len(numeric_rows),
                "nonzero_numeric_role_count": nonzero_count,
                "response_changed_count": changed_count,
                "native_full_j_better_on_native_full_target_count": full_endpoint_better_count,
                "skip_j_better_on_skip_target_count": skip_endpoint_better_count,
                "native_full_target_tie_count": full_endpoint_tie_count,
                "skip_target_tie_count": skip_endpoint_tie_count,
                "categorical_role_count": len(categorical_roles),
                "categorical_response_changed_count": categorical_changed_count,
                "complete_response_object_byte_identical": byte_identical,
                "strict_transport_support": strict_support,
                "transport_falsified": falsified,
                "mechanical_outcome": context_outcome,
                "numeric": numeric_rows,
            }
        rows[donor_label] = donor_rows

    strict_transport_hypothesis_satisfied = strict_contexts == total_contexts
    response_transport_falsified = falsified_contexts > 0
    overall_outcome = (
        "STRICT_TRANSPORT_SUPPORT"
        if strict_transport_hypothesis_satisfied
        else "TRANSPORT_FALSIFIED"
        if response_transport_falsified
        else "MIXED_TRANSPORT_EVIDENCE"
    )
    observation_omission_supported = all(
        rows[donor][receiver]["complete_response_object_byte_identical"]
        for donor in ("A", "B")
        for receiver in ("A", "B")
    )
    material = {
        "schema": "nanogpt-response-transport-cross-validation-v1",
        "status": "PASS",
        "mechanical_scientific_outcome": overall_outcome,
        "strict_transport_hypothesis_satisfied": strict_transport_hypothesis_satisfied,
        "response_transport_falsified": response_transport_falsified,
        "observation_omission_supported": observation_omission_supported,
        "strict_context_count": strict_contexts,
        "falsified_context_count": falsified_contexts,
        "total_context_count": total_contexts,
        "rows": rows,
        "inputs": {
            "reciprocal_validation_sha256": read_json(
                reciprocal_root / "reciprocal_pair_validation.json"
            )["validation_sha256"],
            "a_skip_validation_sha256": read_json(
                a_skip_root / "local_response_jk_validation.json"
            )["validation_sha256"],
            "b_skip_validation_sha256": read_json(
                b_skip_root / "local_response_jk_validation.json"
            )["validation_sha256"],
            "a_native_full_validation_sha256": read_json(
                a_native_full_root / "local_response_jk_validation.json"
            )["validation_sha256"],
            "b_native_full_validation_sha256": read_json(
                b_native_full_root / "local_response_jk_validation.json"
            )["validation_sha256"],
            "a_native_full_protocol_sha256": file_sha256(a_native_full_protocol_path),
            "b_native_full_protocol_sha256": file_sha256(b_native_full_protocol_path),
        },
        "check_count": check_count,
        "future_information_used": False,
        "categorical_values_subtracted": False,
        "scientific_interpretation_performed": False,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_root / "response_transport_cross_validation.json", result)
    return result


def validate_response_transport_replay(
    *,
    primary_root: Path,
    replay_root: Path,
) -> dict[str, Any]:
    primary = read_json(primary_root / "response_transport_cross_validation.json")
    replay = read_json(replay_root / "response_transport_cross_validation.json")
    require(primary["status"] == replay["status"] == "PASS", "SST_RESPONSE_TRANSPORT_REPLAY_STATUS_INVALID")
    for field in (
        "mechanical_scientific_outcome",
        "strict_transport_hypothesis_satisfied",
        "response_transport_falsified",
        "observation_omission_supported",
        "strict_context_count",
        "falsified_context_count",
        "total_context_count",
        "rows",
        "check_count",
        "future_information_used",
        "categorical_values_subtracted",
    ):
        require(primary[field] == replay[field], f"SST_RESPONSE_TRANSPORT_REPLAY_FIELD_MISMATCH:{field}")

    def tensor_inventory(root: Path) -> dict[str, str]:
        return {
            path.name: file_sha256(path)
            for path in sorted((root / "tensor-objects").glob("*.npy"))
        }

    primary_tensors = tensor_inventory(primary_root)
    replay_tensors = tensor_inventory(replay_root)
    require(bool(primary_tensors), "SST_RESPONSE_TRANSPORT_REPLAY_TENSORS_EMPTY")
    require(primary_tensors == replay_tensors, "SST_RESPONSE_TRANSPORT_REPLAY_TENSOR_MISMATCH")
    material = {
        "schema": "nanogpt-response-transport-cross-replay-validation-v1",
        "status": "PASS",
        "primary_validation_sha256": primary["validation_sha256"],
        "replay_validation_sha256": replay["validation_sha256"],
        "tensor_file_count": len(primary_tensors),
        "rows_identical": True,
        "tensor_payloads_identical": True,
        "mechanical_scientific_outcome_identical": True,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(replay_root / "response_transport_replay_validation.json", result)
    return result


__all__ = ["validate_response_transport_cross", "validate_response_transport_replay"]
