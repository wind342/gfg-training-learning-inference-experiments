from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ALPHAS = (-0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
GLOBAL_UNSEEN_ENTRY = "entry-362ded584a953f360aec"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_array(payload_root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    _require(locator.startswith("tensor-objects/"), f"INVALID_LOCATOR:{locator}")
    path = payload_root / locator
    _require(path.is_file(), f"MISSING_PAYLOAD:{path}")
    if reference.get("file_sha256"):
        _require(file_sha256(path) == reference["file_sha256"], f"FILE_HASH_MISMATCH:{path}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    raw = sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))
    _require(raw == reference["raw_tensor_sha256"], f"RAW_HASH_MISMATCH:{path}")
    _require(list(value.shape) == list(reference["shape"]), f"SHAPE_MISMATCH:{path}")
    return value


def load_named_array(payload_root: Path, reference: dict[str, Any]) -> dict[str, np.ndarray]:
    packed = load_array(payload_root, reference)
    result: dict[str, np.ndarray] = {}
    for row in reference["layout"]:
        offset = int(row["offset"])
        count = int(row["element_count"])
        child = np.asarray(packed[offset : offset + count], dtype=np.dtype(row["dtype"]))
        child = np.ascontiguousarray(child.reshape(tuple(int(value) for value in row["shape"])))
        _require(sha256_bytes(child.tobytes(order="C")) == row["raw_tensor_sha256"], f"CHILD_HASH_MISMATCH:{row['name']}")
        result[str(row["name"])] = child
    _require(tuple(result) == tuple(reference["canonical_name_order"]), "NAMED_LAYOUT_ORDER_MISMATCH")
    return result


def _raw_entry(root: Path, entry_id: str) -> Path:
    path = root / entry_id
    _require(path.is_dir(), f"MISSING_ENTRY_ARCHIVE:{entry_id}")
    return path


def _state_path(entry_root: Path, window_id: str, optimizer_step: int) -> Path:
    return entry_root / "windows" / window_id / "states" / f"step-{optimizer_step:05d}.json"


def _transition_path(entry_root: Path, window_id: str, optimizer_step: int) -> Path:
    return entry_root / "windows" / window_id / "transitions" / f"step-{optimizer_step:05d}-to-{optimizer_step + 1:05d}.json"


def _probe_path(entry_root: Path, state_id: str) -> Path:
    return entry_root / "probe-observations" / "CSRG-4C-v1" / f"{state_id}.json"


def _baseline_group_membership(entry_root: Path, probe: dict[str, Any]) -> np.ndarray:
    _require(probe["status"] == "PASS", f"PROBE_NOT_PASS:{probe.get('probe_observation_id')}")
    _require(probe["probe_contract_id"] == "CSRG-4C-v1", "PROBE_CONTRACT_MISMATCH")
    rows = probe["forwards"]
    _require(len(rows) == 12, "PROBE_FORWARD_COUNT_MISMATCH")
    _require(rows[0]["gate_components"] == rows[1]["gate_components"] == [], "BASELINE_ORDER_INVALID")
    groups = np.asarray(load_array(entry_root, rows[0]["group_membership"]), dtype=np.int64)
    for row in rows[1:]:
        other = np.asarray(load_array(entry_root, row["group_membership"]), dtype=np.int64)
        _require(np.array_equal(groups, other), "PROBE_GROUP_MEMBERSHIP_CHANGED")
    return groups


def _entry_sources(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = selection["entry_sources"]
    _require(isinstance(value, dict), "ENTRY_SOURCES_NOT_OBJECT")
    return value


def _upstream_element_identity(
    *, object_id: str, tensor_hash: str, row_hash: str
) -> str:
    material = b"UPSTREAM-ELEMENT-v1\0" + object_id.encode() + b"\0" + tensor_hash.encode() + b"\0" + row_hash.encode()
    return sha256_bytes(material)


def _evaluation_unit_id(
    *, task_commitment: str, row_hash: str, upstream_identity: str, target_group: int
) -> str:
    material = (
        b"EVALUNIT-v1\0CSRG-4C-v1\0"
        + task_commitment.encode()
        + b"\0"
        + row_hash.encode()
        + b"\0"
        + upstream_identity.encode()
        + b"\0"
        + str(target_group).encode()
    )
    return "evalunit-" + sha256_bytes(material)


def iter_sections(selection: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for pair in selection["pairs"]:
        for section in pair["sections"]:
            yield pair, section


def build_inventory(
    *,
    selection_path: Path,
    raw_root: Path,
    validation_input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    raw_root = raw_root.resolve()
    validation_input_root = validation_input_root.resolve()
    output_root = output_root.resolve()
    selection = read_json(selection_path)
    _require("FROZEN" in selection["status"], "SELECTION_NOT_FROZEN")
    _require(int(selection["development_entry_count"]) == 12, "DEVELOPMENT_ENTRY_COUNT_INVALID")
    _require(int(selection["adjacent_pair_count"]) == 36, "PAIR_COUNT_INVALID")
    _require(int(selection["response_section_count"]) == 72, "SECTION_COUNT_INVALID")
    entry_sources = _entry_sources(selection)
    _require(len(entry_sources) == 12, "ENTRY_SOURCE_COUNT_INVALID")
    _require(GLOBAL_UNSEEN_ENTRY not in entry_sources, "GLOBAL_UNSEEN_ENTRY_EXPOSED")

    expected_section_hash = sha256_bytes(
        canonical_bytes(sorted(section["section_id"] for _pair, section in iter_sections(selection)))
    )
    _require(expected_section_hash == selection["section_id_set_sha256"], "SECTION_SET_COMMITMENT_MISMATCH")

    resolved_sections: list[dict[str, Any]] = []
    evaluation_units: dict[str, list[dict[str, Any]]] = {}
    verified_files: set[Path] = set()
    exact_update_identity_sections = 0
    exact_readdition_sections = 0
    max_endpoint_parameter_error = 0.0

    for pair, section in iter_sections(selection):
        entry_id = pair["entry_id_audit_only"]
        window_id = pair["window_id"]
        receiver_step = int(section["receiver_optimizer_step_audit_only"])
        endpoint_step = int(section["native_endpoint_optimizer_step_audit_only"])
        _require(endpoint_step == receiver_step + 1, f"SECTION_NOT_ONE_STEP:{section['section_id']}")
        entry_root = _raw_entry(raw_root, entry_id)
        receiver_state_path = _state_path(entry_root, window_id, receiver_step)
        endpoint_state_path = _state_path(entry_root, window_id, endpoint_step)
        transition_path = _transition_path(entry_root, window_id, receiver_step)
        for path in (receiver_state_path, endpoint_state_path, transition_path):
            _require(path.is_file(), f"MISSING_SECTION_RECORD:{path}")
            verified_files.add(path)
        receiver_doc = read_json(receiver_state_path)
        endpoint_doc = read_json(endpoint_state_path)
        transition_doc = read_json(transition_path)
        _require(receiver_doc["optimizer_step"] == receiver_step, "RECEIVER_STEP_MISMATCH")
        _require(endpoint_doc["optimizer_step"] == endpoint_step, "ENDPOINT_STEP_MISMATCH")
        _require(transition_doc["optimizer_step"] == receiver_step, "TRANSITION_STEP_MISMATCH")
        _require(transition_doc["window_id"] == window_id, "TRANSITION_WINDOW_MISMATCH")
        _require(transition_doc["transition_id"] in transition_doc["step"]["parameter_update"]["representation"], "TRANSITION_IDENTITY_MISMATCH")

        receiver = load_named_array(entry_root, receiver_doc["state"]["parameters"])
        update = load_named_array(entry_root, transition_doc["step"]["parameter_update"])
        endpoint = load_named_array(entry_root, endpoint_doc["state"]["parameters"])
        _require(set(receiver) == set(update) == set(endpoint), "PARAMETER_NAME_SET_MISMATCH")
        section_error = 0.0
        section_readdition_exact = True
        section_update_identity_exact = True
        for name in receiver:
            reconstructed = np.add(receiver[name], update[name], dtype=np.float32)
            exact_update = np.subtract(endpoint[name], receiver[name], dtype=np.float32)
            error = float(np.max(np.abs(reconstructed.astype(np.float64) - endpoint[name].astype(np.float64))))
            section_error = max(section_error, error)
            section_readdition_exact = section_readdition_exact and np.array_equal(reconstructed, endpoint[name])
            section_update_identity_exact = section_update_identity_exact and np.array_equal(exact_update, update[name])
        max_endpoint_parameter_error = max(max_endpoint_parameter_error, section_error)
        exact_readdition_sections += int(section_readdition_exact)
        exact_update_identity_sections += int(section_update_identity_exact)
        _require(section_update_identity_exact, f"NATIVE_UPDATE_IDENTITY_NOT_EXACT:{section['section_id']}")

        receiver_probe_path = _probe_path(entry_root, section["receiver_prestate"]["observed_state_id"])
        endpoint_probe_path = _probe_path(entry_root, section["native_endpoint_adjudication_only"]["observed_state_id"])
        _require(receiver_probe_path.is_file() and endpoint_probe_path.is_file(), "MISSING_PROBE_OBSERVATION")
        verified_files.update((receiver_probe_path, endpoint_probe_path))
        receiver_probe = read_json(receiver_probe_path)
        endpoint_probe = read_json(endpoint_probe_path)
        _require(receiver_probe["probe_observation_id"] == section["receiver_prestate"]["probe_observation_id"], "RECEIVER_PROBE_ID_MISMATCH")
        _require(endpoint_probe["probe_observation_id"] == section["native_endpoint_adjudication_only"]["probe_observation_id"], "ENDPOINT_PROBE_ID_MISMATCH")
        receiver_groups = _baseline_group_membership(entry_root, receiver_probe)
        endpoint_groups = _baseline_group_membership(entry_root, endpoint_probe)
        _require(np.array_equal(receiver_groups, endpoint_groups), "CROSS_STEP_TARGET_IDENTITY_CHANGED")

        source = entry_sources[entry_id]["original_training_gfg"]
        validation = source["validation_input"]
        input_path = validation_input_root / f"{entry_id}.npy"
        _require(input_path.is_file(), f"MISSING_VALIDATION_INPUT:{entry_id}")
        inputs = np.load(input_path, allow_pickle=False)
        _require(list(inputs.shape) == [212, 3] and inputs.dtype == np.int64, f"VALIDATION_INPUT_SCHEMA_INVALID:{entry_id}")
        input_hash = sha256_bytes(np.ascontiguousarray(inputs).tobytes(order="C"))
        _require(input_hash == validation["content_sha256"], f"VALIDATION_INPUT_HASH_MISMATCH:{entry_id}")
        row_hashes = [sha256_bytes(np.ascontiguousarray(row).tobytes(order="C")) for row in inputs]
        _require(len(set(row_hashes)) == 212, f"VALIDATION_ROWS_NOT_UNIQUE:{entry_id}")

        units: list[dict[str, Any]] = []
        for element_index, (row_hash, target_group) in enumerate(zip(row_hashes, receiver_groups.tolist())):
            upstream = _upstream_element_identity(
                object_id=validation["object_id_at_step_100"],
                tensor_hash=input_hash,
                row_hash=row_hash,
            )
            evaluation_unit_id = _evaluation_unit_id(
                task_commitment=source["task_commitment"],
                row_hash=row_hash,
                upstream_identity=upstream,
                target_group=int(target_group),
            )
            units.append(
                {
                    "element_index_audit_only": element_index,
                    "evaluation_unit_id": evaluation_unit_id,
                    "row_content_sha256": row_hash,
                    "target_group": int(target_group),
                    "upstream_element_identity": upstream,
                }
            )
        if entry_id in evaluation_units:
            _require(evaluation_units[entry_id] == units, f"EVALUATION_UNIT_ID_DRIFT:{entry_id}")
        else:
            evaluation_units[entry_id] = units

        resolved_sections.append(
            {
                "entry_id_audit_only": entry_id,
                "evaluation_input_path": str(input_path),
                "evaluation_input_sha256": input_hash,
                "native_endpoint_probe_path": str(endpoint_probe_path),
                "native_endpoint_state_path": str(endpoint_state_path),
                "observation_id": section["observation_id"],
                "pair_id": pair["pair_id"],
                "primary_stratum": pair["primary_stratum"],
                "receiver_probe_path": str(receiver_probe_path),
                "receiver_state_path": str(receiver_state_path),
                "section_id": section["section_id"],
                "section_ordinal_within_pair": section["section_ordinal_within_pair"],
                "transition_path": str(transition_path),
                "window_id": window_id,
            }
        )

    _require(len(resolved_sections) == 72, "RESOLVED_SECTION_COUNT_INVALID")
    _require(exact_update_identity_sections == 72, "NATIVE_UPDATE_IDENTITY_GATE_INCOMPLETE")
    all_unit_ids = [row["evaluation_unit_id"] for rows in evaluation_units.values() for row in rows]
    _require(len(all_unit_ids) == 12 * 212, "EVALUATION_UNIT_COUNT_INVALID")
    _require(len(set(all_unit_ids)) == len(all_unit_ids), "EVALUATION_UNIT_ID_COLLISION")

    output_root.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema": "nanogpt-adjacent-response-resolved-inventory-v1",
        "status": "PASS",
        "alpha_grid": list(ALPHAS),
        "development_entry_count": 12,
        "adjacent_pair_count": 36,
        "response_section_count": 72,
        "evaluation_unit_count": len(all_unit_ids),
        "global_unseen_entry_accessed": False,
        "selection_manifest_file_sha256": file_sha256(selection_path),
        "section_id_set_sha256": selection["section_id_set_sha256"],
        "native_update_identity_exact_section_count": exact_update_identity_sections,
        "float32_readdition_exact_section_count": exact_readdition_sections,
        "float32_readdition_max_abs_error": max_endpoint_parameter_error,
        "alpha1_parameter_source": "stored native endpoint state; endpoint-minus-receiver equals the captured update exactly",
        "verified_record_file_count": len(verified_files),
        "sections": resolved_sections,
    }
    identity = {
        "schema": "nanogpt-adjacent-response-identity-material-v1",
        "status": "PASS",
        "evaluation_unit_formula": "SHA256(EVALUNIT-v1 || CSRG-4C-v1 || task_commitment || row_content_sha256 || upstream_element_identity || target_group)",
        "array_index_is_identity": False,
        "entry_count": 12,
        "evaluation_unit_count": len(all_unit_ids),
        "entries": evaluation_units,
    }
    audit = {
        "schema": "nanogpt-adjacent-response-payload-audit-v1",
        "status": "PASS",
        "missing_blocking_payloads": [],
        "regenerated_substitutes_used": False,
        "verified_section_count": 72,
        "verified_record_file_count": len(verified_files),
        "alpha1_parameter_endpoint_gate": {
            "status": "PASS",
            "native_update_identity_exact_sections": exact_update_identity_sections,
            "float32_readdition_exact_sections": exact_readdition_sections,
            "float32_readdition_max_abs_error": max_endpoint_parameter_error,
            "alpha1_uses_stored_native_endpoint": True,
        },
        "global_unseen_entry_accessed": False,
    }
    write_json(output_root / "RESOLVED_INVENTORY.json", inventory)
    write_json(output_root / "IDENTITY_MATERIAL.json", identity)
    write_json(output_root / "PAYLOAD_AVAILABILITY_AUDIT.json", audit)
    return inventory
