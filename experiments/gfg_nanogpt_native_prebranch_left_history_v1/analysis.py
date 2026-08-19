from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import zlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from experiments.gfg_nanogpt_response_factor_analysis_v1.analysis import COMPONENTS
from experiments.gfg_nanogpt_state_conditioned_response_v1.dataset import build_dataset, load_records
from experiments.gfg_nanogpt_state_conditioned_response_v1.model import boundary_class


FACTOR_ROOT = Path(r"E:\gfg-evidence\nanogpt-response-factor-analysis-v1\submission")
RESPONSE_ROOT = Path(r"E:\gfg-evidence\nanogpt-adjacent-response-transport-v1\submission")
CRITICAL_ROOT = Path(r"E:\gfg-evidence\nanogpt-critical-branch-localization-v1\submission")
STEPWISE_ROOT = Path(r"D:\gfg-evidence\nanogpt-stepwise-support-transition-formal-v3")
FACTOR_RECORDS = FACTOR_ROOT / "PRETARGET_FACTOR_RECORDS.jsonl.gz"
CRITICAL_LEDGER = CRITICAL_ROOT / "CRITICAL_BRANCH_LEDGER.jsonl.gz"
IDENTITY_MATERIAL = RESPONSE_ROOT / "IDENTITY_MATERIAL.json"

SCALES = (1, 2, 5, 10)
PRIMARY_K = 64
BOOTSTRAPS = 1000
BASE_SEED = 20260806
TASKS = (
    "competitor_switch",
    "severe_conflict",
    "support_handoff",
    "turnback",
    "sign_reversal",
    "correct_to_wrong",
    "wrong_to_correct",
)
SPACES = ("X0", "X1", "X2", "X3", "X4")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{BASE_SEED}:{label}".encode("utf-8")).digest()[:8], "big")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


class PayloadLoader:
    def __init__(self, stepwise_root: Path) -> None:
        self.stepwise_root = stepwise_root
        self.cache: dict[tuple[str, str], np.ndarray] = {}
        self.verified_files: dict[str, str] = {}

    def load(self, entry_id: str, descriptor: dict[str, Any]) -> np.ndarray:
        locator = str(descriptor["locator"])
        require(".." not in Path(locator).parts, f"UNSAFE_LOCATOR:{locator}")
        key = (entry_id, locator)
        if key in self.cache:
            return self.cache[key]
        path = self.stepwise_root / entry_id / locator
        require(path.is_file(), f"PAYLOAD_MISSING:{path}")
        expected = str(descriptor["file_sha256"])
        actual = file_sha256(path)
        require(actual == expected, f"PAYLOAD_HASH_MISMATCH:{path}")
        value = np.asarray(np.load(path, allow_pickle=False))
        self.cache[key] = value
        self.verified_files[str(path)] = actual
        return value


def graph_objects(entry_id: str, step: int) -> list[dict[str, Any]]:
    database = STEPWISE_ROOT / entry_id / "stepwise_support_transition_gfg.sqlite3"
    require(database.is_file(), f"STEPWISE_GFG_DATABASE_MISSING:{database}")
    values: list[dict[str, Any]] = []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT payload_zlib FROM graph_blocks WHERE optimizer_step = ? ORDER BY block_ordinal",
            (int(step),),
        )
        for row in rows:
            values.extend(json.loads(zlib.decompress(row["payload_zlib"])).get("objects", []))
    require(bool(values), f"STEPWISE_OBJECTS_MISSING:{entry_id}:{step}")
    return values


def select_temporal_object(
    objects: list[dict[str, Any]], role: str, source_steps: list[int]
) -> dict[str, Any]:
    matches = [
        value
        for value in objects
        if str(value.get("role")) == role
        and list(value.get("payload", {}).get("source_optimizer_steps", [])) == source_steps
    ]
    require(len(matches) == 1, f"TEMPORAL_OBJECT_CARDINALITY:{role}:{source_steps}:{len(matches)}")
    payload = matches[0]["payload"]
    require(payload.get("temporal_role") == "input_available_at_cut", f"TEMPORAL_ROLE_INVALID:{role}")
    return matches[0]


def layout_slice(flat: np.ndarray, descriptor: dict[str, Any], names: Iterable[str]) -> np.ndarray:
    wanted = set(names)
    chunks: list[np.ndarray] = []
    for row in descriptor["layout"]:
        if str(row["name"]) in wanted:
            offset = int(row["offset"])
            count = int(row["element_count"])
            chunks.append(np.asarray(flat[offset : offset + count], dtype=np.float64))
    require(bool(chunks), f"LAYOUT_GROUP_EMPTY:{sorted(wanted)}")
    return np.concatenate(chunks)


def component_layout_names(descriptor: dict[str, Any], component: str) -> list[str]:
    layer = "0" if component.startswith("h0") else "1"
    kind = "attn" if component.endswith("attn") else "mlp"
    prefix = f"transformer.h.{layer}.{kind}."
    names = [str(row["name"]) for row in descriptor["layout"] if str(row["name"]).startswith(prefix)]
    require(bool(names), f"COMPONENT_LAYOUT_MISSING:{component}")
    return names


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 1e-20 else None


def update_continuity(
    entry_id: str,
    transition_path: Path,
    step: int,
    loader: PayloadLoader,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    window = transition_path.parent.parent
    transitions: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    descriptors: list[dict[str, Any]] = []
    for lag in range(0, 4):
        path = window / "transitions" / f"step-{step - lag:05d}-to-{step - lag + 1:05d}.json"
        require(path.is_file(), f"PRIOR_TRANSITION_MISSING:{path}")
        material = read_json(path)
        descriptor = material["step"]["parameter_update"]
        vector = loader.load(entry_id, descriptor).astype(np.float64, copy=False)
        transitions.append({"path": str(path), "sha256": file_sha256(path), "descriptor": descriptor})
        vectors.append(vector)
        descriptors.append(descriptor)
    current = vectors[0]
    current_norm = float(np.linalg.norm(current))
    result: dict[str, float | None] = {}
    for lag in (1, 2, 3):
        past = vectors[lag]
        past_norm = float(np.linalg.norm(past))
        result[f"update_global_cos_lag{lag}"] = cosine(current, past)
        result[f"update_global_norm_ratio_lag{lag}"] = current_norm / past_norm if past_norm > 1e-20 else None
        for component in COMPONENTS:
            names = component_layout_names(descriptors[0], component)
            left = layout_slice(current, descriptors[0], names)
            right = layout_slice(past, descriptors[lag], names)
            result[f"update_{component.replace('.', '_')}_cos_lag{lag}"] = cosine(left, right)
    available_cos = [result[f"update_global_cos_lag{lag}"] for lag in (1, 2, 3)]
    result["update_global_cos_mean3"] = (
        float(np.mean([float(value) for value in available_cos if value is not None]))
        if any(value is not None for value in available_cos)
        else None
    )
    return result, transitions


def _top_competitors(logits: np.ndarray, target: int, count: int = 5) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).copy()
    values[target] = -np.inf
    return np.argsort(values)[::-1][:count]


def _leader(logits: np.ndarray, target: int) -> int:
    return int(_top_competitors(logits, target, 1)[0])


def _sampled_switch_count(values: list[int]) -> float:
    return float(sum(left != right for left, right in zip(values, values[1:])))


def _observed_persistence(values: list[int], identity: int) -> float:
    count = 0
    for value in reversed(values):
        if value != identity:
            break
        count += 1
    return float(count / len(values))


def _support_primary(values: np.ndarray) -> tuple[int, int]:
    finite = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(finite)[::-1]
    return int(order[0]), int(order[1])


def _feature_names(rows: list[dict[str, float | None]]) -> list[str]:
    names = sorted(rows[0])
    for row in rows:
        require(sorted(row) == names, "LEFT_FEATURE_SCHEMA_DRIFT")
    return names


def _matrix(rows: list[dict[str, float | None]], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[np.nan if row[name] is None else float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )


def compile_dataset() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    records = load_records(FACTOR_RECORDS)
    dataset = build_dataset(records)
    identities = read_json(IDENTITY_MATERIAL)["entries"]
    critical = {row["record_id"]: row for row in read_jsonl_gz(CRITICAL_LEDGER)}
    require(set(critical) == {row["record_id"] for row in records}, "CRITICAL_LEDGER_IDENTITY_MISMATCH")
    loader = PayloadLoader(STEPWISE_ROOT)
    by_section: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_section[str(record["section_id"])].append(index)

    x1_rows: list[dict[str, float | None]] = [{} for _ in records]
    x2_rows: list[dict[str, float | None]] = [{} for _ in records]
    x3_rows: list[dict[str, float | None]] = [{} for _ in records]
    x4_rows: list[dict[str, float | None]] = [{} for _ in records]
    meta: list[dict[str, Any]] = [{} for _ in records]
    availability = Counter()
    source_sections: list[dict[str, Any]] = []

    for section_id, indices in sorted(by_section.items()):
        first = records[indices[0]]
        entry_id = str(first["entry_id"])
        step = int(first["optimizer_step"])
        section_path = RESPONSE_ROOT / "sections" / f"{section_id}.npz"
        require(section_path.is_file(), f"RESPONSE_SECTION_MISSING:{section_path}")
        section_sha = file_sha256(section_path)
        require(section_sha == first["source_refs"]["section_npz_sha256"], f"RESPONSE_SECTION_HASH_DRIFT:{section_id}")
        with np.load(section_path, allow_pickle=False) as data:
            current_logits = np.asarray(data["all_logits"], dtype=np.float64)[1, 0]
            current_necessity = np.asarray(data["necessity"], dtype=np.float64)[1]
            current_allocation = np.asarray(data["support_allocation"], dtype=np.float64)[1]
            endpoint_allocation = np.asarray(data["support_allocation"], dtype=np.float64)[6]

        objects = graph_objects(entry_id, step)
        object_roles = Counter(str(value.get("role")) for value in objects)
        forbidden_present = any(str(value.get("role", "")).startswith("target_only_after_cut:") for value in objects)
        require(forbidden_present, f"EXPECTED_TARGET_PARTITION_MISSING:{section_id}")
        left_logits: dict[int, np.ndarray] = {}
        left_necessity: dict[int, np.ndarray] = {}
        left_allocation: dict[int, np.ndarray] = {}
        used_objects: list[dict[str, Any]] = []
        for scale in SCALES:
            for field, target in (
                ("forward_logits", left_logits),
                ("necessity", left_necessity),
                ("support_allocation", left_allocation),
            ):
                obj = select_temporal_object(
                    objects,
                    f"input_available_at_cut:finite_difference_left:{field}",
                    [step - scale, step],
                )
                target[scale] = loader.load(entry_id, obj["payload"]).astype(np.float64, copy=False)
                used_objects.append(obj)
        accel_logits_obj = select_temporal_object(
            objects,
            "input_available_at_cut:finite_difference_left_acceleration:forward_logits",
            [step - 2, step - 1, step],
        )
        accel_logits = loader.load(entry_id, accel_logits_obj["payload"]).astype(np.float64, copy=False)
        accel_support_obj = select_temporal_object(
            objects,
            "input_available_at_cut:finite_difference_left_acceleration:necessity",
            [step - 2, step - 1, step],
        )
        accel_support = loader.load(entry_id, accel_support_obj["payload"]).astype(np.float64, copy=False)
        used_objects.extend((accel_logits_obj, accel_support_obj))

        transition_path = Path(str(first["source_refs"]["transition"]))
        action, transition_refs = update_continuity(entry_id, transition_path, step, loader)
        identity_index = {str(row["evaluation_unit_id"]): idx for idx, row in enumerate(identities[entry_id])}
        require(len(identity_index) == 212, f"IDENTITY_MATERIAL_INVALID:{entry_id}")
        historical_logits = {scale: current_logits - scale * left_logits[scale][0] for scale in SCALES}
        historical_necessity = {scale: current_necessity - scale * left_necessity[scale] for scale in SCALES}
        historical_allocation = {scale: current_allocation - scale * left_allocation[scale] for scale in SCALES}

        availability["section_count"] += 1
        availability["full_current_logits"] += int(current_logits.shape == (212, 24))
        availability["full_identity_aligned_left_logits"] += int(all(value.shape == (12, 212, 24) for value in left_logits.values()))
        availability["full_current_support"] += int(current_necessity.shape == (4, 23))
        availability["full_identity_aligned_left_support"] += int(all(value.shape == (4, 23) for value in left_necessity.values()))
        availability["current_and_three_prior_updates"] += int(all(value is not None for value in action.values()))
        availability["target_only_partition_present_but_excluded"] += int(forbidden_present)
        availability["direct_past_F5_probe"] += int(object_roles.get("complete_probe_observation", 0) == 1)

        source_sections.append(
            {
                "section_id": section_id,
                "entry_id": entry_id,
                "optimizer_step": step,
                "response_section_sha256": section_sha,
                "input_object_ids": sorted(str(value["object_id"]) for value in used_objects),
                "input_payload_hashes": sorted(str(value["payload"]["file_sha256"]) for value in used_objects),
                "transition_refs": transition_refs,
                "target_only_objects_present_and_excluded": forbidden_present,
            }
        )

        for index in indices:
            record = records[index]
            row_index = identity_index[str(record["evaluation_unit_id"])]
            target = int(record["target_group"])
            current = current_logits[row_index]
            top = _top_competitors(current, target, 5)
            c1, c2, c3, c4, c5 = [int(value) for value in top]
            ordered_scales = (10, 5, 2, 1)
            past_rows = {scale: historical_logits[scale][row_index] for scale in SCALES}
            leader_history = [_leader(past_rows[scale], target) for scale in ordered_scales] + [c1]
            gap12_history = [float(past_rows[scale][c1] - past_rows[scale][c2]) for scale in ordered_scales] + [float(current[c1] - current[c2])]
            gap13_history = [float(past_rows[scale][c1] - past_rows[scale][c3]) for scale in ordered_scales] + [float(current[c1] - current[c3])]
            margin_history = [
                float(past_rows[scale][target] - np.max(np.delete(past_rows[scale], target)))
                for scale in ordered_scales
            ] + [float(record["features"]["F1"]["numeric"]["margin"])]
            current_gap12 = float(current[c1] - current[c2])
            current_gap13 = float(current[c1] - current[c3])
            current_top_leads = np.asarray([float(current[c1] - current[value]) for value in (c2, c3, c4, c5)])
            x1 = {
                "current_c1_c4_gap": float(current[c1] - current[c4]),
                "current_c1_c5_gap": float(current[c1] - current[c5]),
                "current_top5_lead_mean": float(np.mean(current_top_leads)),
                "current_top5_lead_std": float(np.std(current_top_leads)),
                "current_crowding_0_1": float(np.sum((current[c1] - current) <= 0.1) - 1),
                "current_crowding_0_25": float(np.sum((current[c1] - current) <= 0.25) - 1),
                "sampled_competitor_switch_count": _sampled_switch_count(leader_history),
                "sampled_current_c1_persistence": _observed_persistence(leader_history, c1),
                "sampled_c2_was_leader": float(c2 in leader_history[:-1]),
                "sampled_c3_was_leader": float(c3 in leader_history[:-1]),
                "sampled_unique_leader_count": float(len(set(leader_history))),
            }
            x2: dict[str, float | None] = {}
            scale_to_position = {10: 0, 5: 1, 2: 2, 1: 3}
            for scale in SCALES:
                pos = scale_to_position[scale]
                g12 = gap12_history[pos]
                g13 = gap13_history[pos]
                margin = margin_history[pos]
                x2[f"gap12_lag{scale}"] = g12
                x2[f"gap13_lag{scale}"] = g13
                x2[f"margin_lag{scale}"] = margin
                x2[f"gap12_velocity_m{scale}"] = (current_gap12 - g12) / scale
                x2[f"gap13_velocity_m{scale}"] = (current_gap13 - g13) / scale
                x2[f"margin_velocity_m{scale}"] = (margin_history[-1] - margin) / scale
            x2["gap12_acceleration_m1"] = (current_gap12 - gap12_history[3]) - (gap12_history[3] - gap12_history[2])
            x2["gap13_acceleration_m1"] = (current_gap13 - gap13_history[3]) - (gap13_history[3] - gap13_history[2])
            x2["margin_acceleration_m1"] = (margin_history[-1] - margin_history[3]) - (margin_history[3] - margin_history[2])
            x2["sampled_gap12_shrink_count"] = float(sum(right < left for left, right in zip(gap12_history, gap12_history[1:])))
            x2["sampled_margin_sign_switch_count"] = float(
                sum((left >= 0.0) != (right >= 0.0) for left, right in zip(margin_history, margin_history[1:]))
            )
            # Assert that the independently reconstructed lag-one acceleration agrees with the frozen GFG object.
            expected_accel = float(accel_logits[0, row_index, c1] - accel_logits[0, row_index, c2])
            require(abs(expected_accel - float(x2["gap12_acceleration_m1"])) <= 2e-5, f"LEFT_ACCELERATION_MISMATCH:{record['record_id']}")

            group = target
            current_support = current_necessity[:, group]
            primary, secondary = _support_primary(current_support)
            support_histories = {scale: historical_necessity[scale][:, group] for scale in SCALES}
            support_leaders = [_support_primary(support_histories[scale])[0] for scale in ordered_scales] + [primary]
            support_gap_now = float(current_support[primary] - current_support[secondary]) if np.all(np.isfinite(current_support[[primary, secondary]])) else None
            x4: dict[str, float | None] = {
                "support_necessity_primary_secondary_gap": support_gap_now,
                "support_allocation_primary_secondary_gap": (
                    float(current_allocation[primary, group] - current_allocation[secondary, group])
                    if np.all(np.isfinite(current_allocation[[primary, secondary], group]))
                    else None
                ),
                "sampled_support_switch_count": _sampled_switch_count(support_leaders),
                "sampled_primary_support_persistence": _observed_persistence(support_leaders, primary),
                "sampled_unique_primary_support_count": float(len(set(support_leaders))),
            }
            for scale in SCALES:
                hist = support_histories[scale]
                gap = float(hist[primary] - hist[secondary]) if np.all(np.isfinite(hist[[primary, secondary]])) else None
                x4[f"support_gap_same_id_lag{scale}"] = gap
                x4[f"support_gap_velocity_m{scale}"] = (
                    (float(support_gap_now) - float(gap)) / scale if support_gap_now is not None and gap is not None else None
                )
                allocation_hist = historical_allocation[scale][:, group]
                x4[f"support_allocation_primary_lag{scale}"] = finite_or_none(allocation_hist[primary])
            prior_gap1 = x4["support_gap_same_id_lag1"]
            prior_gap2 = x4["support_gap_same_id_lag2"]
            x4["support_gap_acceleration_m1"] = (
                (float(support_gap_now) - float(prior_gap1)) - (float(prior_gap1) - float(prior_gap2))
                if support_gap_now is not None and prior_gap1 is not None and prior_gap2 is not None
                else None
            )
            expected_support_accel = float(accel_support[primary, group] - accel_support[secondary, group])
            if x4["support_gap_acceleration_m1"] is not None and math.isfinite(expected_support_accel):
                require(abs(expected_support_accel - float(x4["support_gap_acceleration_m1"])) <= 2e-8, f"SUPPORT_ACCELERATION_MISMATCH:{record['record_id']}")

            endpoint_support = endpoint_allocation[:, group]
            support_evaluable = bool(np.all(np.isfinite(current_allocation[:, group])) and np.all(np.isfinite(endpoint_support)))
            support_handoff = (
                bool(int(np.argmax(current_allocation[:, group])) != int(np.argmax(endpoint_support))) if support_evaluable else None
            )
            response = record["response"]
            labels: dict[str, bool | None] = {
                "competitor_switch": bool(response["competitor_switch"]),
                "severe_conflict": bool(critical[record["record_id"]]["severe_conflict"]),
                "support_handoff": support_handoff,
                "turnback": bool(response["response_type_detail"]["turnback"]),
                "sign_reversal": bool(response["response_type_detail"]["sign_reversal"]),
                "correct_to_wrong": str(response["boundary_class"]) == "CORRECT_TO_WRONG",
                "wrong_to_correct": str(response["boundary_class"]) == "WRONG_TO_CORRECT",
            }
            x1_rows[index] = x1
            x2_rows[index] = x2
            x3_rows[index] = dict(action)
            x4_rows[index] = x4
            meta[index] = {
                "record_id": record["record_id"],
                "entry_id": entry_id,
                "optimizer_step": step,
                "section_id": section_id,
                "evaluation_unit_id": record["evaluation_unit_id"],
                "target_identity": target,
                "current_competitor_identity": c1,
                "history_identity_aligned": True,
                "gap": current_gap12,
                "gap_velocity": x2["gap12_velocity_m1"],
                "gap_acceleration": x2["gap12_acceleration_m1"],
                "labels": labels,
                "m4_in_support": bool(critical[record["record_id"]]["m4_in_support"]),
                "response_displacement": [float(value) for value in record["response"]["displacement_curve"][2:]],
                "margin0": float(record["response"]["margin_curve"][1]),
                "truth_boundary": str(record["response"]["boundary_class"]),
                "missing_reasons": [] if support_evaluable else ["POST_RESPONSE_SUPPORT_HANDOFF_UNDEFINED"],
            }

    x1_names = _feature_names(x1_rows)
    x2_names = _feature_names(x2_rows)
    x3_names = _feature_names(x3_rows)
    x4_names = _feature_names(x4_rows)
    x0 = np.asarray(dataset["matrices"]["M4"], dtype=np.float64)
    spaces = {
        "X0": x0,
        "X1": np.concatenate([x0, _matrix(x1_rows, x1_names)], axis=1),
        "X2": np.concatenate([x0, _matrix(x1_rows, x1_names), _matrix(x2_rows, x2_names)], axis=1),
        "X3": np.concatenate([x0, _matrix(x1_rows, x1_names), _matrix(x2_rows, x2_names), _matrix(x3_rows, x3_names)], axis=1),
        "X4": np.concatenate(
            [x0, _matrix(x1_rows, x1_names), _matrix(x2_rows, x2_names), _matrix(x3_rows, x3_names), _matrix(x4_rows, x4_names)],
            axis=1,
        ),
    }
    feature_names = {
        "X0": list(dataset["feature_names"]["M4"]),
        "X1": list(dataset["feature_names"]["M4"]) + [f"left_X1:{name}" for name in x1_names],
        "X2": list(dataset["feature_names"]["M4"]) + [f"left_X1:{name}" for name in x1_names] + [f"left_X2:{name}" for name in x2_names],
        "X3": list(dataset["feature_names"]["M4"]) + [f"left_X1:{name}" for name in x1_names] + [f"left_X2:{name}" for name in x2_names] + [f"left_X3:{name}" for name in x3_names],
        "X4": list(dataset["feature_names"]["M4"]) + [f"left_X1:{name}" for name in x1_names] + [f"left_X2:{name}" for name in x2_names] + [f"left_X3:{name}" for name in x3_names] + [f"left_X4:{name}" for name in x4_names],
    }
    prohibited = ("run_id", "entry_id", "optimizer_step", "section_id", "record_id", "phase", "alpha", "future", "response")
    violations = [name for names in feature_names.values() for name in names if any(token in name.lower() for token in prohibited)]
    require(not violations, f"PROHIBITED_FEATURE_NAMES:{sorted(set(violations))}")
    labels = {
        task: np.asarray([np.nan if row["labels"][task] is None else float(bool(row["labels"][task])) for row in meta], dtype=np.float64)
        for task in TASKS
    }
    source_ledger_hash = sha256_bytes(canonical_json(source_sections).encode("utf-8"))
    post_effective_support_undefined_all = sum(
        row["post_response_diagnostics"].get("effective_support_abs_change") is None
        or not math.isfinite(float(row["post_response_diagnostics"].get("effective_support_abs_change")))
        for row in critical.values()
    )
    post_effective_support_undefined_in_support = sum(
        bool(row["m4_in_support"])
        and (
            row["post_response_diagnostics"].get("effective_support_abs_change") is None
            or not math.isfinite(float(row["post_response_diagnostics"].get("effective_support_abs_change")))
        )
        for row in critical.values()
    )
    audit = {
        "schema": "nanogpt-native-prebranch-availability-v1",
        "status": "PASS",
        "record_count": len(records),
        "section_count": len(by_section),
        "run_count": len(dataset["unique_entries"]),
        "coverage_counts": dict(availability),
        "feature_dimensions": {name: int(matrix.shape[1]) for name, matrix in spaces.items()},
        "response_before_available": {
            "current_correct_logit_and_margin": "AVAILABLE_ALL_RECORDS",
            "current_strongest_competitor_identity": "AVAILABLE_ALL_RECORDS",
            "complete_incorrect_competitor_set": "AVAILABLE_ALL_RECORDS_23_IDENTITIES",
            "identity_aligned_logits_lags_1_2_5_10": "AVAILABLE_ALL_RECORDS_FROM_VALIDATED_LEFT_DIFFERENCES",
            "current_primary_and_secondary_support_identity": "AVAILABLE_ALL_RECORDS",
            "identity_aligned_support_lags_1_2_5_10": "AVAILABLE_ALL_RECORDS_FROM_VALIDATED_LEFT_DIFFERENCES",
            "current_F1_F3_F5": "AVAILABLE_ALL_RECORDS",
            "past_F1": "AVAILABLE_ALL_RECORDS_FROM_VALIDATED_LEFT_DIFFERENCES",
            "past_F3_native_update": "AVAILABLE_ALL_SECTIONS_FOR_LAGS_1_2_3",
            "past_F5": "PARTIAL_DIRECT_PROBE_COVERAGE_NOT_USED_AS_MODEL_INPUT",
            "current_vs_past_native_update_geometry": "AVAILABLE_ALL_SECTIONS",
            "F3_identity_relative_action": "AVAILABLE_FOR_OUTPUT_EMBEDDING_ROWS; NOT_AVAILABLE_FOR_FULL_TARGET_JACOBIAN_OR_SUPPORT_COMPONENT_EFFECT",
        },
        "prohibited_current_response_inputs_used": False,
        "target_only_objects_seen_but_excluded": True,
        "source_section_ledger_sha256": source_ledger_hash,
        "verified_payload_file_count": len(loader.verified_files),
        "support_handoff_unevaluable_count": int(np.sum(~np.isfinite(labels["support_handoff"]))),
        "post_effective_support_change_unevaluable_count_all_records": int(post_effective_support_undefined_all),
        "post_effective_support_change_unevaluable_count_m4_in_support": int(post_effective_support_undefined_in_support),
        "post_effective_support_change_used_as_input_or_label": False,
    }
    compiled = {
        "records": records,
        "meta": meta,
        "entries": np.asarray([row["entry_id"] for row in meta], dtype=object),
        "unique_entries": list(dataset["unique_entries"]),
        "spaces": spaces,
        "feature_names": feature_names,
        "labels": labels,
        "curves": np.asarray([row["response_displacement"] for row in meta], dtype=np.float64),
        "margin0": np.asarray([row["margin0"] for row in meta], dtype=np.float64),
        "m4_in_support": np.asarray([row["m4_in_support"] for row in meta], dtype=bool),
        "approach_raw": np.asarray(
            [[-float(row["gap"]), -float(row["gap_velocity"]), -float(row["gap_acceleration"]), float(x3_rows[index]["update_global_cos_lag1"] or 0.0)] for index, row in enumerate(meta)],
            dtype=np.float64,
        ),
        "baseline_raw": {
            # Higher scores must always mean higher prospective switch risk.
            "gap_only": np.asarray([-float(row["gap"]) for row in meta], dtype=np.float64),
            "past_switch_count_only": np.asarray(
                [float(x1_rows[index]["sampled_competitor_switch_count"]) for index in range(len(meta))],
                dtype=np.float64,
            ),
        },
    }
    return compiled, audit, source_sections


class RobustSpace:
    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.keep: np.ndarray | None = None
        self.missing_columns: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustSpace":
        values = np.asarray(values, dtype=np.float64)
        self.median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        iqr = q75 - q25
        self.keep = np.isfinite(self.median) & np.isfinite(iqr) & (iqr >= 1e-12)
        self.scale = np.where(self.keep, iqr, 1.0)
        self.missing_columns = np.any(~np.isfinite(values), axis=0) & self.keep
        require(bool(np.any(self.keep)), "NO_NONCONSTANT_FEATURES")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        require(self.median is not None and self.scale is not None and self.keep is not None, "ROBUST_SPACE_NOT_FITTED")
        values = np.asarray(values, dtype=np.float64)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median[None, :], values)
        scaled = ((filled - self.median[None, :]) / self.scale[None, :])[:, self.keep]
        if self.missing_columns is not None and np.any(self.missing_columns):
            scaled = np.concatenate([scaled, missing[:, self.missing_columns].astype(np.float64)], axis=1)
        require(np.all(np.isfinite(scaled)), "NONFINITE_STANDARDIZED_FEATURE")
        return scaled


def query_neighbors(train_x: np.ndarray, test_x: np.ndarray, k: int = PRIMARY_K) -> tuple[np.ndarray, np.ndarray]:
    use_k = min(int(k), len(train_x))
    distances, neighbors = cKDTree(train_x).query(test_x, k=use_k, workers=-1)
    if use_k == 1:
        distances = np.asarray(distances)[:, None]
        neighbors = np.asarray(neighbors)[:, None]
    return np.asarray(distances, dtype=np.float64), np.asarray(neighbors, dtype=np.int64)


def weights_from_distance(distances: np.ndarray) -> np.ndarray:
    weights = 1.0 / np.maximum(np.asarray(distances, dtype=np.float64), 1e-9)
    return weights / np.sum(weights, axis=1, keepdims=True)


def weighted_values(values: np.ndarray, neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    selected = np.asarray(values)[neighbors]
    return np.sum(selected * weights[..., None], axis=1) if selected.ndim == 3 else np.sum(selected * weights, axis=1)


def self_excluded_neighbors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(values)
    use_k = min(PRIMARY_K + 1, len(values))
    distances, neighbors = tree.query(values, k=use_k, workers=-1)
    output_neighbors = np.empty((len(values), min(PRIMARY_K, len(values) - 1)), dtype=np.int64)
    output_weights = np.empty_like(output_neighbors, dtype=np.float64)
    for index in range(len(values)):
        selected = [(float(distance), int(neighbor)) for distance, neighbor in zip(np.atleast_1d(distances[index]), np.atleast_1d(neighbors[index])) if int(neighbor) != index]
        selected = selected[:PRIMARY_K]
        d = np.asarray([row[0] for row in selected], dtype=np.float64)
        n = np.asarray([row[1] for row in selected], dtype=np.int64)
        w = 1.0 / np.maximum(d, 1e-9)
        w /= np.sum(w)
        output_neighbors[index] = n
        output_weights[index] = w
    return output_neighbors, output_weights


def risk_from_neighbors(labels: np.ndarray, neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(labels, dtype=np.float64)[neighbors] * weights, axis=1)


def choose_threshold(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.any(labels) or not np.any(~labels):
        return float("inf")
    positive = int(np.sum(labels))
    negative = int(np.sum(~labels))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(~sorted_labels)
    # A threshold admits all tied scores. Evaluate only the final position of each tie group.
    group_end = np.r_[np.where(sorted_scores[:-1] != sorted_scores[1:])[0], len(sorted_scores) - 1]
    fpr = cumulative_fp[group_end] / negative
    recall = cumulative_tp[group_end] / positive
    admitted = np.where(fpr <= max_fpr + 1e-12)[0]
    if not len(admitted):
        return float(np.max(scores)) + 1e-12
    best_recall = float(np.max(recall[admitted]))
    if best_recall <= 0.0:
        return float(np.max(scores)) + 1e-12
    recall_candidates = admitted[np.isclose(recall[admitted], best_recall, rtol=0.0, atol=1e-12)]
    best = int(recall_candidates[np.argmin(fpr[recall_candidates])])
    return float(sorted_scores[group_end[best]])


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels))
    negative = int(np.sum(~labels))
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float((np.sum(ranks[labels]) - positive * (positive + 1) / 2.0) / (positive * negative))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels))
    if positive == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision[y]) / positive)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    threshold_values = np.full(len(labels), float(thresholds)) if np.isscalar(thresholds) else np.asarray(thresholds, dtype=np.float64)
    prediction = scores >= threshold_values
    tp = int(np.sum(prediction & labels))
    fp = int(np.sum(prediction & (~labels)))
    tn = int(np.sum((~prediction) & (~labels)))
    fn = int(np.sum((~prediction) & labels))
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    calibration = []
    edges = np.linspace(0.0, 1.0, 11)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & ((scores < right) if right < 1.0 else (scores <= right))
        calibration.append(
            {
                "lower": float(left),
                "upper": float(right),
                "count": int(np.sum(mask)),
                "mean_risk": float(np.mean(scores[mask])) if np.any(mask) else None,
                "observed_rate": float(np.mean(labels[mask])) if np.any(mask) else None,
            }
        )
    q90 = float(np.quantile(scores, 0.90))
    high = scores >= q90
    high_rate = float(np.mean(labels[high])) if np.any(high) else 0.0
    rest_rate = float(np.mean(labels[~high])) if np.any(~high) else 0.0
    return {
        "count": len(labels),
        "positive_count": int(np.sum(labels)),
        "prevalence": float(np.mean(labels)),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "brier": float(np.mean(np.square(scores - labels.astype(np.float64)))),
        "threshold_recall": float(recall),
        "threshold_fpr": float(fpr),
        "threshold_precision": float(precision),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "top_decile_risk_ratio_vs_rest": (high_rate / rest_rate if rest_rate > 0.0 else None),
        "calibration": calibration,
    }


def percentile_score(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for column in range(train.shape[1]):
        sorted_values = np.sort(train[:, column])
        result[:, column] = np.searchsorted(sorted_values, values[:, column], side="right") / len(sorted_values)
    return np.mean(result, axis=1)


def branch_neighbor_curves(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_curves: np.ndarray,
    train_branch: np.ndarray,
    requested_branch: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    predictions = np.empty((len(test_x), train_curves.shape[1]), dtype=np.float64)
    sources: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(len(test_x))]
    for branch in (False, True):
        test_indices = np.where(requested_branch == branch)[0]
        if not len(test_indices):
            continue
        pool = np.where(train_branch == branch)[0]
        require(len(pool) > 0, f"ORACLE_BRANCH_POOL_EMPTY:{branch}")
        distances, local = query_neighbors(train_x[pool], test_x[test_indices], PRIMARY_K)
        weights = weights_from_distance(distances)
        global_neighbors = pool[local]
        predictions[test_indices] = weighted_values(train_curves, global_neighbors, weights)
        for row, original in enumerate(test_indices):
            sources[int(original)] = global_neighbors[row]
    return predictions, sources


def response_metrics(curves: np.ndarray, predicted: np.ndarray, margin0: np.ndarray, severe: np.ndarray) -> dict[str, Any]:
    truth_endpoint = curves[:, -1]
    pred_endpoint = predicted[:, -1]
    truth_boundary = np.asarray([boundary_class(float(start), np.asarray([delta])) for start, delta in zip(margin0, truth_endpoint)], dtype=object)
    pred_boundary = np.asarray([boundary_class(float(start), np.asarray([delta])) for start, delta in zip(margin0, pred_endpoint)], dtype=object)

    def subset(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"count": 0}
        truth = curves[mask]
        estimate = predicted[mask]
        tb = truth_boundary[mask]
        pb = pred_boundary[mask]
        unchanged = np.isin(tb, ["MAINTAIN_CORRECT", "MAINTAIN_WRONG"])
        wrong_to_correct = tb == "WRONG_TO_CORRECT"
        return {
            "count": int(np.sum(mask)),
            "curve_rmse": float(np.sqrt(np.mean(np.square(estimate - truth)))),
            "endpoint_direction_accuracy": float(np.mean(np.sign(estimate[:, -1]) == np.sign(truth[:, -1]))),
            "boundary_accuracy": float(np.mean(pb == tb)),
            "unchanged_false_crossing_rate": float(np.mean(~np.isin(pb[unchanged], ["MAINTAIN_CORRECT", "MAINTAIN_WRONG"]))) if np.any(unchanged) else None,
            "wrong_to_correct_recall": float(np.mean(pb[wrong_to_correct] == "WRONG_TO_CORRECT")) if np.any(wrong_to_correct) else None,
        }

    return {"overall": subset(np.ones(len(curves), dtype=bool)), "severe_conflict": subset(severe)}


def cluster_bootstrap_delta(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    entries: np.ndarray,
    unique_entries: list[str],
    metric: str,
    label: str,
) -> dict[str, Any]:
    generator = np.random.default_rng(stable_seed(f"bootstrap:{label}:{metric}"))
    by_run = {entry: np.where(entries == entry)[0] for entry in unique_entries}
    draws: list[float] = []
    for _ in range(BOOTSTRAPS):
        selected = generator.choice(unique_entries, size=len(unique_entries), replace=True)
        indices = np.concatenate([by_run[str(entry)] for entry in selected])
        y = labels[indices]
        if not np.any(y) or not np.any(~y):
            continue
        fn = pr_auc if metric == "pr_auc" else roc_auc
        value_left = fn(y, left[indices])
        value_right = fn(y, right[indices])
        if value_left is not None and value_right is not None:
            draws.append(float(value_right - value_left))
    require(bool(draws), f"BOOTSTRAP_EMPTY:{label}:{metric}")
    array = np.asarray(draws, dtype=np.float64)
    return {
        "replicates": len(draws),
        "delta": float((pr_auc if metric == "pr_auc" else roc_auc)(labels, right) - (pr_auc if metric == "pr_auc" else roc_auc)(labels, left)),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
    }


def evaluate(compiled: dict[str, Any]) -> dict[str, Any]:
    entries = compiled["entries"]
    unique_entries = compiled["unique_entries"]
    n = len(entries)
    risks = {space: {task: np.full(n, np.nan, dtype=np.float64) for task in TASKS} for space in SPACES}
    thresholds = {space: {task: np.full(n, np.nan, dtype=np.float64) for task in TASKS} for space in SPACES}
    neighbors_x4: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n)]
    ordinary_curves = np.full_like(compiled["curves"], np.nan)
    oracle_curves = np.full_like(compiled["curves"], np.nan)
    routed_curves = np.full_like(compiled["curves"], np.nan)
    approach_scores = np.full(n, np.nan, dtype=np.float64)
    approach_thresholds = np.full(n, np.nan, dtype=np.float64)
    baseline_names = ("gap_only", "past_switch_count_only", "prevalence_only")
    baseline_scores = {name: np.full(n, np.nan, dtype=np.float64) for name in baseline_names}
    baseline_thresholds = {name: np.full(n, np.nan, dtype=np.float64) for name in baseline_names}
    fold_details: list[dict[str, Any]] = []

    for test_entry in unique_entries:
        test = entries == test_entry
        train = ~test
        fold: dict[str, Any] = {"test_entry": test_entry, "train_record_count": int(np.sum(train)), "test_record_count": int(np.sum(test)), "spaces": {}}
        for space in SPACES:
            scaler = RobustSpace().fit(compiled["spaces"][space][train])
            train_x = scaler.transform(compiled["spaces"][space][train])
            test_x = scaler.transform(compiled["spaces"][space][test])
            distances, local_neighbors = query_neighbors(train_x, test_x, PRIMARY_K)
            weights = weights_from_distance(distances)
            train_global = np.where(train)[0]
            global_neighbors = train_global[local_neighbors]
            calibration_neighbors, calibration_weights = self_excluded_neighbors(train_x)
            if space == "X4":
                for row, global_index in enumerate(np.where(test)[0]):
                    neighbors_x4[int(global_index)] = global_neighbors[row]
                ordinary_curves[test] = weighted_values(compiled["curves"], global_neighbors, weights)
            fold_space: dict[str, Any] = {"effective_dimension": int(train_x.shape[1]), "tasks": {}}
            for task in TASKS:
                label_values = compiled["labels"][task]
                train_valid_global = np.where(train & np.isfinite(label_values))[0]
                test_valid_global = np.where(test & np.isfinite(label_values))[0]
                if len(train_valid_global) == int(np.sum(train)):
                    train_labels = label_values[train].astype(bool)
                    test_risk = weighted_values(train_labels.astype(np.float64), local_neighbors, weights)
                    train_calibration = risk_from_neighbors(train_labels.astype(np.float64), calibration_neighbors, calibration_weights)
                    threshold = choose_threshold(train_labels, train_calibration)
                    risks[space][task][test] = test_risk
                    thresholds[space][task][test] = threshold
                    fold_space["tasks"][task] = {"threshold": finite_or_none(threshold), "train_positive": int(np.sum(train_labels))}
                else:
                    # Support-handoff has NA labels. Fit/query a dedicated tree using only evaluable training labels.
                    valid_scaler = RobustSpace().fit(compiled["spaces"][space][train_valid_global])
                    valid_train_x = valid_scaler.transform(compiled["spaces"][space][train_valid_global])
                    valid_test_x = valid_scaler.transform(compiled["spaces"][space][test_valid_global])
                    d, local = query_neighbors(valid_train_x, valid_test_x, PRIMARY_K)
                    w = weights_from_distance(d)
                    train_labels = label_values[train_valid_global].astype(bool)
                    test_risk = weighted_values(train_labels.astype(np.float64), local, w)
                    valid_calibration_neighbors, valid_calibration_weights = self_excluded_neighbors(valid_train_x)
                    train_calibration = risk_from_neighbors(train_labels.astype(np.float64), valid_calibration_neighbors, valid_calibration_weights)
                    threshold = choose_threshold(train_labels, train_calibration)
                    risks[space][task][test_valid_global] = test_risk
                    thresholds[space][task][test_valid_global] = threshold
                    fold_space["tasks"][task] = {"threshold": finite_or_none(threshold), "train_positive": int(np.sum(train_labels)), "unevaluable_test": int(np.sum(test) - len(test_valid_global))}
            fold["spaces"][space] = fold_space

        # The approach score has no learned coefficients; only train empirical ranks and a train-only threshold.
        train_approach = percentile_score(compiled["approach_raw"][train], compiled["approach_raw"][train])
        test_approach = percentile_score(compiled["approach_raw"][train], compiled["approach_raw"][test])
        approach_scores[test] = test_approach
        approach_threshold = choose_threshold(compiled["labels"]["competitor_switch"][train].astype(bool), train_approach)
        approach_thresholds[test] = approach_threshold
        fold["approach_threshold"] = finite_or_none(approach_threshold)

        # Frozen scalar baselines. Empirical ranks and thresholds are fit only on
        # the training runs of the current outer fold.
        main_train_labels = compiled["labels"]["competitor_switch"][train].astype(bool)
        fold["baselines"] = {}
        for baseline_name in ("gap_only", "past_switch_count_only"):
            raw = np.asarray(compiled["baseline_raw"][baseline_name], dtype=np.float64)
            train_score = percentile_score(raw[train, None], raw[train, None])
            test_score = percentile_score(raw[train, None], raw[test, None])
            threshold = choose_threshold(main_train_labels, train_score)
            baseline_scores[baseline_name][test] = test_score
            baseline_thresholds[baseline_name][test] = threshold
            fold["baselines"][baseline_name] = {"threshold": finite_or_none(threshold)}
        prevalence = float(np.mean(main_train_labels))
        train_prevalence_score = np.full(int(np.sum(train)), prevalence, dtype=np.float64)
        prevalence_threshold = choose_threshold(main_train_labels, train_prevalence_score)
        baseline_scores["prevalence_only"][test] = prevalence
        baseline_thresholds["prevalence_only"][test] = prevalence_threshold
        fold["baselines"]["prevalence_only"] = {
            "training_prevalence": prevalence,
            "threshold": finite_or_none(prevalence_threshold),
        }

        # Branch-conditioned response prediction uses the frozen X4 distance.
        scaler = RobustSpace().fit(compiled["spaces"]["X4"][train])
        train_x = scaler.transform(compiled["spaces"]["X4"][train])
        test_x = scaler.transform(compiled["spaces"]["X4"][test])
        train_branch = compiled["labels"]["competitor_switch"][train].astype(bool)
        true_branch = compiled["labels"]["competitor_switch"][test].astype(bool)
        predicted_branch = risks["X4"]["competitor_switch"][test] >= thresholds["X4"]["competitor_switch"][test]
        oracle, _ = branch_neighbor_curves(train_x, test_x, compiled["curves"][train], train_branch, true_branch)
        routed, _ = branch_neighbor_curves(train_x, test_x, compiled["curves"][train], train_branch, predicted_branch)
        oracle_curves[test] = oracle
        routed_curves[test] = routed
        fold_details.append(fold)

    require(all(len(value) == PRIMARY_K for value in neighbors_x4), "X4_NEIGHBOR_CARDINALITY_INVALID")
    require(np.all(np.isfinite(ordinary_curves)) and np.all(np.isfinite(oracle_curves)) and np.all(np.isfinite(routed_curves)), "CURVE_PREDICTION_NONFINITE")

    task_results: dict[str, Any] = {}
    runwise: dict[str, Any] = {}
    for task in TASKS:
        valid = np.isfinite(compiled["labels"][task])
        labels = compiled["labels"][task][valid].astype(bool)
        task_results[task] = {}
        runwise[task] = {}
        for space in SPACES:
            task_results[task][space] = binary_metrics(labels, risks[space][task][valid], thresholds[space][task][valid])
        if task == "competitor_switch":
            task_results[task]["approach_score"] = binary_metrics(labels, approach_scores[valid], approach_thresholds[valid])
            for baseline_name in baseline_names:
                task_results[task][baseline_name] = binary_metrics(
                    labels,
                    baseline_scores[baseline_name][valid],
                    baseline_thresholds[baseline_name][valid],
                )
        for entry in unique_entries:
            mask = valid & (entries == entry)
            runwise[task][entry] = {
                space: binary_metrics(compiled["labels"][task][mask].astype(bool), risks[space][task][mask], thresholds[space][task][mask])
                for space in SPACES
            }
            if task == "competitor_switch":
                runwise[task][entry].update(
                    {
                        baseline_name: binary_metrics(
                            compiled["labels"][task][mask].astype(bool),
                            baseline_scores[baseline_name][mask],
                            baseline_thresholds[baseline_name][mask],
                        )
                        for baseline_name in baseline_names
                    }
                )

    main_labels = compiled["labels"]["competitor_switch"].astype(bool)
    severe_labels = compiled["labels"]["severe_conflict"].astype(bool)
    bootstrap = {
        "competitor_switch_pr_auc_X4_minus_X0": cluster_bootstrap_delta(main_labels, risks["X0"]["competitor_switch"], risks["X4"]["competitor_switch"], entries, unique_entries, "pr_auc", "competitor_switch"),
        "competitor_switch_roc_auc_X4_minus_X0": cluster_bootstrap_delta(main_labels, risks["X0"]["competitor_switch"], risks["X4"]["competitor_switch"], entries, unique_entries, "roc_auc", "competitor_switch"),
        "severe_conflict_pr_auc_X4_minus_X0": cluster_bootstrap_delta(severe_labels, risks["X0"]["severe_conflict"], risks["X4"]["severe_conflict"], entries, unique_entries, "pr_auc", "severe_conflict"),
    }
    response = {
        "ordinary_X4": response_metrics(compiled["curves"], ordinary_curves, compiled["margin0"], severe_labels),
        "oracle_same_competitor_switch_branch_X4": response_metrics(compiled["curves"], oracle_curves, compiled["margin0"], severe_labels),
        "executable_routed_X4": response_metrics(compiled["curves"], routed_curves, compiled["margin0"], severe_labels),
    }
    distribution_subgroups: dict[str, Any] = {}
    for subgroup_name, subgroup_mask in (
        ("m4_in_support", compiled["m4_in_support"]),
        ("m4_out_of_support", ~compiled["m4_in_support"]),
    ):
        distribution_subgroups[subgroup_name] = {
            "record_count": int(np.sum(subgroup_mask)),
            "competitor_switch": {
                **{
                    space: binary_metrics(
                        main_labels[subgroup_mask],
                        risks[space]["competitor_switch"][subgroup_mask],
                        thresholds[space]["competitor_switch"][subgroup_mask],
                    )
                    for space in SPACES
                },
                **{
                    baseline_name: binary_metrics(
                        main_labels[subgroup_mask],
                        baseline_scores[baseline_name][subgroup_mask],
                        baseline_thresholds[baseline_name][subgroup_mask],
                    )
                    for baseline_name in baseline_names
                },
            },
        }
    run_nonnegative = 0
    run_deltas: dict[str, float | None] = {}
    for entry in unique_entries:
        left = runwise["competitor_switch"][entry]["X0"]["pr_auc"]
        right = runwise["competitor_switch"][entry]["X4"]["pr_auc"]
        delta = None if left is None or right is None else float(right - left)
        run_deltas[entry] = delta
        run_nonnegative += int(delta is not None and delta >= 0.0)
    return {
        "schema": "nanogpt-native-prebranch-left-history-results-v1",
        "status": "PASS",
        "record_count": n,
        "run_count": len(unique_entries),
        "task_results": task_results,
        "runwise_results": runwise,
        "fold_details": fold_details,
        "bootstrap": bootstrap,
        "response_prediction": response,
        "history_sufficiency": {
            "sufficient_left_history": {"record_count": n},
            "insufficient_left_history": {"record_count": 0, "metrics": None},
        },
        "distribution_subgroups": distribution_subgroups,
        "runwise_competitor_switch_pr_auc_X4_minus_X0": run_deltas,
        "runs_with_nonnegative_competitor_switch_pr_auc_delta": run_nonnegative,
        "risks": risks,
        "thresholds": thresholds,
        "approach_scores": approach_scores,
        "approach_thresholds": approach_thresholds,
        "baseline_scores": baseline_scores,
        "baseline_thresholds": baseline_thresholds,
        "neighbors_x4": neighbors_x4,
        "ordinary_curves": ordinary_curves,
        "oracle_curves": oracle_curves,
        "routed_curves": routed_curves,
    }


def adjudicate(results: dict[str, Any], leakage_pass: bool = True) -> dict[str, Any]:
    pr_delta = results["bootstrap"]["competitor_switch_pr_auc_X4_minus_X0"]
    severe_delta = results["bootstrap"]["severe_conflict_pr_auc_X4_minus_X0"]
    ordinary = results["response_prediction"]["ordinary_X4"]
    routed = results["response_prediction"]["executable_routed_X4"]
    oracle = results["response_prediction"]["oracle_same_competitor_switch_branch_X4"]
    ordinary_severe = ordinary["severe_conflict"]["curve_rmse"]
    routed_severe = routed["severe_conflict"]["curve_rmse"]
    oracle_severe = oracle["severe_conflict"]["curve_rmse"]
    routed_improvement = (ordinary_severe - routed_severe) / ordinary_severe if ordinary_severe else 0.0
    oracle_improvement = (ordinary_severe - oracle_severe) / ordinary_severe if ordinary_severe else 0.0
    boundary_drop = ordinary["overall"]["boundary_accuracy"] - routed["overall"]["boundary_accuracy"]
    risk_supported = (
        pr_delta["ci95"][0] > 0.0
        and int(results["runs_with_nonnegative_competitor_switch_pr_auc_delta"]) >= 8
    )
    downstream_supported = severe_delta["ci95"][0] > 0.0 or (routed_improvement >= 0.03 and boundary_drop <= 0.01)
    if leakage_pass and risk_supported and downstream_supported:
        verdict = "LEFT_HISTORY_BRANCH_SIGNAL_SUPPORTED"
    elif leakage_pass and pr_delta["delta"] > 0.0 and int(results["runs_with_nonnegative_competitor_switch_pr_auc_delta"]) >= 7:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    diagnosis = (
        "BRANCH_REAL_BUT_PRESTATE_NONIDENTIFIABLE_UNDER_CURRENT_OBSERVATION"
        if verdict != "LEFT_HISTORY_BRANCH_SIGNAL_SUPPORTED" and oracle_improvement >= 0.03
        else None
    )
    return {
        "schema": "nanogpt-native-prebranch-left-history-decision-v1",
        "status": "PASS",
        "verdict": verdict,
        "additional_diagnosis": diagnosis,
        "criteria": {
            "competitor_switch_pr_auc_delta": pr_delta,
            "severe_conflict_pr_auc_delta": severe_delta,
            "runs_with_nonnegative_main_delta": int(results["runs_with_nonnegative_competitor_switch_pr_auc_delta"]),
            "routed_severe_curve_rmse_relative_improvement": float(routed_improvement),
            "routed_overall_boundary_accuracy_drop": float(boundary_drop),
            "oracle_severe_curve_rmse_relative_improvement": float(oracle_improvement),
            "leakage_checks_pass": bool(leakage_pass),
        },
        "interpretation_boundary": "The verdict is limited to the frozen 12-run nanoGPT response corpus and the available pre-response observation contract.",
    }


def public_results(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key
        not in {
            "risks",
            "thresholds",
            "approach_scores",
            "approach_thresholds",
            "baseline_scores",
            "baseline_thresholds",
            "neighbors_x4",
            "ordinary_curves",
            "oracle_curves",
            "routed_curves",
            "runwise_results",
            "fold_details",
        }
    }


__all__ = [
    "BOOTSTRAPS",
    "CRITICAL_LEDGER",
    "FACTOR_RECORDS",
    "IDENTITY_MATERIAL",
    "RESPONSE_ROOT",
    "SPACES",
    "STEPWISE_ROOT",
    "TASKS",
    "adjudicate",
    "binary_metrics",
    "canonical_json",
    "compile_dataset",
    "evaluate",
    "file_sha256",
    "public_results",
    "read_json",
    "read_jsonl_gz",
    "response_metrics",
    "roc_auc",
    "pr_auc",
]
