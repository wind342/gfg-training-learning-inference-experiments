from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
from typing import Any, Iterable
import zlib

import numpy as np


GLOBAL_UNSEEN_ENTRY = "entry-362ded584a953f360aec"
ALPHAS = np.asarray([-0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
POSITIVE_CURVE_INDICES = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)
PRIMARY_K = 1
PRIMARY_CALIPER_QUANTILE = 0.90
BOOTSTRAP_REPLICATES = 2000
RANDOM_PERMUTATIONS = 200
BASE_SEED = 20260806
COMPONENTS = ("h0.attn", "h0.mlp", "h1.attn", "h1.mlp")


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite(value: float | np.floating[Any]) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size < 2:
        return None
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return _finite(np.corrcoef(left, right)[0, 1])


def _first_crossing(alphas: np.ndarray, margins: np.ndarray) -> float | None:
    initial = float(margins[1]) >= 0.0
    for alpha, margin in zip(alphas[2:], margins[2:], strict=True):
        if (float(margin) >= 0.0) != initial:
            return float(alpha)
    return None


def _boundary_class(margins: np.ndarray) -> str:
    start = float(margins[1]) >= 0.0
    end = float(margins[-1]) >= 0.0
    if start and end:
        return "MAINTAIN_CORRECT"
    if start and not end:
        return "CORRECT_TO_WRONG"
    if not start and end:
        return "WRONG_TO_CORRECT"
    return "MAINTAIN_WRONG"


def _response_type(margins: np.ndarray) -> tuple[str, dict[str, float | bool]]:
    margins = np.asarray(margins, dtype=np.float64)
    delta = margins - margins[1]
    scale = float(np.max(np.abs(delta))) + 1e-12
    positive_alpha = ALPHAS[POSITIVE_CURVE_INDICES]
    positive_delta = delta[POSITIVE_CURVE_INDICES]
    endpoint = float(delta[-1])
    linear = positive_alpha * endpoint
    residual_rms = float(np.sqrt(np.mean(np.square((positive_delta - linear) / scale))))
    slopes = np.diff(positive_delta) / np.diff(positive_alpha)
    active = np.abs(slopes) > 0.05 * scale
    active_signs = np.sign(slopes[active])
    turnback = bool(active_signs.size >= 2 and np.any(active_signs[1:] != active_signs[:-1]))
    initial_sign = float(np.sign(positive_delta[0]))
    endpoint_sign = float(np.sign(endpoint))
    sign_reversal = bool(initial_sign != 0 and endpoint_sign != 0 and initial_sign != endpoint_sign)
    first = float(np.mean(np.abs(slopes[:2]))) if slopes.size >= 2 else float(np.mean(np.abs(slopes)))
    last = float(np.mean(np.abs(slopes[-2:]))) if slopes.size >= 2 else first
    median_slope = float(np.median(np.abs(slopes))) + 1e-12
    max_jump = float(np.max(np.abs(np.diff(slopes)))) if slopes.size >= 2 else 0.0
    if turnback:
        label = "TURNBACK"
    elif sign_reversal:
        label = "SIGN_REVERSAL"
    elif residual_rms <= 0.10:
        label = "NEAR_LINEAR"
    elif last <= 0.50 * (first + 1e-12):
        label = "SATURATING"
    elif last >= 2.0 * (first + 1e-12):
        label = "ACCELERATING"
    elif max_jump >= 2.0 * median_slope:
        label = "PIECEWISE"
    else:
        label = "OTHER"
    return label, {
        "nonlinear_residual_rms": residual_rms,
        "turnback": turnback,
        "sign_reversal": sign_reversal,
        "early_abs_slope": first,
        "late_abs_slope": last,
        "max_slope_jump": max_jump,
    }


def _load_locator(entry_root: Path, descriptor: dict[str, Any]) -> np.ndarray:
    locator = str(descriptor["locator"])
    require(".." not in Path(locator).parts, f"UNSAFE_LOCATOR:{locator}")
    path = entry_root / locator
    require(path.is_file(), f"PAYLOAD_MISSING:{path}")
    require(file_sha256(path) == descriptor["file_sha256"], f"PAYLOAD_HASH_MISMATCH:{path}")
    value = np.load(path, allow_pickle=False)
    return np.asarray(value)


def _layout_slice(flat: np.ndarray, descriptor: dict[str, Any], name: str) -> np.ndarray:
    for row in descriptor["layout"]:
        if row["name"] == name:
            offset = int(row["offset"])
            count = int(row["element_count"])
            return np.asarray(flat[offset : offset + count]).reshape(tuple(row["shape"]))
    raise RuntimeError(f"LAYOUT_MEMBER_MISSING:{name}")


def _component_update_features(component_norms: dict[str, float]) -> dict[str, float]:
    vector = np.asarray([float(component_norms[name]) for name in COMPONENTS], dtype=np.float64)
    total = float(np.sum(vector)) + 1e-12
    shares = vector / total
    result: dict[str, float] = {
        "component_update_sum": float(np.sum(vector)),
        "component_update_rms": float(np.sqrt(np.mean(np.square(vector)))),
        "component_update_concentration": float(np.sum(np.square(shares))),
        "component_update_max_share": float(np.max(shares)),
    }
    for name, norm, share in zip(COMPONENTS, vector, shares, strict=True):
        key = name.replace(".", "_")
        result[f"{key}_update_l2"] = float(norm)
        result[f"{key}_update_share"] = float(share)
    return result


class ProbeHistory:
    def __init__(self, stepwise_root: Path) -> None:
        self.stepwise_root = stepwise_root
        self._probe_cache: dict[tuple[str, str, int], dict[str, Any] | None] = {}
        self._transition_cache: dict[Path, dict[str, Any]] = {}

    def _objects_for_step(self, entry_id: str, step: int) -> list[dict[str, Any]]:
        database = self.stepwise_root / entry_id / "stepwise_support_transition_gfg.sqlite3"
        require(database.is_file(), f"STEPWISE_GFG_DATABASE_MISSING:{database}")
        objects: list[dict[str, Any]] = []
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT payload_zlib FROM graph_blocks WHERE optimizer_step = ? ORDER BY block_ordinal",
                (int(step),),
            )
            for row in rows:
                block = json.loads(zlib.decompress(row["payload_zlib"]))
                objects.extend(block.get("objects", []))
        return objects

    def probe(self, entry_id: str, window_path: Path, step: int) -> dict[str, Any] | None:
        key = (entry_id, str(window_path), int(step))
        if key in self._probe_cache:
            return self._probe_cache[key]
        objects = self._objects_for_step(entry_id, step)
        if not objects:
            self._probe_cache[key] = None
            return None
        by_role: dict[str, list[dict[str, Any]]] = {}
        for value in objects:
            by_role.setdefault(str(value.get("role")), []).append(value)
        summaries = by_role.get("complete_probe_observation", [])
        baseline = {
            str(value.get("role")): value
            for value in objects
            if ":forward:0:" in str(value.get("semantic_key", ""))
            and str(value.get("role"))
            in {"complete_decision_logits", "complete_per_example_margins", "complete_predictions"}
        }
        support = {
            str(value.get("role")): value
            for value in objects
            if str(value.get("role")) in {"support_metric_effective_support", "support_metric_support_concentration"}
        }
        required_roles = {"complete_decision_logits", "complete_per_example_margins", "complete_predictions"}
        if len(summaries) != 1 or set(baseline) != required_roles or len(support) != 2:
            self._probe_cache[key] = None
            return None
        summary = summaries[0]
        value = summary["payload"]
        entry_root = self.stepwise_root / entry_id
        source_objects = [summary, *baseline.values(), *support.values()]
        result = {
            "path": str(self.stepwise_root / entry_id / "stepwise_support_transition_gfg.sqlite3"),
            "path_sha256": sha256_bytes(canonical_json(sorted(str(row["object_id"]) for row in source_objects)).encode("utf-8")),
            "source_object_ids": sorted(str(row["object_id"]) for row in source_objects),
            "capability": float(value["capability_accuracy"]),
            "margins": _load_locator(entry_root, baseline["complete_per_example_margins"]["payload"]).astype(np.float64),
            "predictions": _load_locator(entry_root, baseline["complete_predictions"]["payload"]).astype(np.int64),
            "logits": _load_locator(entry_root, baseline["complete_decision_logits"]["payload"]).astype(np.float64),
            "effective_support": _load_locator(entry_root, support["support_metric_effective_support"]["payload"]).astype(np.float64),
            "support_concentration": _load_locator(entry_root, support["support_metric_support_concentration"]["payload"]).astype(np.float64),
            "component_loads": value["component_loads"],
        }
        self._probe_cache[key] = result
        return result

    def update_norm(self, entry_id: str, window_path: Path, step: int) -> float | None:
        path = window_path / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json"
        if not path.is_file():
            return None
        value = self._transition_cache.setdefault(path, read_json(path))
        entry_root = self.stepwise_root / entry_id
        update = _load_locator(entry_root, value["step"]["parameter_update"]).astype(np.float64)
        return float(np.linalg.norm(update))


def _flatten_sections(selection: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair in selection["pairs"]:
        for section in pair["sections"]:
            value = dict(section)
            value.update(
                {
                    "entry_id": pair["entry_id_audit_only"],
                    "pair_id": pair["pair_id"],
                    "window_id": pair["window_id"],
                    "primary_stratum": pair["primary_stratum"],
                }
            )
            result.append(value)
    return sorted(result, key=lambda row: row["section_id"])


def build_records(response_root: Path, stepwise_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = read_json(response_root / "SELECTION_MANIFEST.json")
    identity = read_json(response_root / "IDENTITY_MATERIAL.json")
    inventory = read_json(response_root / "RESOLVED_INVENTORY.json")
    geometry = read_json(response_root / "UPDATE_GEOMETRY_CONTROL_MANIFEST.json")
    require(selection["status"] == "SELECTION_FROZEN_NO_NEW_CURVES_VIEWED", "SELECTION_NOT_FROZEN")
    require(inventory["status"] == "PASS", "INVENTORY_NOT_PASS")
    require(geometry["curve_values_read"] is False, "GEOMETRY_WAS_NOT_PREFROZEN")
    require(selection["global_unseen_policy"]["entry_id"] == GLOBAL_UNSEEN_ENTRY, "UNSEEN_POLICY_ID_MISMATCH")
    sections = _flatten_sections(selection)
    inventory_by_section = {row["section_id"]: row for row in inventory["sections"]}
    geometry_by_section = {row["section_id"]: row for row in geometry["section_geometry"]}
    pair_previous: dict[str, str] = {}
    for pair in selection["pairs"]:
        ordered = sorted(pair["sections"], key=lambda row: int(row["section_ordinal_within_pair"]))
        pair_previous[str(ordered[1]["section_id"])] = str(ordered[0]["section_id"])

    history = ProbeHistory(stepwise_root)
    records: list[dict[str, Any]] = []
    section_response: dict[str, dict[str, dict[str, Any]]] = {}
    source_files: set[str] = set()
    availability_counts: dict[str, int] = {}

    for section in sections:
        section_id = str(section["section_id"])
        entry_id = str(section["entry_id"])
        require(entry_id != GLOBAL_UNSEEN_ENTRY, "GLOBAL_UNSEEN_ACCESSED")
        resolved = inventory_by_section[section_id]
        geom = geometry_by_section[section_id]
        data_path = response_root / "sections" / f"{section_id}.npz"
        metadata_path = response_root / "sections" / f"{section_id}.json"
        source_files.update((str(data_path), str(metadata_path), str(resolved["transition_path"]), str(resolved["receiver_probe_path"])))
        metadata = read_json(metadata_path)
        require(file_sha256(data_path) == metadata["data_file_sha256"], f"SECTION_HASH_MISMATCH:{section_id}")
        with np.load(data_path, allow_pickle=False) as data:
            require(np.array_equal(data["alphas"], ALPHAS), f"ALPHA_GRID_MISMATCH:{section_id}")
            groups = np.asarray(data["groups"], dtype=np.int64)
            all_logits = np.asarray(data["all_logits"], dtype=np.float64)
            all_margins = np.asarray(data["all_margins"], dtype=np.float64)
            all_predictions = np.asarray(data["all_predictions"], dtype=np.int64)
            competitors = np.asarray(data["baseline_competitor_ids"], dtype=np.int64)
            necessity = np.asarray(data["necessity"], dtype=np.float64)
            pair_backup = np.asarray(data["pair_backup"], dtype=np.float64)
            allocation = np.asarray(data["support_allocation"], dtype=np.float64)
            concentration = np.asarray(data["support_concentration"], dtype=np.float64)
            effective = np.asarray(data["effective_support"], dtype=np.float64)
            single_slack = np.asarray(data["single_failure_slack"], dtype=np.float64)
            double_slack = np.asarray(data["double_failure_slack"], dtype=np.float64)

        identities = identity["entries"][entry_id]
        require(len(identities) == len(groups) == 212, f"IDENTITY_COUNT_MISMATCH:{section_id}")
        for index, row in enumerate(identities):
            require(int(row["target_group"]) == int(groups[index]), f"IDENTITY_GROUP_MISMATCH:{section_id}:{index}")

        transition_path = Path(str(resolved["transition_path"]))
        transition = read_json(transition_path)
        entry_root = stepwise_root / entry_id
        update_descriptor = transition["step"]["parameter_update"]
        update_flat = _load_locator(entry_root, update_descriptor).astype(np.float64)
        wte_update = _layout_slice(update_flat, update_descriptor, "transformer.wte.weight")
        receiver_probe = read_json(Path(str(resolved["receiver_probe_path"])))
        step = int(section["receiver_optimizer_step_audit_only"])
        window_path = transition_path.parent.parent
        component_features = _component_update_features(geom["component_update_l2_norms"])
        component_features["update_l2_norm"] = float(geom["update_l2_norm"])

        current_logits = all_logits[1, 0]
        current_margins = all_margins[1, 0]
        current_predictions = all_predictions[1, 0]
        current_competitors = competitors[1]
        order_without_true: list[np.ndarray] = []
        for idx, target in enumerate(groups):
            logits = current_logits[idx].copy()
            logits[int(target)] = -np.inf
            order_without_true.append(np.argsort(logits)[::-1][:3])

        past: dict[int, dict[str, Any] | None] = {
            lag: history.probe(entry_id, window_path, step - lag) for lag in range(1, 6)
        }
        past_update_norms = {lag: history.update_norm(entry_id, window_path, step - lag) for lag in range(1, 6)}

        section_response[section_id] = {}
        for index, identity_row in enumerate(identities):
            target = int(groups[index])
            top = order_without_true[index]
            competitor = int(current_competitors[index])
            require(competitor == int(top[0]), f"COMPETITOR_ID_MISMATCH:{section_id}:{index}")
            margin_curve = all_margins[:, 0, index]
            delta_curve = margin_curve - margin_curve[1]
            scale = float(np.max(np.abs(delta_curve))) + 1e-12
            normalized_curve = delta_curve / scale
            response_type, response_type_detail = _response_type(margin_curve)
            competitor_curve = competitors[:, index]
            competitor_changes = int(np.sum(competitor_curve[1:] != competitor_curve[:-1]))
            first_switch = next((float(ALPHAS[j]) for j in range(2, len(ALPHAS)) if competitor_curve[j] != competitor_curve[1]), None)
            semantic_key = f"{identity_row['row_content_sha256']}:{target}"
            evaluation_unit_id = str(identity_row["evaluation_unit_id"])

            true_row = wte_update[target]
            comp_row = wte_update[competitor]
            true_norm = float(np.linalg.norm(true_row))
            comp_norm = float(np.linalg.norm(comp_row))
            denom = true_norm * comp_norm
            row_cosine = float(np.dot(true_row, comp_row) / denom) if denom > 1e-20 else 0.0
            row_difference = float(np.linalg.norm(true_row - comp_row))
            row_cancellation = float((true_norm + comp_norm - row_difference) / (true_norm + comp_norm + 1e-12))

            f1 = {
                "margin": float(current_margins[index]),
                "correct": float(current_predictions[index] == target),
                "correct_logit": float(current_logits[index, target]),
                "competitor1_logit": float(current_logits[index, top[0]]),
                "competitor2_logit": float(current_logits[index, top[1]]),
                "competitor3_logit": float(current_logits[index, top[2]]),
                "competitor12_gap": float(current_logits[index, top[0]] - current_logits[index, top[1]]),
                "competitor13_gap": float(current_logits[index, top[0]] - current_logits[index, top[2]]),
                "competitor_crowding_0_5": float(np.sum((current_logits[index, top[0]] - current_logits[index]) <= 0.5) - 1),
                "abs_margin": abs(float(current_margins[index])),
            }
            previous_competitor = None
            if past[1] is not None:
                prior_logits = np.asarray(past[1]["logits"])[index].copy()
                prior_logits[target] = -np.inf
                previous_competitor = int(np.argmax(prior_logits))
            f2_num = {
                "competitor12_gap": f1["competitor12_gap"],
                "competitor13_gap": f1["competitor13_gap"],
                "near_tie_0_1": float(f1["competitor12_gap"] <= 0.1),
                "prior_competitor_switch": float(previous_competitor is not None and previous_competitor != competitor),
            }
            f2_cat = {
                "competitor1": str(int(top[0])),
                "competitor2": str(int(top[1])),
                "competitor3": str(int(top[2])),
                "prior_competitor": "MISSING" if previous_competitor is None else str(previous_competitor),
            }
            f3 = dict(component_features)
            f3.update(
                {
                    "target_row_update_l2": true_norm,
                    "competitor_row_update_l2": comp_norm,
                    "target_competitor_row_difference_l2": row_difference,
                    "target_competitor_row_cosine": row_cosine,
                    "target_competitor_row_cancellation": row_cancellation,
                }
            )
            group = target
            f4 = {
                **{f"necessity_{COMPONENTS[j].replace('.', '_')}": float(necessity[1, j, group]) for j in range(4)},
                **{f"allocation_{COMPONENTS[j].replace('.', '_')}": float(allocation[1, j, group]) for j in range(4)},
                **{f"backup_pair_{j}": float(pair_backup[1, j, group]) for j in range(6)},
                "support_concentration": float(concentration[1, group]),
                "effective_support": float(effective[1, group]),
                "single_failure_slack": float(single_slack[1, group]),
                "double_failure_slack": float(double_slack[1, group]),
            }
            for feature_name in list(f4):
                undefined = not math.isfinite(float(f4[feature_name]))
                f4[f"{feature_name}_undefined"] = float(undefined)
                if undefined:
                    f4[feature_name] = 0.0
            f4_cat = {
                "primary_support_component": COMPONENTS[int(np.nanargmax(necessity[1, :, group]))],
                "backup_pair_identity": str(int(np.nanargmax(pair_backup[1, :, group]))),
            }
            f5: dict[str, float] = {}
            load_rows: list[list[float]] = []
            for component in COMPONENTS:
                loads = receiver_probe["component_loads"][component]
                row_values = []
                for name in ("parameter_rms", "exp_avg_rms", "exp_avg_sq_sqrt_mean", "preconditioned_rms"):
                    value = float(loads[name])
                    f5[f"{component.replace('.', '_')}_{name}"] = value
                    row_values.append(value)
                load_rows.append(row_values)
            load_matrix = np.asarray(load_rows, dtype=np.float64)
            for col, name in enumerate(("parameter_rms", "exp_avg_rms", "exp_avg_sq_sqrt_mean", "preconditioned_rms")):
                values = load_matrix[:, col]
                f5[f"component_{name}_cv"] = float(np.std(values) / (abs(float(np.mean(values))) + 1e-12))
            f5["update_adam_preconditioned_interaction"] = float(
                component_features["component_update_concentration"] * np.mean(load_matrix[:, 3])
            )

            f6: dict[str, float] = {}
            f6_cat: dict[str, str] = {}
            past_competitors: list[int] = []
            available_history = 0
            for lag in range(1, 6):
                previous = past[lag]
                missing = previous is None
                update_missing = past_update_norms[lag] is None
                f6[f"history_missing_{lag}"] = float(missing)
                f6[f"prior_update_missing_{lag}"] = float(update_missing)
                f6[f"prior_update_l2_{lag}"] = 0.0 if update_missing else float(past_update_norms[lag])
                if missing:
                    for name in ("margin", "correct", "capability", "effective_support", "support_concentration"):
                        f6[f"{name}_lag_{lag}"] = 0.0
                        f6[f"{name}_undefined_lag_{lag}"] = 1.0
                    f6_cat[f"competitor_lag_{lag}"] = "MISSING"
                    continue
                available_history += 1
                previous_logits = np.asarray(previous["logits"])[index].copy()
                previous_logits[target] = -np.inf
                previous_comp = int(np.argmax(previous_logits))
                past_competitors.append(previous_comp)
                historical_values = {
                    "margin": float(np.asarray(previous["margins"])[index]),
                    "correct": float(int(np.asarray(previous["predictions"])[index]) == target),
                    "capability": float(previous["capability"]),
                    "effective_support": float(np.asarray(previous["effective_support"])[group]),
                    "support_concentration": float(np.asarray(previous["support_concentration"])[group]),
                }
                for name, historical_value in historical_values.items():
                    undefined = not math.isfinite(historical_value)
                    f6[f"{name}_lag_{lag}"] = 0.0 if undefined else historical_value
                    f6[f"{name}_undefined_lag_{lag}"] = float(undefined)
                f6_cat[f"competitor_lag_{lag}"] = str(previous_comp)
            f6["history_available_count"] = float(available_history)
            if past[1] is not None:
                f6["margin_velocity_1"] = float(current_margins[index] - f6["margin_lag_1"])
                f6["support_velocity_1"] = float(f4["effective_support"] - f6["effective_support_lag_1"])
                f6["margin_velocity_1_undefined"] = f6["margin_undefined_lag_1"]
                f6["support_velocity_1_undefined"] = float(
                    bool(f4["effective_support_undefined"]) or bool(f6["effective_support_undefined_lag_1"])
                )
                if f6["margin_velocity_1_undefined"]:
                    f6["margin_velocity_1"] = 0.0
                if f6["support_velocity_1_undefined"]:
                    f6["support_velocity_1"] = 0.0
            else:
                f6["margin_velocity_1"] = 0.0
                f6["support_velocity_1"] = 0.0
                f6["margin_velocity_1_undefined"] = 1.0
                f6["support_velocity_1_undefined"] = 1.0
            if past_competitors:
                f6["competitor_switch_count_5"] = float(sum(a != b for a, b in zip(past_competitors, past_competitors[1:])))
            else:
                f6["competitor_switch_count_5"] = 0.0

            response = {
                "margin_curve": [float(x) for x in margin_curve],
                "displacement_curve": [float(x) for x in delta_curve],
                "normalized_curve": [float(x) for x in normalized_curve],
                "endpoint_delta": float(delta_curve[-1]),
                "response_type": response_type,
                "response_type_detail": response_type_detail,
                "competitor_switch": bool(competitor_changes > 0),
                "competitor_switch_count": competitor_changes,
                "first_competitor_switch_alpha": first_switch,
                "final_competitor": str(int(competitor_curve[-1])),
                "boundary_class": _boundary_class(margin_curve),
                "first_zero_crossing_alpha": _first_crossing(ALPHAS, margin_curve),
            }
            record = {
                "record_id": "factorunit-" + sha256_bytes(f"{section_id}:{evaluation_unit_id}".encode("utf-8"))[:32],
                "section_id": section_id,
                "pair_id": str(section["pair_id"]),
                "section_ordinal": int(section["section_ordinal_within_pair"]),
                "entry_id": entry_id,
                "window_id": str(section["window_id"]),
                "optimizer_step": step,
                "evaluation_unit_id": evaluation_unit_id,
                "semantic_target_key": semantic_key,
                "row_content_sha256": str(identity_row["row_content_sha256"]),
                "upstream_element_identity": str(identity_row["upstream_element_identity"]),
                "target_group": target,
                "features": {
                    "F1": {"numeric": f1, "categorical": {}},
                    "F2": {"numeric": f2_num, "categorical": f2_cat},
                    "F3": {"numeric": f3, "categorical": {}},
                    "F4": {"numeric": f4, "categorical": f4_cat},
                    "F5": {"numeric": f5, "categorical": {}},
                    "F6": {"numeric": f6, "categorical": f6_cat},
                    "F7": {"numeric": {}, "categorical": {"availability": "MISSING"}},
                },
                "response": response,
                "source_refs": {
                    "section_npz": str(data_path),
                    "section_npz_sha256": file_sha256(data_path),
                    "transition": str(transition_path),
                    "transition_sha256": file_sha256(transition_path),
                    "receiver_probe": str(resolved["receiver_probe_path"]),
                    "receiver_probe_sha256": file_sha256(Path(str(resolved["receiver_probe_path"]))),
                },
            }
            records.append(record)
            section_response[section_id][semantic_key] = response

    for record in records:
        previous_section = pair_previous.get(record["section_id"])
        if previous_section is None:
            continue
        previous = section_response[previous_section].get(record["semantic_target_key"])
        require(previous is not None, f"PRIOR_CURVE_IDENTITY_MISSING:{record['record_id']}")
        numeric = {f"prior_q_{index}": float(value) for index, value in enumerate(previous["normalized_curve"])}
        numeric["prior_endpoint_delta"] = float(previous["endpoint_delta"])
        numeric["prior_competitor_switch_count"] = float(previous["competitor_switch_count"])
        record["features"]["F7"] = {
            "numeric": numeric,
            "categorical": {
                "availability": "AVAILABLE",
                "prior_response_type": str(previous["response_type"]),
                "prior_boundary_class": str(previous["boundary_class"]),
            },
        }
        record["prior_curve_section_id"] = previous_section

    for block in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        availability_counts[block] = sum(
            record["features"][block]["categorical"].get("availability", "AVAILABLE") == "AVAILABLE"
            for record in records
        )
    require(len(records) == 72 * 212, "FACTOR_RECORD_COUNT_INVALID")
    audit = {
        "schema": "nanogpt-response-factor-feature-availability-v1",
        "status": "PASS",
        "record_count": len(records),
        "section_count": 72,
        "entry_count": 12,
        "availability_counts": availability_counts,
        "natural_pretarget_blocks": ["F1", "F2", "F3", "F5", "F6"],
        "pretarget_diagnostic_blocks": ["F4"],
        "prior_active_diagnostic_memory_blocks": ["F7"],
        "diagnostic_probe_derived_unavailable": [
            "full target-logit parameter Jacobian",
            "target-specific hidden-state/update alignment outside output embedding rows",
            "counterfactual component-specific update endpoints",
        ],
        "target_specific_update_geometry_available": [
            "correct-class output-embedding update row",
            "current-competitor output-embedding update row",
            "row difference, cosine and cancellation",
        ],
        "global_unseen_entry_accessed": False,
        "source_file_count": len(source_files),
    }
    return records, audit


def build_feature_space(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks: dict[str, Any] = {}
    serialized: dict[str, Any] = {
        "schema": "nanogpt-response-factor-standardization-v1",
        "method": "median/IQR; missing numeric -> standardized zero plus missing indicator",
        "blocks": {},
    }
    for block in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        numeric_names = sorted({name for row in records for name in row["features"][block]["numeric"]})
        categorical_names = sorted({name for row in records for name in row["features"][block]["categorical"]})
        raw = np.full((len(records), len(numeric_names)), np.nan, dtype=np.float64)
        for row_index, row in enumerate(records):
            for column, name in enumerate(numeric_names):
                value = row["features"][block]["numeric"].get(name)
                if value is not None and math.isfinite(float(value)):
                    raw[row_index, column] = float(value)
        median = np.nanmedian(raw, axis=0) if numeric_names else np.empty(0, dtype=np.float64)
        q25 = np.nanpercentile(raw, 25, axis=0) if numeric_names else np.empty(0, dtype=np.float64)
        q75 = np.nanpercentile(raw, 75, axis=0) if numeric_names else np.empty(0, dtype=np.float64)
        iqr = q75 - q25
        keep = np.isfinite(median) & np.isfinite(iqr) & (iqr >= 1e-12)
        kept_names = [name for name, flag in zip(numeric_names, keep, strict=True) if flag]
        dropped_names = [name for name, flag in zip(numeric_names, keep, strict=True) if not flag]
        kept_raw = raw[:, keep]
        kept_median = median[keep]
        kept_iqr = iqr[keep]
        missing = ~np.isfinite(kept_raw)
        standardized = (np.where(missing, kept_median, kept_raw) - kept_median) / kept_iqr
        numeric_matrix = np.concatenate((standardized, missing.astype(np.float64)), axis=1) if kept_names else np.empty((len(records), 0))
        numeric_matrix_names = kept_names + [f"{name}__missing" for name in kept_names]
        categorical_matrix = np.empty((len(records), len(categorical_names)), dtype=object)
        for row_index, row in enumerate(records):
            for column, name in enumerate(categorical_names):
                categorical_matrix[row_index, column] = str(row["features"][block]["categorical"].get(name, "MISSING"))
        blocks[block] = {
            "numeric": numeric_matrix,
            "categorical": categorical_matrix,
            "numeric_names": numeric_matrix_names,
            "categorical_names": categorical_names,
        }
        serialized["blocks"][block] = {
            "original_numeric_names": numeric_names,
            "kept_numeric_names": kept_names,
            "dropped_numeric_names": dropped_names,
            "median": {name: float(value) for name, value in zip(kept_names, kept_median, strict=True)},
            "iqr": {name: float(value) for name, value in zip(kept_names, kept_iqr, strict=True)},
            "categorical_names": categorical_names,
            "distance_numeric_names": numeric_matrix_names,
        }
    return blocks, serialized


def _block_distance_squared(space: dict[str, Any], block: str, query: int, candidates: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    numeric = space[block]["numeric"]
    categorical = space[block]["categorical"]
    if numeric.shape[1]:
        values.append(np.mean(np.square(numeric[candidates] - numeric[query]), axis=1))
    if categorical.shape[1]:
        values.append(np.mean(categorical[candidates] != categorical[query], axis=1).astype(np.float64))
    if not values:
        return np.zeros(len(candidates), dtype=np.float64)
    return np.mean(np.vstack(values), axis=0)


def _response_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    left_q = np.asarray(left["normalized_curve"], dtype=np.float64)[POSITIVE_CURVE_INDICES]
    right_q = np.asarray(right["normalized_curve"], dtype=np.float64)[POSITIVE_CURVE_INDICES]
    left_d = np.asarray(left["displacement_curve"], dtype=np.float64)[POSITIVE_CURVE_INDICES]
    right_d = np.asarray(right["displacement_curve"], dtype=np.float64)[POSITIVE_CURVE_INDICES]
    first_left = left["first_competitor_switch_alpha"]
    first_right = right["first_competitor_switch_alpha"]
    zero_left = left["first_zero_crossing_alpha"]
    zero_right = right["first_zero_crossing_alpha"]
    return {
        "normalized_shape_correlation": _corr(left_q, right_q),
        "normalized_shape_rmse": float(np.sqrt(np.mean(np.square(left_q - right_q)))),
        "raw_displacement_correlation": _corr(left_d, right_d),
        "endpoint_abs_difference": abs(float(left["endpoint_delta"]) - float(right["endpoint_delta"])),
        "endpoint_direction_agreement": float(np.sign(left["endpoint_delta"]) == np.sign(right["endpoint_delta"])),
        "response_type_agreement": float(left["response_type"] == right["response_type"]),
        "competitor_switch_agreement": float(bool(left["competitor_switch"]) == bool(right["competitor_switch"])),
        "first_switch_alpha_abs_difference": 0.0 if first_left is None and first_right is None else (1.125 if first_left is None or first_right is None else abs(float(first_left) - float(first_right))),
        "first_zero_alpha_abs_difference": 0.0 if zero_left is None and zero_right is None else (1.125 if zero_left is None or zero_right is None else abs(float(zero_left) - float(zero_right))),
        "boundary_class_agreement": float(left["boundary_class"] == right["boundary_class"]),
        "correct_to_wrong_agreement": float((left["boundary_class"] == "CORRECT_TO_WRONG") == (right["boundary_class"] == "CORRECT_TO_WRONG")),
        "wrong_to_correct_agreement": float((left["boundary_class"] == "WRONG_TO_CORRECT") == (right["boundary_class"] == "WRONG_TO_CORRECT")),
        "maintain_correct_agreement": float((left["boundary_class"] == "MAINTAIN_CORRECT") == (right["boundary_class"] == "MAINTAIN_CORRECT")),
        "maintain_wrong_agreement": float((left["boundary_class"] == "MAINTAIN_WRONG") == (right["boundary_class"] == "MAINTAIN_WRONG")),
    }


def _average_metrics(metrics: list[dict[str, float | None]]) -> dict[str, float | None]:
    names = sorted({name for row in metrics for name in row})
    result: dict[str, float | None] = {}
    for name in names:
        values = [float(row[name]) for row in metrics if row.get(name) is not None and math.isfinite(float(row[name]))]
        result[name] = float(np.mean(values)) if values else None
    return result


def match_configuration(
    records: list[dict[str, Any]],
    space: dict[str, Any],
    *,
    name: str,
    blocks: tuple[str, ...],
    pool: str,
    k: int = PRIMARY_K,
    caliper_quantile: float = PRIMARY_CALIPER_QUANTILE,
    subset: set[int] | None = None,
    random_selection: bool = False,
    exclude_same_semantic: bool = False,
) -> dict[str, Any]:
    require(pool in {"group", "semantic"}, f"INVALID_POOL:{pool}")
    included = sorted(subset if subset is not None else set(range(len(records))))
    group_pool: dict[int, np.ndarray] = {}
    semantic_pool: dict[str, np.ndarray] = {}
    for index in included:
        group_pool.setdefault(int(records[index]["target_group"]), []).append(index)
        semantic_pool.setdefault(str(records[index]["semantic_target_key"]), []).append(index)
    group_pool = {key: np.asarray(value, dtype=np.int64) for key, value in group_pool.items()}
    semantic_pool = {key: np.asarray(value, dtype=np.int64) for key, value in semantic_pool.items()}

    provisional: list[dict[str, Any]] = []
    for query in included:
        key: int | str = records[query]["target_group"] if pool == "group" else records[query]["semantic_target_key"]
        candidates = (group_pool if pool == "group" else semantic_pool).get(key, np.empty(0, dtype=np.int64))
        candidates = candidates[np.asarray([records[int(index)]["entry_id"] != records[query]["entry_id"] for index in candidates])]
        if exclude_same_semantic:
            candidates = candidates[
                np.asarray(
                    [records[int(index)]["semantic_target_key"] != records[query]["semantic_target_key"] for index in candidates]
                )
            ]
        if candidates.size == 0:
            provisional.append({"query": query, "indices": np.empty(0, dtype=np.int64), "distances": np.empty(0)})
            continue
        if random_selection or not blocks:
            ordered = sorted(
                (int(index) for index in candidates),
                key=lambda index: sha256_bytes(f"{BASE_SEED}:{name}:{records[query]['record_id']}:{records[index]['record_id']}".encode("utf-8")),
            )
            selected = np.asarray(ordered[:k], dtype=np.int64)
            distances = np.zeros(len(selected), dtype=np.float64)
        else:
            block_d2 = [_block_distance_squared(space, block, query, candidates) for block in blocks]
            distances_all = np.sqrt(np.mean(np.vstack(block_d2), axis=0))
            order = np.argsort(distances_all, kind="stable")
            selected = candidates[order[:k]]
            distances = distances_all[order[:k]]
        provisional.append({"query": query, "indices": selected, "distances": distances})

    nearest = [float(row["distances"][0]) for row in provisional if len(row["distances"])]
    caliper = float(np.quantile(nearest, caliper_quantile)) if nearest and blocks and not random_selection else None
    rows: list[dict[str, Any]] = []
    for row in provisional:
        query = int(row["query"])
        selected: np.ndarray = row["indices"]
        distances: np.ndarray = row["distances"]
        if selected.size == 0 or (caliper is not None and float(distances[0]) > caliper):
            continue
        if caliper is not None:
            mask = distances <= caliper
            selected = selected[mask]
            distances = distances[mask]
        if selected.size == 0:
            continue
        pair_metrics = [_response_metrics(records[query]["response"], records[int(reference)]["response"]) for reference in selected]
        metrics = _average_metrics(pair_metrics)
        reference_entries = [str(records[int(reference)]["entry_id"]) for reference in selected]
        cluster = "|".join(sorted((str(records[query]["entry_id"]), reference_entries[0])))
        rows.append(
            {
                "query_index": query,
                "query_record_id": records[query]["record_id"],
                "query_section_id": records[query]["section_id"],
                "query_entry_id": records[query]["entry_id"],
                "reference_indices": [int(value) for value in selected],
                "reference_record_ids": [records[int(value)]["record_id"] for value in selected],
                "reference_section_ids": [records[int(value)]["section_id"] for value in selected],
                "reference_entry_ids": reference_entries,
                "distances": [float(value) for value in distances],
                "cluster_run_pair": cluster,
                "metrics": metrics,
            }
        )
    return {
        "name": name,
        "blocks": list(blocks),
        "pool": pool,
        "k": k,
        "caliper_quantile": caliper_quantile,
        "caliper": caliper,
        "eligible_query_count": len(included),
        "matched_query_count": len(rows),
        "unmatched_query_count": len(included) - len(rows),
        "exclude_same_semantic": exclude_same_semantic,
        "rows": rows,
    }


def _cluster_bootstrap(values: list[tuple[str, float]], label: str) -> dict[str, float | int | None]:
    grouped: dict[str, list[float]] = {}
    for cluster, value in values:
        if math.isfinite(float(value)):
            grouped.setdefault(cluster, []).append(float(value))
    if not grouped:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "cluster_count": 0}
    point = float(np.mean([value for cluster in grouped.values() for value in cluster]))
    clusters = sorted(grouped)
    rng = np.random.default_rng(stable_seed(label))
    sums = np.asarray([np.sum(grouped[cluster]) for cluster in clusters], dtype=np.float64)
    counts = np.asarray([len(grouped[cluster]) for cluster in clusters], dtype=np.float64)
    chosen = rng.integers(0, len(clusters), size=(BOOTSTRAP_REPLICATES, len(clusters)))
    draws = np.sum(sums[chosen], axis=1) / np.sum(counts[chosen], axis=1)
    return {
        "estimate": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "cluster_count": len(clusters),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def summarize_configuration(config: dict[str, Any], label: str) -> dict[str, Any]:
    rows = config["rows"]
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    metrics: dict[str, Any] = {}
    for metric in metric_names:
        values = [
            (row["cluster_run_pair"], float(row["metrics"][metric]))
            for row in rows
            if row["metrics"].get(metric) is not None
        ]
        metrics[metric] = _cluster_bootstrap(values, f"{label}:{metric}")
    reference_ids = [reference for row in rows for reference in row["reference_record_ids"]]
    reference_counts: dict[str, int] = {}
    for reference in reference_ids:
        reference_counts[reference] = reference_counts.get(reference, 0) + 1
    entries = sorted({row["query_entry_id"] for row in rows} | {entry for row in rows for entry in row["reference_entry_ids"]})
    run_pairs = sorted({row["cluster_run_pair"] for row in rows})
    per_query_entry: dict[str, dict[str, float | int | None]] = {}
    for entry in sorted({row["query_entry_id"] for row in rows}):
        selected = [row for row in rows if row["query_entry_id"] == entry]
        shape = [float(row["metrics"]["normalized_shape_correlation"]) for row in selected if row["metrics"]["normalized_shape_correlation"] is not None]
        per_query_entry[entry] = {
            "matched_count": len(selected),
            "mean_normalized_shape_correlation": float(np.mean(shape)) if shape else None,
        }
    return {
        "name": config["name"],
        "blocks": config["blocks"],
        "pool": config["pool"],
        "k": config["k"],
        "caliper_quantile": config["caliper_quantile"],
        "caliper": config["caliper"],
        "eligible_query_count": config["eligible_query_count"],
        "matched_query_count": config["matched_query_count"],
        "unmatched_query_count": config["unmatched_query_count"],
        "unmatched_fraction": config["unmatched_query_count"] / max(1, config["eligible_query_count"]),
        "covered_entry_count": len(entries),
        "covered_run_pair_count": len(run_pairs),
        "unique_reference_count": len(reference_counts),
        "maximum_reference_reuse": max(reference_counts.values()) if reference_counts else 0,
        "metrics": metrics,
        "per_query_entry": per_query_entry,
    }


def compare_configurations(current: dict[str, Any], previous: dict[str, Any], label: str) -> dict[str, Any]:
    current_by_query = {row["query_record_id"]: row for row in current["rows"]}
    previous_by_query = {row["query_record_id"]: row for row in previous["rows"]}
    shared = sorted(set(current_by_query) & set(previous_by_query))
    orientations = {
        "normalized_shape_correlation": 1.0,
        "normalized_shape_rmse": -1.0,
        "raw_displacement_correlation": 1.0,
        "endpoint_abs_difference": -1.0,
        "endpoint_direction_agreement": 1.0,
        "response_type_agreement": 1.0,
        "competitor_switch_agreement": 1.0,
        "first_switch_alpha_abs_difference": -1.0,
        "first_zero_alpha_abs_difference": -1.0,
        "boundary_class_agreement": 1.0,
        "correct_to_wrong_agreement": 1.0,
        "wrong_to_correct_agreement": 1.0,
        "maintain_correct_agreement": 1.0,
        "maintain_wrong_agreement": 1.0,
    }
    metrics: dict[str, Any] = {}
    for metric, orientation in orientations.items():
        values: list[tuple[str, float]] = []
        for query in shared:
            left = current_by_query[query]
            right = previous_by_query[query]
            if left["metrics"].get(metric) is None or right["metrics"].get(metric) is None:
                continue
            improvement = orientation * (float(left["metrics"][metric]) - float(right["metrics"][metric]))
            values.append((left["cluster_run_pair"], improvement))
        metrics[metric] = _cluster_bootstrap(values, f"{label}:{metric}")
        metrics[metric]["positive_means_current_is_better"] = True
    return {
        "current": current["name"],
        "previous": previous["name"],
        "shared_query_count": len(shared),
        "improvements": metrics,
    }


def write_match_ledger(path: Path, configs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for config in configs:
            for row in config["rows"]:
                payload = {"configuration": config["name"], **row}
                handle.write(canonical_json(payload) + "\n")
                count += 1
    return {"path": path.name, "sha256": file_sha256(path), "row_count": count, "compression": "gzip"}
