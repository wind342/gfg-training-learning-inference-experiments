from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
import zlib

import numpy as np

from experiments.gfg_nanogpt_support_redundancy_v1.task_mapping import (
    recover_cyclic_target_mapping,
)


COMPONENTS = ("h0.attn", "h0.mlp", "h1.attn", "h1.mlp")
PAIRS = tuple(
    (left, right)
    for index, left in enumerate(COMPONENTS)
    for right in COMPONENTS[index + 1 :]
)
PHASES = ("pre_formation", "formed", "decline", "recovered")
MASTER_SEED = 20260829
NULL_REPLICATES = 1000
EPS = 1e-12

DEFAULT_PROJECTION_ROOT = Path(
    r"E:\gfg-evidence\nanogpt-training-learning-inference-projection-v1\gfg"
)
DEFAULT_SOURCE_ROOT = Path(
    r"D:\codex\source-information-continuity-gfg-nanogpt-autonomous-capability-v1"
    r"\data_private\gfg_nanogpt_mechanism_discovery_archive\gfg"
)
HERE = Path(__file__).resolve().parent


def demand(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_blocks(run_root: Path) -> dict[str, tuple[int, dict[str, Any]]]:
    database = run_root / "training_learning_inference_gfg.sqlite3"
    output: dict[str, tuple[int, dict[str, Any]]] = {}
    database_uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
            "SELECT stage,optimizer_step,payload_zlib FROM graph_blocks ORDER BY block_ordinal"
        ):
            output[str(row["stage"])] = (
                int(row["optimizer_step"]),
                json.loads(zlib.decompress(row["payload_zlib"])),
            )
    return output


def read_origins(run_root: Path) -> dict[str, dict[str, Any]]:
    database = run_root / "training_learning_inference_gfg.sqlite3"
    output: dict[str, dict[str, Any]] = {}
    database_uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        query = (
            "SELECT source_bundle_id,source_object_id,source_content_sha256,"
            "source_optimizer_step,source_role,source_semantic_key FROM origin_catalog"
        )
        for row in connection.execute(query):
            role = str(row["source_role"])
            semantic = str(row["source_semantic_key"])
            if role == "training_batch_inputs":
                output["training_inputs"] = dict(row)
            elif role == "training_batch_targets":
                output["training_targets"] = dict(row)
            elif (
                role == "layer_input"
                and "evaluation_validation:module:token_embedding:input:0" in semantic
            ):
                output["validation_inputs"] = dict(row)
    demand(
        set(output) == {"training_inputs", "training_targets", "validation_inputs"},
        "FA_RECOMPUTE_SOURCE_ORIGINS",
    )
    return output


def projection_array(
    run_root: Path,
    block: dict[str, Any],
    role: str,
    gate: tuple[str, ...] | None = None,
) -> np.ndarray:
    candidates = [row for row in block["objects"] if row["role"] == role]
    if gate is not None:
        candidates = [
            row
            for row in candidates
            if row.get("payload", {}).get("gate_components") == list(gate)
        ]
    demand(len(candidates) == 1, f"FA_RECOMPUTE_OBJECT:{role}:{gate}")
    payload = candidates[0]["payload"]
    path = run_root / payload["locator"]
    demand(path.is_file(), f"FA_RECOMPUTE_MISSING:{path}")
    demand(file_sha256(path) == payload["file_sha256"], f"FA_RECOMPUTE_FILE_HASH:{path}")
    value = np.load(path, allow_pickle=False)
    demand(
        tensor_sha256(value) == payload["raw_tensor_sha256"],
        f"FA_RECOMPUTE_ARRAY_HASH:{path}",
    )
    return value


def source_array(source_root: Path, origin: dict[str, Any]) -> np.ndarray:
    path = (
        source_root
        / str(origin["source_bundle_id"])
        / "tensor-objects"
        / f"{origin['source_content_sha256']}.npy"
    )
    demand(path.is_file(), f"FA_RECOMPUTE_SOURCE_MISSING:{path}")
    value = np.load(path, allow_pickle=False)
    demand(
        tensor_sha256(value) == origin["source_content_sha256"],
        f"FA_RECOMPUTE_SOURCE_HASH:{path}",
    )
    return value


def margins(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    desired = values[np.arange(len(labels)), labels]
    competitors = values.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    return desired - np.max(competitors, axis=1)


def normalize(raw: np.ndarray, discovery: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(raw if discovery is None else discovery, dtype=np.float64)
    scale = np.maximum(np.sqrt(np.mean(reference**2, axis=0)), EPS)
    scaled = np.asarray(raw, dtype=np.float64) / scale
    norm = np.linalg.norm(scaled, axis=1)
    unit = np.zeros_like(scaled)
    usable = norm > EPS
    unit[usable] = scaled[usable] / norm[usable, None]
    return scaled, unit, scale


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > EPS else 0.0


def pair_values(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaled, unit, _ = normalize(raw)
    left, right = np.triu_indices(len(unit), 1)
    cosine = np.sum(unit[left] * unit[right], axis=1)
    correlation = np.asarray(
        [pearson(scaled[a], scaled[b]) for a, b in zip(left, right, strict=True)],
        dtype=np.float64,
    )
    distance = np.linalg.norm(unit[left] - unit[right], axis=1)
    return left, right, cosine, correlation, distance


def masks(inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    left, right = np.triu_indices(len(inputs), 1)
    left_a, right_a = inputs[left, 0], inputs[right, 0]
    left_b, right_b = inputs[left, 2], inputs[right, 2]
    left_y, right_y = labels[left], labels[right]
    commute = (left_a == right_b) & (left_b == right_a) & (left_a != left_b)
    return {
        "main_different_output_unmatched": (
            (left_y != right_y) & (left_a != right_a) & (left_b != right_b)
        ),
        "same_output_different_instance": (left_y == right_y) & ~commute,
        "commuted": commute,
        "shared_left": (left_y != right_y) & (left_a == right_a) & (left_b != right_b),
        "shared_right": (left_y != right_y) & (left_b == right_b) & (left_a != right_a),
    }


def grouped_mean(
    values: np.ndarray,
    selected: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    labels: np.ndarray,
) -> float:
    groups: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for index in np.flatnonzero(selected):
        a, b = int(pair_left[index]), int(pair_right[index])
        label_a, label_b = int(labels[a]), int(labels[b])
        key = (label_a,) if label_a == label_b else tuple(sorted((label_a, label_b)))
        groups[key].append(float(values[index]))
    return float(np.mean([np.mean(rows) for rows in groups.values()])) if groups else 0.0


def matched_main(
    reference: np.ndarray,
    main: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    reference_counts: dict[tuple[int, int], int] = defaultdict(int)
    for index in np.flatnonzero(reference):
        a, b = int(pair_left[index]), int(pair_right[index])
        key = tuple(sorted((int(labels[a]), int(labels[b]))))
        if key[0] != key[1]:
            reference_counts[key] += 1
    candidates: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(main):
        a, b = int(pair_left[index]), int(pair_right[index])
        candidates[tuple(sorted((int(labels[a]), int(labels[b]))))].append(int(index))
    selected = np.zeros_like(main, dtype=bool)
    for key, count in sorted(reference_counts.items()):
        for index in candidates.get(key, [])[:count]:
            selected[index] = True
    return selected


def class_metrics(
    raw: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    baseline_margin: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    left, right, cosine, correlation, distance = pair_values(raw)
    categories = masks(inputs, labels)
    bins = np.digitize(
        baseline_margin,
        np.quantile(baseline_margin, [0.2, 0.4, 0.6, 0.8]),
        right=True,
    )
    same_bin = bins[left] == bins[right]
    output: dict[str, dict[str, float | int]] = {}
    for name, selected in categories.items():
        output[name] = {
            "pair_count": int(np.count_nonzero(selected)),
            "cosine": grouped_mean(cosine, selected, left, right, labels),
            "pearson": grouped_mean(correlation, selected, left, right, labels),
            "normalized_euclidean": grouped_mean(distance, selected, left, right, labels),
            "same_margin_quintile_cosine": grouped_mean(
                cosine, selected & same_bin, left, right, labels
            ),
        }
    for reference in ("shared_left", "shared_right"):
        selected = matched_main(
            categories[reference],
            categories["main_different_output_unmatched"],
            left,
            right,
            labels,
        )
        output[f"fully_unmatched_matched_to_{reference}"] = {
            "pair_count": int(np.count_nonzero(selected)),
            "cosine": grouped_mean(cosine, selected, left, right, labels),
            "pearson": grouped_mean(correlation, selected, left, right, labels),
            "normalized_euclidean": grouped_mean(distance, selected, left, right, labels),
            "same_margin_quintile_cosine": grouped_mean(
                cosine, selected & same_bin, left, right, labels
            ),
        }
    return output


def heldout_score(raw: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores: list[float] = []
    eligible = 0
    for output in range(23):
        discovery = labels != output
        heldout = labels == output
        demand(np.any(heldout), f"FA_RECOMPUTE_LOO_EMPTY:{output}")
        _, discovery_unit, scale = normalize(raw[discovery])
        consensus = np.mean(discovery_unit, axis=0)
        consensus_norm = float(np.linalg.norm(consensus))
        demand(consensus_norm > EPS, f"FA_RECOMPUTE_CONSENSUS_ZERO:{output}")
        consensus /= consensus_norm
        scaled = raw[heldout] / scale
        norm = np.linalg.norm(scaled, axis=1)
        usable = norm > EPS
        demand(np.any(usable), f"FA_RECOMPUTE_HELDOUT_ZERO:{output}")
        scores.append(float(np.mean((scaled[usable] / norm[usable, None]) @ consensus)))
        eligible += int(np.count_nonzero(usable))
    return float(np.mean(scores)), eligible / len(raw)


def compare_scalar(actual: float, expected: float, code: str, tolerance: float = 1e-12) -> None:
    demand(abs(float(actual) - float(expected)) <= tolerance, f"{code}:{actual}:{expected}")


def compare_metric_tree(actual: dict[str, Any], expected: dict[str, Any], code: str) -> None:
    for category, values in actual.items():
        demand(category in expected, f"{code}_CATEGORY:{category}")
        for name, value in values.items():
            if name == "pair_count":
                demand(int(value) == int(expected[category][name]), f"{code}_COUNT:{category}")
            else:
                compare_scalar(float(value), float(expected[category][name]), f"{code}:{category}:{name}")


def recompute_nulls(
    raw: np.ndarray,
    labels: np.ndarray,
    inputs: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    coordinate = np.empty(NULL_REPLICATES, dtype=np.float64)
    output = np.empty(NULL_REPLICATES, dtype=np.float64)
    shared_left = np.empty(NULL_REPLICATES, dtype=np.float64)
    shared_right = np.empty(NULL_REPLICATES, dtype=np.float64)
    commuted = np.empty(NULL_REPLICATES, dtype=np.float64)
    pair_left, pair_right, cosine, _, _ = pair_values(raw)
    for replicate in range(NULL_REPLICATES):
        shuffled = np.empty_like(raw)
        for coordinate_index in range(raw.shape[1]):
            shuffled[:, coordinate_index] = raw[
                rng.permutation(len(raw)), coordinate_index
            ]
        coordinate[replicate] = heldout_score(shuffled, labels)[0]
        output[replicate] = heldout_score(raw, labels[rng.permutation(len(labels))])[0]
        input_masks = masks(inputs[rng.permutation(len(inputs))], labels)
        shared_left[replicate] = grouped_mean(
            cosine, input_masks["shared_left"], pair_left, pair_right, labels
        )
        shared_right[replicate] = grouped_mean(
            cosine, input_masks["shared_right"], pair_left, pair_right, labels
        )
        commuted[replicate] = grouped_mean(
            cosine, input_masks["commuted"], pair_left, pair_right, labels
        )
    return {
        "coordinate_independence": coordinate,
        "output_label_permutation": output,
        "shared_left": shared_left,
        "shared_right": shared_right,
        "commuted": commuted,
    }


def compare_array(actual: np.ndarray, expected: Any, code: str) -> None:
    expected_array = np.asarray(expected, dtype=np.float64)
    demand(actual.shape == expected_array.shape, f"{code}_SHAPE")
    maximum = float(np.max(np.abs(actual - expected_array))) if actual.size else 0.0
    demand(maximum <= 1e-12, f"{code}_VALUE:{maximum}")


def derive_verdict(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    count = len(rows)
    threshold = math.ceil(0.69 * count)
    formed = [row["formed"] for row in rows]
    excess = [row["null_excess"] for row in formed]
    percentile_pass = sum(row["coordinate_percentile"] >= 0.95 for row in formed)
    equivalence = 0
    same_gap = 0
    shared_gap = 0
    robustness_opposite = 0
    for row in formed:
        classes = row["classes"]
        main = float(classes["main_different_output_unmatched"]["cosine"])
        same = float(classes["same_output_different_instance"]["cosine"])
        shared_left = float(classes["shared_left"]["cosine"])
        shared_right = float(classes["shared_right"]["cosine"])
        equivalence += int(
            main >= same - 0.10
            and main >= shared_left - 0.10
            and main >= shared_right - 0.10
        )
        same_gap += int(same - main > 0.10)
        shared_gap += int(max(shared_left, shared_right) - main > 0.10)
        if main >= same - 0.10:
            robustness_opposite += int(
                float(classes["main_different_output_unmatched"]["pearson"])
                < float(classes["same_output_different_instance"]["pearson"]) - 0.10
                or float(classes["main_different_output_unmatched"]["normalized_euclidean"])
                > float(classes["same_output_different_instance"]["normalized_euclidean"])
                + 0.20
            )
    phase_delta = [row["formed"]["null_excess"] - row["pre_formation"]["null_excess"] for row in rows]
    positive = sum(value > 0.0 for value in phase_delta)
    eligible = all(row["eligible_fraction"] >= 0.90 for row in formed)
    commonality = (
        float(np.median(excess)) >= 0.05
        and percentile_pass >= threshold
        and equivalence >= threshold
    )
    historical = float(np.median(phase_delta)) >= 0.05 and positive >= threshold
    robustness = robustness_opposite == 0
    if commonality and historical and eligible and robustness:
        verdict = "SINGLE_RELATION_FUNCTIONAL_INVARIANCE_SUPPORTED"
    elif commonality:
        verdict = (
            "STATIC_ARCHITECTURAL_PATTERN_ONLY"
            if not historical
            else "PARTIAL_FUNCTIONAL_INVARIANCE_SUPPORTED"
        )
    elif same_gap >= threshold:
        verdict = "OUTPUT_CONDITIONED_CONVERGENCE_ONLY"
    elif shared_gap >= threshold:
        verdict = "INPUT_STRUCTURE_DOMINATED"
    elif float(np.median(excess)) <= 0.0 and percentile_pass < math.ceil(count / 2):
        verdict = "NO_FUNCTIONAL_INVARIANCE"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, {
        "run_replication_threshold": threshold,
        "formed_null_excess_median": float(np.median(excess)),
        "formed_primary_null_pass_run_count": percentile_pass,
        "formed_cross_output_equivalence_run_count": equivalence,
        "formed_minus_pre_median": float(np.median(phase_delta)),
        "formed_minus_pre_positive_run_count": positive,
        "formed_direction_eligibility_all_at_least_0_90": eligible,
        "robustness_opposite_run_count": robustness_opposite,
        "same_output_gap_over_0_10_run_count": same_gap,
        "shared_operand_gap_over_0_10_run_count": shared_gap,
    }


def independent_recompute(
    projection_root: Path,
    source_root: Path,
    results_path: Path,
    fingerprints_path: Path,
) -> dict[str, Any]:
    results = load_json(results_path)
    archive = load_json(projection_root / "ARCHIVE_MANIFEST.json")
    demand(results["status"] == "PASS", "FA_RECOMPUTE_RESULT_STATUS")
    demand(archive["status"] == "PASS", "FA_RECOMPUTE_ARCHIVE_STATUS")
    demand(file_sha256(fingerprints_path) == results["fingerprint_archive"]["sha256"], "FA_RECOMPUTE_NPZ_HASH")
    result_by_id = {row["entry_id"]: row for row in results["run_rows"]}
    recomputed_rows: list[dict[str, Any]] = []
    checked_arrays = 0
    with np.load(fingerprints_path, allow_pickle=False) as fingerprints:
        for run_index, entry in enumerate(archive["entries"]):
            entry_id = str(entry["entry_id"])
            demand(entry_id in result_by_id, f"FA_RECOMPUTE_RUN_MISSING:{entry_id}")
            expected_run = result_by_id[entry_id]
            run_root = projection_root / entry_id
            manifest = run_root / "GFG_MANIFEST.json"
            validation = run_root / "GFG_VALIDATION.json"
            demand(file_sha256(manifest) == entry["manifest_sha256"], f"FA_RECOMPUTE_MANIFEST:{entry_id}")
            demand(file_sha256(validation) == entry["validation_sha256"], f"FA_RECOMPUTE_VALIDATION:{entry_id}")
            manifest_payload = load_json(manifest)
            demand(
                file_sha256(run_root / manifest_payload["database"])
                == manifest_payload["database_sha256"],
                f"FA_RECOMPUTE_DATABASE:{entry_id}",
            )
            demand(load_json(validation)["status"] == "PASS", f"FA_RECOMPUTE_VALIDATION_STATUS:{entry_id}")
            blocks = read_blocks(run_root)
            origins = read_origins(run_root)
            train_inputs = source_array(source_root, origins["training_inputs"])
            train_targets = source_array(source_root, origins["training_targets"])
            validation_inputs = source_array(source_root, origins["validation_inputs"])
            target_block = blocks["target_mapping_and_query_scope"][1]
            stored_targets = projection_array(
                run_root, target_block, "derived_validation_targets"
            )
            reconstructed, _ = recover_cyclic_target_mapping(
                train_inputs, train_targets, validation_inputs
            )
            demand(np.array_equal(stored_targets, reconstructed), f"FA_RECOMPUTE_TARGETS:{entry_id}")
            labels = stored_targets[:, -1].astype(np.int64)
            phase_rows: dict[str, Any] = {}
            for phase_index, phase in enumerate(PHASES):
                block = blocks[f"native_inference:{phase}"][1]
                baseline = projection_array(run_root, block, "native_inference_logits")
                singles = {
                    component: projection_array(
                        run_root, block, "single_component_gate_logits", (component,)
                    )
                    for component in COMPONENTS
                }
                pairs = {
                    pair: projection_array(
                        run_root, block, "pair_component_gate_logits", pair
                    )
                    for pair in PAIRS
                }
                base_margin = margins(baseline, labels)
                single = np.stack(
                    [base_margin - margins(singles[component], labels) for component in COMPONENTS],
                    axis=1,
                )
                pair_effect = np.stack(
                    [base_margin - margins(pairs[pair], labels) for pair in PAIRS],
                    axis=1,
                )
                raw = np.concatenate((single, pair_effect), axis=1)
                key = f"{entry_id}__{phase}__raw"
                demand(key in fingerprints, f"FA_RECOMPUTE_NPZ_KEY:{key}")
                compare_array(raw, fingerprints[key], f"FA_RECOMPUTE_FINGERPRINT:{key}")
                checked_arrays += 1
                observed, eligible = heldout_score(raw, labels)
                expected_phase = expected_run["phases"][phase]
                compare_scalar(observed, expected_phase["loo"]["observed_score"], f"FA_RECOMPUTE_LOO:{entry_id}:{phase}")
                compare_scalar(eligible, expected_phase["direction_eligible_fraction"], f"FA_RECOMPUTE_ELIGIBLE:{entry_id}:{phase}")
                classes = class_metrics(raw, validation_inputs, labels, base_margin)
                compare_metric_tree(classes, expected_phase["classes"], f"FA_RECOMPUTE_CLASSES:{entry_id}:{phase}")
                rng = np.random.Generator(
                    np.random.PCG64(MASTER_SEED + run_index * 100 + phase_index)
                )
                nulls = recompute_nulls(raw, labels, validation_inputs, rng)
                expected_null = expected_phase["nulls"]
                compare_array(
                    nulls["coordinate_independence"],
                    expected_null["coordinate_independence"]["values"],
                    f"FA_RECOMPUTE_COORD_NULL:{entry_id}:{phase}",
                )
                compare_array(
                    nulls["output_label_permutation"],
                    expected_null["output_label_permutation"]["values"],
                    f"FA_RECOMPUTE_OUTPUT_NULL:{entry_id}:{phase}",
                )
                for name in ("shared_left", "shared_right", "commuted"):
                    compare_array(
                        nulls[name],
                        expected_null["input_identity_permutation"][name]["values"],
                        f"FA_RECOMPUTE_INPUT_NULL:{entry_id}:{phase}:{name}",
                    )
                coordinate_median = float(np.median(nulls["coordinate_independence"]))
                coordinate_percentile = float(
                    np.mean(nulls["coordinate_independence"] < observed)
                )
                compare_scalar(
                    observed - coordinate_median,
                    expected_phase["loo"]["null_excess"],
                    f"FA_RECOMPUTE_EXCESS:{entry_id}:{phase}",
                )
                phase_rows[phase] = {
                    "null_excess": observed - coordinate_median,
                    "coordinate_percentile": coordinate_percentile,
                    "eligible_fraction": eligible,
                    "classes": classes,
                }
            recomputed_rows.append(phase_rows)

    verdict, verdict_inputs = derive_verdict(recomputed_rows)
    demand(verdict == results["verdict"], f"FA_RECOMPUTE_VERDICT:{verdict}:{results['verdict']}")
    for key, actual in verdict_inputs.items():
        expected = results["verdict_inputs"][key]
        if isinstance(actual, bool):
            demand(actual is bool(expected), f"FA_RECOMPUTE_VERDICT_INPUT:{key}")
        elif isinstance(actual, int):
            demand(actual == int(expected), f"FA_RECOMPUTE_VERDICT_INPUT:{key}")
        else:
            compare_scalar(float(actual), float(expected), f"FA_RECOMPUTE_VERDICT_INPUT:{key}")

    phase_values = np.asarray(
        [[row[phase]["null_excess"] for phase in PHASES] for row in recomputed_rows],
        dtype=np.float64,
    )
    permutations = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(MASTER_SEED))
    phase_null = np.empty(10000, dtype=np.float64)
    for index in range(len(phase_null)):
        selected = permutations[rng.integers(0, len(permutations), size=len(phase_values))]
        permuted = np.take_along_axis(phase_values, selected, axis=1)
        phase_null[index] = float(np.median(permuted[:, 1] - permuted[:, 0]))
    compare_array(
        phase_null,
        results["historical_contrasts"]["phase_label_permutation"]["values"],
        "FA_RECOMPUTE_PHASE_NULL",
    )
    return {
        "schema": "fa-e01-independent-recompute-v1",
        "status": "PASS",
        "eligible_run_count": len(recomputed_rows),
        "checked_raw_fingerprint_arrays": checked_arrays,
        "checked_run_phase_null_distributions": len(recomputed_rows) * len(PHASES),
        "checked_phase_null_replicates": len(phase_null),
        "verdict": verdict,
        "results_sha256": file_sha256(results_path),
        "fingerprints_sha256": file_sha256(fingerprints_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--results", type=Path, default=HERE / "FA_E01_RESULTS.json")
    parser.add_argument("--fingerprints", type=Path, default=HERE / "FA_E01_FINGERPRINTS.npz")
    parser.add_argument("--output", type=Path, default=HERE / "FA_E01_INDEPENDENT_RECOMPUTE.json")
    args = parser.parse_args()
    result = independent_recompute(
        args.projection_root.resolve(),
        args.source_root.resolve(),
        args.results.resolve(),
        args.fingerprints.resolve(),
    )
    save_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
