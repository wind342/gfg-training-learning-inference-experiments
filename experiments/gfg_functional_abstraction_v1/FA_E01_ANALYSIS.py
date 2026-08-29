from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import zlib

import numpy as np

from experiments.gfg_nanogpt_support_redundancy_v1.task_mapping import (
    recover_cyclic_target_mapping,
)


COMPONENTS = ("h0.attn", "h0.mlp", "h1.attn", "h1.mlp")
COMPONENT_PAIRS = tuple(
    (left, right)
    for index, left in enumerate(COMPONENTS)
    for right in COMPONENTS[index + 1 :]
)
PHASES = ("pre_formation", "formed", "decline", "recovered")
MASTER_SEED = 20260829
NULL_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 73129
EPS = 1e-12

DEFAULT_PROJECTION_ROOT = Path(
    r"E:\gfg-evidence\nanogpt-training-learning-inference-projection-v1\gfg"
)
DEFAULT_SOURCE_ROOT = Path(
    r"D:\codex\source-information-continuity-gfg-nanogpt-autonomous-capability-v1"
    r"\data_private\gfg_nanogpt_mechanism_discovery_archive\gfg"
)
DEFAULT_OUTPUT = Path(__file__).with_name("FA_E01_RESULTS.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def graph_blocks(run_root: Path) -> list[tuple[str, int, dict[str, Any]]]:
    database = run_root / "training_learning_inference_gfg.sqlite3"
    rows: list[tuple[str, int, dict[str, Any]]] = []
    database_uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
            "SELECT stage,optimizer_step,payload_zlib FROM graph_blocks ORDER BY block_ordinal"
        ):
            rows.append(
                (
                    str(row["stage"]),
                    int(row["optimizer_step"]),
                    json.loads(zlib.decompress(row["payload_zlib"])),
                )
            )
    return rows


def source_origins(run_root: Path) -> dict[str, dict[str, Any]]:
    database = run_root / "training_learning_inference_gfg.sqlite3"
    result: dict[str, dict[str, Any]] = {}
    database_uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
            "SELECT source_bundle_id,source_object_id,source_content_sha256,"
            "source_optimizer_step,source_role,source_semantic_key FROM origin_catalog"
        ):
            role = str(row["source_role"])
            semantic = str(row["source_semantic_key"])
            if role == "training_batch_inputs":
                result["training_inputs"] = dict(row)
            elif role == "training_batch_targets":
                result["training_targets"] = dict(row)
            elif role == "layer_input" and "evaluation_validation:module:token_embedding:input:0" in semantic:
                result["validation_inputs"] = dict(row)
    require(set(result) == {"training_inputs", "training_targets", "validation_inputs"}, "FA_SOURCE_ORIGINS_INCOMPLETE")
    return result


def select_object(
    block: dict[str, Any], role: str, gate: Iterable[str] | None = None
) -> dict[str, Any]:
    candidates = [value for value in block["objects"] if value["role"] == role]
    if gate is not None:
        gate_list = list(gate)
        candidates = [
            value
            for value in candidates
            if value.get("payload", {}).get("gate_components") == gate_list
        ]
    require(len(candidates) == 1, f"FA_OBJECT_NOT_UNIQUE:{role}:{gate}")
    return candidates[0]


def load_projection_tensor(
    run_root: Path, block: dict[str, Any], role: str, gate: Iterable[str] | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    row = select_object(block, role, gate)
    payload = row["payload"]
    path = run_root / payload["locator"]
    require(path.is_file(), f"FA_TENSOR_MISSING:{path}")
    require(sha256_file(path) == payload["file_sha256"], f"FA_TENSOR_FILE_HASH:{path}")
    value = np.load(path, allow_pickle=False)
    require(array_sha256(value) == payload["raw_tensor_sha256"], f"FA_TENSOR_RAW_HASH:{path}")
    return value, row


def load_source_tensor(source_root: Path, origin: dict[str, Any]) -> tuple[np.ndarray, Path]:
    bundle = source_root / str(origin["source_bundle_id"])
    path = bundle / "tensor-objects" / f"{origin['source_content_sha256']}.npy"
    require(path.is_file(), f"FA_SOURCE_TENSOR_MISSING:{path}")
    value = np.load(path, allow_pickle=False)
    require(array_sha256(value) == origin["source_content_sha256"], f"FA_SOURCE_TENSOR_HASH:{path}")
    return value, path


def true_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    selected = logits64[np.arange(len(targets)), targets]
    masked = logits64.copy()
    masked[np.arange(len(targets)), targets] = -np.inf
    return selected - np.max(masked, axis=1)


def normalize_rows(raw: np.ndarray, discovery: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = raw if discovery is None else discovery
    scale = np.sqrt(np.mean(np.asarray(source, dtype=np.float64) ** 2, axis=0))
    scale = np.maximum(scale, EPS)
    scaled = np.asarray(raw, dtype=np.float64) / scale
    norms = np.linalg.norm(scaled, axis=1)
    unit = np.zeros_like(scaled)
    eligible = norms > EPS
    unit[eligible] = scaled[eligible] / norms[eligible, None]
    return scaled, unit, scale


def safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else 0.0


def pair_pearson(left: np.ndarray, right: np.ndarray) -> float:
    lx = left - np.mean(left)
    rx = right - np.mean(right)
    denom = float(np.linalg.norm(lx) * np.linalg.norm(rx))
    return float(np.dot(lx, rx) / denom) if denom > EPS else 0.0


def pair_arrays(
    scaled: np.ndarray, unit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(unit), 1)
    cosine = np.sum(unit[left] * unit[right], axis=1)
    distance = np.linalg.norm(unit[left] - unit[right], axis=1)
    correlation = np.asarray(
        [pair_pearson(scaled[a], scaled[b]) for a, b in zip(left, right, strict=True)],
        dtype=np.float64,
    )
    return left, right, cosine, correlation, distance


def class_masks(inputs: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    left_index, right_index = np.triu_indices(len(inputs), 1)
    li, ri = inputs[left_index, 0], inputs[right_index, 0]
    lj, rj = inputs[left_index, 2], inputs[right_index, 2]
    ti, tj = targets[left_index], targets[right_index]
    same_output = ti == tj
    commuted = (li == rj) & (lj == ri) & (li != lj)
    return {
        "main_different_output_unmatched": (ti != tj) & (li != ri) & (lj != rj),
        "same_output_different_instance": same_output & ~commuted,
        "commuted": commuted,
        "shared_left": (ti != tj) & (li == ri) & (lj != rj),
        "shared_right": (ti != tj) & (lj == rj) & (li != ri),
    }


def group_weighted_mean(
    values: np.ndarray,
    mask: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    targets: np.ndarray,
) -> float:
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        return 0.0
    grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for index in selected:
        a, b = int(pair_left[index]), int(pair_right[index])
        ta, tb = int(targets[a]), int(targets[b])
        key = (ta,) if ta == tb else tuple(sorted((ta, tb)))
        grouped[key].append(float(values[index]))
    return float(np.mean([np.mean(rows) for rows in grouped.values()]))


def frequency_matched_main_mask(
    reference_mask: np.ndarray,
    main_mask: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Deterministically match different-output group-pair frequencies."""

    reference_counts: dict[tuple[int, int], int] = defaultdict(int)
    for index in np.flatnonzero(reference_mask):
        a, b = int(pair_left[index]), int(pair_right[index])
        ta, tb = sorted((int(targets[a]), int(targets[b])))
        if ta != tb:
            reference_counts[(ta, tb)] += 1
    main_by_group: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(main_mask):
        a, b = int(pair_left[index]), int(pair_right[index])
        key = tuple(sorted((int(targets[a]), int(targets[b]))))
        main_by_group[key].append(int(index))
    selected = np.zeros_like(main_mask, dtype=bool)
    for key, count in sorted(reference_counts.items()):
        candidates = main_by_group.get(key, [])
        for index in candidates[: min(count, len(candidates))]:
            selected[index] = True
    return selected


def class_summary(
    raw: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    baseline_margin: np.ndarray,
) -> dict[str, Any]:
    scaled, unit, scales = normalize_rows(raw)
    pair_left, pair_right, cosine, correlation, distance = pair_arrays(scaled, unit)
    masks = class_masks(inputs, targets)
    quintiles = np.digitize(
        baseline_margin,
        np.quantile(baseline_margin, [0.2, 0.4, 0.6, 0.8]),
        right=True,
    )
    same_quintile = quintiles[pair_left] == quintiles[pair_right]
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        result[name] = {
            "pair_count": int(np.count_nonzero(mask)),
            "cosine": group_weighted_mean(cosine, mask, pair_left, pair_right, targets),
            "pearson": group_weighted_mean(correlation, mask, pair_left, pair_right, targets),
            "normalized_euclidean": group_weighted_mean(distance, mask, pair_left, pair_right, targets),
            "same_margin_quintile_cosine": group_weighted_mean(
                cosine, mask & same_quintile, pair_left, pair_right, targets
            ),
        }
    for reference in ("shared_left", "shared_right"):
        matched = frequency_matched_main_mask(
            masks[reference],
            masks["main_different_output_unmatched"],
            pair_left,
            pair_right,
            targets,
        )
        result[f"fully_unmatched_matched_to_{reference}"] = {
            "pair_count": int(np.count_nonzero(matched)),
            "cosine": group_weighted_mean(
                cosine, matched, pair_left, pair_right, targets
            ),
            "pearson": group_weighted_mean(
                correlation, matched, pair_left, pair_right, targets
            ),
            "normalized_euclidean": group_weighted_mean(
                distance, matched, pair_left, pair_right, targets
            ),
            "same_margin_quintile_cosine": group_weighted_mean(
                cosine, matched & same_quintile, pair_left, pair_right, targets
            ),
        }
    result["coordinate_rms_scale"] = scales.tolist()
    return result


def loo_score(raw: np.ndarray, labels: np.ndarray) -> tuple[float, list[float], float]:
    output_scores: list[float] = []
    eligible_total = 0
    for output in range(23):
        discovery_mask = labels != output
        held_mask = labels == output
        require(np.any(held_mask), f"FA_LOO_GROUP_EMPTY:{output}")
        _, discovery_unit, scale = normalize_rows(raw[discovery_mask])
        consensus = np.mean(discovery_unit, axis=0)
        consensus_norm = float(np.linalg.norm(consensus))
        require(consensus_norm > EPS, f"FA_LOO_CONSENSUS_ZERO:{output}")
        consensus /= consensus_norm
        held_scaled = raw[held_mask] / scale
        held_norms = np.linalg.norm(held_scaled, axis=1)
        eligible = held_norms > EPS
        require(np.any(eligible), f"FA_LOO_HELDOUT_DIRECTION_EMPTY:{output}")
        held_unit = held_scaled[eligible] / held_norms[eligible, None]
        output_scores.append(float(np.mean(held_unit @ consensus)))
        eligible_total += int(np.count_nonzero(eligible))
    return float(np.mean(output_scores)), output_scores, eligible_total / len(raw)


def null_summaries(
    raw: np.ndarray,
    labels: np.ndarray,
    inputs: np.ndarray,
    baseline_margin: np.ndarray,
    rng: np.random.Generator,
    observed_score: float,
) -> dict[str, Any]:
    coordinate_null = np.empty(NULL_REPLICATES, dtype=np.float64)
    output_null = np.empty(NULL_REPLICATES, dtype=np.float64)
    input_shared_left = np.empty(NULL_REPLICATES, dtype=np.float64)
    input_shared_right = np.empty(NULL_REPLICATES, dtype=np.float64)
    input_commuted = np.empty(NULL_REPLICATES, dtype=np.float64)

    scaled, unit, _ = normalize_rows(raw)
    pair_left, pair_right, cosine, _, _ = pair_arrays(scaled, unit)
    for replicate in range(NULL_REPLICATES):
        shuffled = np.empty_like(raw)
        for coordinate in range(raw.shape[1]):
            shuffled[:, coordinate] = raw[rng.permutation(len(raw)), coordinate]
        coordinate_null[replicate] = loo_score(shuffled, labels)[0]

        permuted_labels = labels[rng.permutation(len(labels))]
        output_null[replicate] = loo_score(raw, permuted_labels)[0]

        permuted_inputs = inputs[rng.permutation(len(inputs))]
        masks = class_masks(permuted_inputs, labels)
        input_shared_left[replicate] = group_weighted_mean(
            cosine, masks["shared_left"], pair_left, pair_right, labels
        )
        input_shared_right[replicate] = group_weighted_mean(
            cosine, masks["shared_right"], pair_left, pair_right, labels
        )
        input_commuted[replicate] = group_weighted_mean(
            cosine, masks["commuted"], pair_left, pair_right, labels
        )

    def summarize(values: np.ndarray) -> dict[str, Any]:
        return {
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "values": values.tolist(),
        }

    coordinate = summarize(coordinate_null)
    coordinate["observed_percentile"] = float(np.mean(coordinate_null < observed_score))
    coordinate["observed_minus_median"] = float(observed_score - coordinate["median"])
    return {
        "coordinate_independence": coordinate,
        "output_label_permutation": {
            **summarize(output_null),
            "observed_percentile": float(np.mean(output_null < observed_score)),
        },
        "input_identity_permutation": {
            "shared_left": summarize(input_shared_left),
            "shared_right": summarize(input_shared_right),
            "commuted": summarize(input_commuted),
        },
    }


def bootstrap_interval(values: list[float]) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples = data[rng.integers(0, len(data), size=(BOOTSTRAP_REPLICATES, len(data)))]
    medians = np.median(samples, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def phase_permutation_null(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [
            [row["phases"][phase]["loo"]["null_excess"] for phase in PHASES]
            for row in rows
        ],
        dtype=np.float64,
    )
    observed = float(np.median(values[:, 1] - values[:, 0]))
    permutations = np.asarray(list(__import__("itertools").permutations(range(4))), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(MASTER_SEED))
    null = np.empty(10000, dtype=np.float64)
    for index in range(len(null)):
        selected = permutations[rng.integers(0, len(permutations), size=len(values))]
        permuted = np.take_along_axis(values, selected, axis=1)
        null[index] = float(np.median(permuted[:, 1] - permuted[:, 0]))
    return {
        "observed_median_formed_minus_pre": observed,
        "null_median": float(np.median(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "observed_percentile": float(np.mean(null < observed)),
        "values": null.tolist(),
    }


def analyze(
    projection_root: Path,
    source_root: Path,
    output: Path,
    fingerprint_output: Path,
) -> dict[str, Any]:
    archive = read_json(projection_root / "ARCHIVE_MANIFEST.json")
    require(archive["status"] == "PASS" and archive["entry_count"] == 13, "FA_ARCHIVE_INVALID")
    fingerprint_arrays: dict[str, np.ndarray] = {}
    result_rows: list[dict[str, Any]] = []
    eligibility_failures: list[dict[str, str]] = []
    source_manifest: list[dict[str, Any]] = []

    for run_index, entry in enumerate(archive["entries"]):
        entry_id = str(entry["entry_id"])
        run_root = projection_root / entry_id
        try:
            manifest_path = run_root / "GFG_MANIFEST.json"
            validation_path = run_root / "GFG_VALIDATION.json"
            require(sha256_file(manifest_path) == entry["manifest_sha256"], "FA_MANIFEST_HASH")
            require(sha256_file(validation_path) == entry["validation_sha256"], "FA_VALIDATION_HASH")
            manifest = read_json(manifest_path)
            require(
                sha256_file(run_root / manifest["database"]) == manifest["database_sha256"],
                "FA_DATABASE_HASH",
            )
            require(read_json(validation_path)["status"] == "PASS", "FA_VALIDATION_STATUS")
            blocks = graph_blocks(run_root)
            block_map = {stage: (step, block) for stage, step, block in blocks}
            target_step, target_block = block_map["target_mapping_and_query_scope"]
            stored_targets, target_row = load_projection_tensor(
                run_root, target_block, "derived_validation_targets"
            )
            targets = stored_targets[:, -1].astype(np.int64)
            origins = source_origins(run_root)
            train_inputs, train_input_path = load_source_tensor(source_root, origins["training_inputs"])
            train_targets, train_target_path = load_source_tensor(source_root, origins["training_targets"])
            validation_inputs, validation_input_path = load_source_tensor(source_root, origins["validation_inputs"])
            reconstructed_targets, mapping_certificate = recover_cyclic_target_mapping(
                train_inputs, train_targets, validation_inputs
            )
            require(np.array_equal(reconstructed_targets, stored_targets), "FA_TARGET_RECONSTRUCTION_MISMATCH")
            require(len(train_inputs) == 317 and len(validation_inputs) == 212, "FA_ROW_COUNT")
            require(len(np.unique(validation_inputs, axis=0)) == 212, "FA_VALIDATION_DUPLICATE")
            require(set(targets.tolist()) == set(range(23)), "FA_TARGET_GROUP_COVERAGE")

            phase_rows: dict[str, Any] = {}
            for phase_index, phase in enumerate(PHASES):
                stage = f"native_inference:{phase}"
                require(stage in block_map, f"FA_PHASE_MISSING:{phase}")
                optimizer_step, block = block_map[stage]
                baseline, baseline_row = load_projection_tensor(run_root, block, "native_inference_logits")
                singles = {
                    component: load_projection_tensor(
                        run_root, block, "single_component_gate_logits", [component]
                    )[0]
                    for component in COMPONENTS
                }
                pairs = {
                    pair: load_projection_tensor(
                        run_root, block, "pair_component_gate_logits", list(pair)
                    )[0]
                    for pair in COMPONENT_PAIRS
                }
                require(baseline.shape == (212, 24), "FA_BASELINE_SHAPE")
                require(all(value.shape == (212, 24) for value in singles.values()), "FA_SINGLE_SHAPE")
                require(all(value.shape == (212, 24) for value in pairs.values()), "FA_PAIR_SHAPE")

                base_margin = true_margin(baseline, targets)
                single_effect = np.stack(
                    [base_margin - true_margin(singles[component], targets) for component in COMPONENTS],
                    axis=1,
                )
                pair_effect = np.stack(
                    [base_margin - true_margin(pairs[pair], targets) for pair in COMPONENT_PAIRS],
                    axis=1,
                )
                raw = np.concatenate([single_effect, pair_effect], axis=1)
                interactions = np.stack(
                    [
                        pair_effect[:, index]
                        - single_effect[:, COMPONENTS.index(pair[0])]
                        - single_effect[:, COMPONENTS.index(pair[1])]
                        for index, pair in enumerate(COMPONENT_PAIRS)
                    ],
                    axis=1,
                )
                scaled, unit, scales = normalize_rows(raw)
                _, single_unit, _ = normalize_rows(single_effect)
                _, interaction_unit, _ = normalize_rows(interactions)
                loo_observed, loo_outputs, eligible_fraction = loo_score(raw, targets)
                rng = np.random.Generator(
                    np.random.PCG64(MASTER_SEED + run_index * 100 + phase_index)
                )
                nulls = null_summaries(
                    raw, targets, validation_inputs, base_margin, rng, loo_observed
                )
                class_values = class_summary(raw, validation_inputs, targets, base_margin)
                single_class = class_summary(single_effect, validation_inputs, targets, base_margin)
                interaction_class = class_summary(interactions, validation_inputs, targets, base_margin)
                single_sq = float(np.sum(single_effect**2))
                pair_sq = float(np.sum(pair_effect**2))
                total_sq = single_sq + pair_sq
                single_share = single_sq / total_sq if total_sq > EPS else 0.0
                dominance = (
                    "single-component-dominated"
                    if single_share >= 0.75
                    else "pair-intervention-dominated"
                    if single_share <= 0.25
                    else "distributed"
                )
                predictions = np.argmax(baseline, axis=1)
                single_flip_counts = {
                    component: int(
                        np.count_nonzero(np.argmax(singles[component], axis=1) != predictions)
                    )
                    for component in COMPONENTS
                }
                pair_flip_counts = {
                    "+".join(pair): int(
                        np.count_nonzero(np.argmax(pairs[pair], axis=1) != predictions)
                    )
                    for pair in COMPONENT_PAIRS
                }
                phase_rows[phase] = {
                    "optimizer_step": optimizer_step,
                    "baseline_accuracy": float(np.mean(predictions == targets)),
                    "baseline_margin_mean": float(np.mean(base_margin)),
                    "baseline_margin_median": float(np.median(base_margin)),
                    "raw_fingerprint_rms": float(np.sqrt(np.mean(raw**2))),
                    "single_gate_prediction_flip_counts": single_flip_counts,
                    "pair_gate_prediction_flip_counts": pair_flip_counts,
                    "direction_eligible_fraction": eligible_fraction,
                    "loo": {
                        "observed_score": loo_observed,
                        "heldout_output_scores": loo_outputs,
                        "null_excess": nulls["coordinate_independence"]["observed_minus_median"],
                    },
                    "classes": class_values,
                    "single_only_classes": single_class,
                    "interaction_only_classes": interaction_class,
                    "nulls": nulls,
                    "decomposition": {
                        "single_squared_magnitude_share": single_share,
                        "pair_squared_magnitude_share": 1.0 - single_share,
                        "classification": dominance,
                        "single_rms": float(np.sqrt(np.mean(single_effect**2))),
                        "pair_rms": float(np.sqrt(np.mean(pair_effect**2))),
                        "interaction_rms": float(np.sqrt(np.mean(interactions**2))),
                    },
                    "tensor_hashes": {
                        "baseline": baseline_row["payload"]["raw_tensor_sha256"],
                    },
                }
                prefix = f"{entry_id}__{phase}"
                fingerprint_arrays[f"{prefix}__raw"] = raw
                fingerprint_arrays[f"{prefix}__scaled"] = scaled
                fingerprint_arrays[f"{prefix}__unit"] = unit
                fingerprint_arrays[f"{prefix}__single_effect"] = single_effect
                fingerprint_arrays[f"{prefix}__single_unit"] = single_unit
                fingerprint_arrays[f"{prefix}__pair_effect"] = pair_effect
                fingerprint_arrays[f"{prefix}__interaction"] = interactions
                fingerprint_arrays[f"{prefix}__interaction_unit"] = interaction_unit
                fingerprint_arrays[f"{prefix}__baseline_margin"] = base_margin

            result_rows.append(
                {
                    "entry_id": entry_id,
                    "source_bundle_id": origins["training_inputs"]["source_bundle_id"],
                    "target_mapping_certificate": mapping_certificate,
                    "validation_target_sha256": array_sha256(stored_targets),
                    "validation_input_sha256": array_sha256(validation_inputs),
                    "phases": phase_rows,
                }
            )
            source_manifest.append(
                {
                    "entry_id": entry_id,
                    "projection_manifest_sha256": entry["manifest_sha256"],
                    "projection_validation_sha256": entry["validation_sha256"],
                    "training_input_path": str(train_input_path),
                    "training_input_raw_sha256": array_sha256(train_inputs),
                    "training_target_path": str(train_target_path),
                    "training_target_raw_sha256": array_sha256(train_targets),
                    "validation_input_path": str(validation_input_path),
                    "validation_input_raw_sha256": array_sha256(validation_inputs),
                }
            )
        except Exception as error:
            eligibility_failures.append({"entry_id": entry_id, "error": str(error)})

    require(len(result_rows) >= 10, "FA_INSUFFICIENT_ELIGIBLE_RUNS")
    np.savez_compressed(fingerprint_output, **fingerprint_arrays)

    threshold = math.ceil(0.69 * len(result_rows))
    formed_rows = [row["phases"]["formed"] for row in result_rows]
    formed_excess = [row["loo"]["null_excess"] for row in formed_rows]
    formed_percentile_pass = sum(
        row["nulls"]["coordinate_independence"]["observed_percentile"] >= 0.95
        for row in formed_rows
    )
    equivalence_pass = 0
    same_output_gap_count = 0
    shared_gap_count = 0
    for row in formed_rows:
        classes = row["classes"]
        main = classes["main_different_output_unmatched"]["cosine"]
        same = classes["same_output_different_instance"]["cosine"]
        shared_left = classes["shared_left"]["cosine"]
        shared_right = classes["shared_right"]["cosine"]
        equivalent = (
            main >= same - 0.10
            and main >= shared_left - 0.10
            and main >= shared_right - 0.10
        )
        equivalence_pass += int(equivalent)
        same_output_gap_count += int(same - main > 0.10)
        shared_gap_count += int(max(shared_left, shared_right) - main > 0.10)
    formed_minus_pre = [
        row["phases"]["formed"]["loo"]["null_excess"]
        - row["phases"]["pre_formation"]["loo"]["null_excess"]
        for row in result_rows
    ]
    positive_phase_count = sum(value > 0.0 for value in formed_minus_pre)
    robustness_opposite = 0
    for row in formed_rows:
        classes = row["classes"]
        main = classes["main_different_output_unmatched"]
        same = classes["same_output_different_instance"]
        if main["cosine"] >= same["cosine"] - 0.10:
            if main["pearson"] < same["pearson"] - 0.10 or main["normalized_euclidean"] > same["normalized_euclidean"] + 0.20:
                robustness_opposite += 1

    commonality = (
        float(np.median(formed_excess)) >= 0.05
        and formed_percentile_pass >= threshold
        and equivalence_pass >= threshold
    )
    historical = (
        float(np.median(formed_minus_pre)) >= 0.05
        and positive_phase_count >= threshold
    )
    eligibility = all(row["direction_eligible_fraction"] >= 0.90 for row in formed_rows)
    robustness = robustness_opposite == 0
    if commonality and historical and eligibility and robustness:
        verdict = "SINGLE_RELATION_FUNCTIONAL_INVARIANCE_SUPPORTED"
    elif commonality:
        if not historical:
            verdict = "STATIC_ARCHITECTURAL_PATTERN_ONLY"
        else:
            verdict = "PARTIAL_FUNCTIONAL_INVARIANCE_SUPPORTED"
    elif same_output_gap_count >= threshold:
        verdict = "OUTPUT_CONDITIONED_CONVERGENCE_ONLY"
    elif shared_gap_count >= threshold:
        verdict = "INPUT_STRUCTURE_DOMINATED"
    elif float(np.median(formed_excess)) <= 0.0 and formed_percentile_pass < math.ceil(len(result_rows) / 2):
        verdict = "NO_FUNCTIONAL_INVARIANCE"
    else:
        verdict = "INCONCLUSIVE"

    phase_summary: dict[str, Any] = {}
    for phase in PHASES:
        phase_summary[phase] = {}
        for key, getter in {
            "loo_score": lambda r: r["phases"][phase]["loo"]["observed_score"],
            "null_excess": lambda r: r["phases"][phase]["loo"]["null_excess"],
            "baseline_accuracy": lambda r: r["phases"][phase]["baseline_accuracy"],
            "baseline_margin_mean": lambda r: r["phases"][phase]["baseline_margin_mean"],
            "raw_fingerprint_rms": lambda r: r["phases"][phase]["raw_fingerprint_rms"],
        }.items():
            values = [float(getter(row)) for row in result_rows]
            phase_summary[phase][key] = {
                "run_values": values,
                "median": float(np.median(values)),
                "bootstrap_95_median": bootstrap_interval(values),
            }

    result: dict[str, Any] = {
        "schema": "fa-e01-frozen-evidence-single-relation-functional-invariance-v1",
        "status": "PASS",
        "protocol_commit": "88b10b2",
        "repository_base": "36dab5ce347dbbdac157ef23205f556606d18294",
        "projection_root": str(projection_root),
        "source_root": str(source_root),
        "eligible_run_count": len(result_rows),
        "eligibility_failures": eligibility_failures,
        "null_replicates_per_run_phase": NULL_REPLICATES,
        "fingerprint_archive": {
            "path": fingerprint_output.name,
            "sha256": sha256_file(fingerprint_output),
            "array_count": len(fingerprint_arrays),
        },
        "source_manifest": source_manifest,
        "run_rows": result_rows,
        "phase_summary": phase_summary,
        "historical_contrasts": {
            "formed_minus_pre": formed_minus_pre,
            "formed_minus_pre_median": float(np.median(formed_minus_pre)),
            "formed_minus_pre_positive_run_count": positive_phase_count,
            "decline_minus_formed": [
                row["phases"]["decline"]["loo"]["null_excess"]
                - row["phases"]["formed"]["loo"]["null_excess"]
                for row in result_rows
            ],
            "recovered_minus_decline": [
                row["phases"]["recovered"]["loo"]["null_excess"]
                - row["phases"]["decline"]["loo"]["null_excess"]
                for row in result_rows
            ],
            "phase_label_permutation": phase_permutation_null(result_rows),
        },
        "verdict_inputs": {
            "run_replication_threshold": threshold,
            "formed_null_excess_median": float(np.median(formed_excess)),
            "formed_primary_null_pass_run_count": formed_percentile_pass,
            "formed_cross_output_equivalence_run_count": equivalence_pass,
            "formed_minus_pre_median": float(np.median(formed_minus_pre)),
            "formed_minus_pre_positive_run_count": positive_phase_count,
            "formed_direction_eligibility_all_at_least_0_90": eligibility,
            "robustness_opposite_run_count": robustness_opposite,
            "same_output_gap_over_0_10_run_count": same_output_gap_count,
            "shared_operand_gap_over_0_10_run_count": shared_gap_count,
        },
        "verdict": verdict,
        "claim_ceiling": (
            "Existing frozen nanoGPT evidence supports single-relation causal functional "
            "invariance across concrete-input and output variation."
        ),
        "mandatory_limitation": (
            "The existing task does not test across-relation selectivity because only one "
            "underlying operation is present."
        ),
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path(__file__).with_name("FA_E01_FINGERPRINTS.npz"),
    )
    args = parser.parse_args()
    result = analyze(
        args.projection_root.resolve(),
        args.source_root.resolve(),
        args.output.resolve(),
        args.fingerprints.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "eligible_run_count": result["eligible_run_count"],
                "verdict": result["verdict"],
                "formed_null_excess_median": result["verdict_inputs"]["formed_null_excess_median"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
