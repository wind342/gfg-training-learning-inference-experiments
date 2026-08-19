from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import require


POSITIVE_SOURCE_INDICES = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    require(bool(records), "NO_FACTOR_RECORDS")
    records.sort(key=lambda value: (value["entry_id"], int(value["section_ordinal"]), value["record_id"]))
    return records


def feature_names(records: list[dict[str, Any]], block: str) -> list[str]:
    names = sorted(records[0]["features"][block]["numeric"])
    require(bool(names), f"EMPTY_FEATURE_BLOCK:{block}")
    for record in records:
        require(sorted(record["features"][block]["numeric"]) == names, f"FEATURE_SCHEMA_DRIFT:{block}")
        require(not record["features"][block]["categorical"], f"CATEGORICAL_FEATURES_NOT_ADMITTED:{block}")
    return names


def build_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    f1 = feature_names(records, "F1")
    f3 = feature_names(records, "F3")
    f5 = feature_names(records, "F5")
    require("update_l2_norm" in f3, "GLOBAL_UPDATE_L2_MISSING")

    block_names = {
        "B1_M1": [f"F1:{name}" for name in f1],
        "B2": [f"F1:{name}" for name in f1] + ["F3:update_l2_norm"],
        "M2": [f"F1:{name}" for name in f1] + [f"F3:{name}" for name in f3],
        "M3": [f"F1:{name}" for name in f1] + [f"F5:{name}" for name in f5],
        "M4": [f"F1:{name}" for name in f1] + [f"F3:{name}" for name in f3] + [f"F5:{name}" for name in f5],
    }

    def matrix(names: list[str]) -> np.ndarray:
        values: list[list[float]] = []
        for record in records:
            row: list[float] = []
            for qualified in names:
                block, name = qualified.split(":", 1)
                row.append(float(record["features"][block]["numeric"][name]))
            values.append(row)
        result = np.asarray(values, dtype=np.float64)
        require(np.all(np.isfinite(result)), "NONFINITE_ADMITTED_FEATURE")
        return result

    matrices = {name: matrix(names) for name, names in block_names.items()}
    displacement = np.asarray(
        [[record["response"]["displacement_curve"][index] for index in POSITIVE_SOURCE_INDICES] for record in records],
        dtype=np.float64,
    )
    margin0 = np.asarray([record["response"]["margin_curve"][1] for record in records], dtype=np.float64)
    entries = np.asarray([record["entry_id"] for record in records], dtype=object)
    require(np.all(np.isfinite(displacement)), "NONFINITE_RESPONSE_TARGET")
    require(np.all(np.isfinite(margin0)), "NONFINITE_MARGIN_ZERO")
    unique_entries = sorted(set(entries.tolist()))
    require(len(unique_entries) == 12, f"EXPECTED_12_RUNS_GOT:{len(unique_entries)}")
    counts = {entry: int(np.sum(entries == entry)) for entry in unique_entries}
    require(set(counts.values()) == {1272}, f"RUN_RECORD_COUNTS_UNEXPECTED:{counts}")

    prohibited_tokens = (
        "run_id",
        "entry_id",
        "optimizer_step",
        "phase",
        "alpha_",
        "response",
        "endpoint",
        "future",
    )
    violations = [name for names in block_names.values() for name in names if any(token in name.lower() for token in prohibited_tokens)]
    require(not violations, f"PROHIBITED_FEATURE_NAMES:{sorted(set(violations))}")

    return {
        "records": records,
        "feature_names": block_names,
        "matrices": matrices,
        "displacement": displacement,
        "margin0": margin0,
        "entries": entries,
        "unique_entries": unique_entries,
        "entry_counts": counts,
    }


def j_jk_predictions(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    from .model import POSITIVE_ALPHAS

    curves = np.asarray([record["response"]["margin_curve"] for record in records], dtype=np.float64)
    minus = curves[:, 0]
    zero = curves[:, 1]
    plus = curves[:, 2]
    step = 0.125
    derivative = (plus - minus) / (2.0 * step)
    curvature = (plus - 2.0 * zero + minus) / (step * step)
    j = derivative[:, None] * POSITIVE_ALPHAS[None, :]
    jk = j + 0.5 * curvature[:, None] * np.square(POSITIVE_ALPHAS[None, :])
    return j, jk


__all__ = ["build_dataset", "j_jk_predictions", "load_records"]
