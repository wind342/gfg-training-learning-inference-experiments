from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np


def _connection(bundle: Path) -> sqlite3.Connection:
    database = bundle / "support_gfg.sqlite3"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _decode(blob: bytes) -> dict[str, object]:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def summary(bundle: Path) -> dict[str, object]:
    connection = _connection(bundle)
    try:
        counts = {
            "blocks": connection.execute("SELECT COUNT(*) FROM graph_blocks").fetchone()[0],
            "checkpoints": connection.execute("SELECT COUNT(*) FROM checkpoint_index").fetchone()[0],
            "facts": connection.execute("SELECT COALESCE(SUM(fact_count),0) FROM graph_blocks").fetchone()[0],
            "objects": connection.execute("SELECT COALESCE(SUM(object_count),0) FROM graph_blocks").fetchone()[0],
            "occurrences": connection.execute("SELECT COALESCE(SUM(occurrence_count),0) FROM graph_blocks").fetchone()[0],
            "origins": connection.execute("SELECT COUNT(*) FROM origin_catalog").fetchone()[0],
            "relations": connection.execute("SELECT COALESCE(SUM(relation_count),0) FROM graph_blocks").fetchone()[0],
        }
        historical = dict(
            connection.execute(
                """
                SELECT
                  SUM(historical_raw_logits_exact) AS raw_exact,
                  SUM(historical_predictions_exact) AS prediction_exact,
                  SUM(historical_capability_exact) AS capability_exact,
                  MAX(historical_max_abs_logit_error) AS max_abs_logit_error
                FROM checkpoint_index
                """
            ).fetchone()
        )
        return {"bundle": bundle.name, "counts": counts, "historical_runtime_audit": historical}
    finally:
        connection.close()


def step_graph(bundle: Path, optimizer_step: int) -> dict[str, object]:
    connection = _connection(bundle)
    try:
        blocks = [
            {
                "block_id": row["block_id"],
                "stage": row["stage"],
                "payload": _decode(row["payload_zlib"]),
            }
            for row in connection.execute(
                "SELECT * FROM graph_blocks WHERE optimizer_step=? ORDER BY block_ordinal",
                (optimizer_step,),
            )
        ]
        checkpoint = connection.execute(
            "SELECT * FROM checkpoint_index WHERE optimizer_step=?", (optimizer_step,)
        ).fetchone()
        if checkpoint is None:
            raise RuntimeError("CSRG_QUERY_CHECKPOINT_NOT_FOUND")
        return {"blocks": blocks, "checkpoint": dict(checkpoint), "optimizer_step": optimizer_step}
    finally:
        connection.close()


def series(bundle: Path, role: str) -> dict[str, object]:
    connection = _connection(bundle)
    rows: list[dict[str, object]] = []
    try:
        for block_row in connection.execute(
            "SELECT optimizer_step,payload_zlib FROM graph_blocks ORDER BY block_ordinal"
        ):
            block = _decode(block_row["payload_zlib"])
            for value in block["objects"]:
                if value["role"] != role:
                    continue
                payload = value["payload"]
                row: dict[str, object] = {
                    "object_id": value["object_id"],
                    "optimizer_step": block_row["optimizer_step"],
                    "payload": payload,
                }
                locator = payload.get("locator")
                if locator:
                    array = np.load(bundle / locator, allow_pickle=False)
                    row["value"] = array.tolist()
                rows.append(row)
        return {"bundle": bundle.name, "role": role, "rows": rows}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary")
    step_parser = subparsers.add_parser("step")
    step_parser.add_argument("optimizer_step", type=int)
    series_parser = subparsers.add_parser("series")
    series_parser.add_argument(
        "role",
        choices=[
            "component_optimizer_loads",
            "component_target_group_necessity",
            "double_failure_slack",
            "effective_support",
            "pair_target_group_backup",
            "single_failure_slack",
            "support_turnover",
            "target_group_q10_margin",
        ],
    )
    arguments = parser.parse_args()
    bundle = arguments.bundle.resolve()
    if arguments.command == "summary":
        result = summary(bundle)
    elif arguments.command == "step":
        result = step_graph(bundle, arguments.optimizer_step)
    else:
        result = series(bundle, arguments.role)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
