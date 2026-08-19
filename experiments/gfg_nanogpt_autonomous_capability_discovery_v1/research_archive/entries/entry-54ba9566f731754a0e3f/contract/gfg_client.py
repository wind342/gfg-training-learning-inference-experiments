from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zlib
import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class GFG:
    def __init__(self, root=None, max_step=None):
        self.root = Path(root or os.environ.get("NANOGPT_GFG_ROOT", "/evidence"))
        self.max_step = max_step
        self.database = self.root / "participant_gfg.sqlite3"
        self.connection = sqlite3.connect(
            "file:" + self.database.as_posix() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.log_path = Path(os.environ.get(
            "GFG_QUERY_LOG", "submission/query_log.jsonl"))

    def _log(self, operation, arguments, count):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"operation": operation, "arguments": arguments, "count": count}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(row) + "\n")

    def _blocks(self, min_step=None, max_step=None, stage=None):
        upper = self.max_step if max_step is None else max_step
        clauses, values = [], []
        if min_step is not None:
            clauses.append("optimizer_step>=?"); values.append(min_step)
        if upper is not None:
            clauses.append("optimizer_step<=?"); values.append(upper)
        if stage is not None:
            clauses.append("stage=?"); values.append(stage)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        for row in self.connection.execute(
                "SELECT * FROM graph_blocks" + where +
                " ORDER BY block_ordinal", values):
            yield row, json.loads(zlib.decompress(
                row["payload_zlib"]).decode("utf-8"))

    def summary(self):
        where, values = ("", ())
        if self.max_step is not None:
            where, values = (" WHERE optimizer_step<=?", (self.max_step,))
        row = self.connection.execute(
            """SELECT COUNT(*) blocks,COALESCE(SUM(object_count),0) objects,
               COALESCE(SUM(occurrence_count),0) occurrences,
               COALESCE(SUM(fact_count),0) facts,
               COALESCE(SUM(explicit_edge_count),0) explicit_edges
               FROM graph_blocks""" + where, values).fetchone()
        result = dict(row)
        self._log("summary", {"max_step": self.max_step}, 1)
        return result

    def evaluations(self):
        where, values = ("", ())
        if self.max_step is not None:
            where, values = (" WHERE optimizer_step<=?", (self.max_step,))
        rows = [dict(row) for row in self.connection.execute(
            "SELECT * FROM evaluations" + where +
            " ORDER BY optimizer_step", values)]
        self._log("evaluations", {"max_step": self.max_step}, len(rows))
        return rows

    def objects(self, role=None, name_contains=None, min_step=None,
                max_step=None, materialized=None):
        rows = []
        for _header, block in self._blocks(min_step, max_step):
            for row in block["objects"]:
                if role is not None and row["role"] != role: continue
                if name_contains is not None and name_contains not in row["name"]: continue
                if materialized is not None and bool(row["materialized"]) != materialized: continue
                rows.append(row)
        self._log("objects", {
            "role": role, "name_contains": name_contains,
            "min_step": min_step, "max_step": max_step,
            "materialized": materialized}, len(rows))
        return rows

    def occurrences(self, occurrence_type=None, min_step=None, max_step=None):
        rows = []
        for _header, block in self._blocks(min_step, max_step):
            for row in block["occurrences"]:
                if occurrence_type is None or row["occurrence_type"] == occurrence_type:
                    rows.append(row)
        self._log("occurrences", {
            "occurrence_type": occurrence_type,
            "min_step": min_step, "max_step": max_step}, len(rows))
        return rows

    def fact_blocks(self, min_step=None, max_step=None):
        rows = [row for _header, block in self._blocks(min_step, max_step)
                for row in block["fact_blocks"]]
        self._log("fact_blocks", {
            "min_step": min_step, "max_step": max_step}, len(rows))
        return rows

    def edges(self, relation_type=None, min_step=None, max_step=None):
        rows = []
        for _header, block in self._blocks(min_step, max_step):
            for row in block["edges"]:
                if relation_type is None or row["relation_type"] == relation_type:
                    rows.append(row)
        self._log("edges", {
            "relation_type": relation_type,
            "min_step": min_step, "max_step": max_step}, len(rows))
        return rows

    def load_tensor(self, object_row):
        if not object_row["materialized"]:
            raise ValueError("TENSOR_REQUIRES_DETERMINISTIC_REPLAY")
        prefix = "objects://"
        if not object_row["locator"].startswith(prefix):
            raise ValueError("NOT_A_MATERIALIZED_TENSOR")
        value = np.load(self.root / "tensor-objects" /
                        object_row["locator"][len(prefix):],
                        allow_pickle=False)
        digest = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        if digest != object_row["content_sha256"]:
            raise ValueError("TENSOR_CONTENT_HASH_MISMATCH")
        self._log("load_tensor", {
            "object_id": object_row["object_id"]}, int(value.size))
        return value

    def close(self):
        self.connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "summary", "evaluations", "objects", "occurrences",
        "fact-blocks", "edges"])
    parser.add_argument("--root")
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--role")
    parser.add_argument("--name-contains")
    parser.add_argument("--occurrence-type")
    parser.add_argument("--relation-type")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    gfg = GFG(args.root, args.max_step)
    try:
        if args.command == "summary": result = gfg.summary()
        elif args.command == "evaluations": result = gfg.evaluations()
        elif args.command == "objects": result = gfg.objects(
            args.role, args.name_contains, args.min_step, args.max_step)
        elif args.command == "occurrences": result = gfg.occurrences(
            args.occurrence_type, args.min_step, args.max_step)
        elif args.command == "fact-blocks": result = gfg.fact_blocks(
            args.min_step, args.max_step)
        else: result = gfg.edges(
            args.relation_type, args.min_step, args.max_step)
        if isinstance(result, list): result = result[:args.limit]
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    finally:
        gfg.close()

if __name__ == "__main__":
    main()
