from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from experiments.database_lineage.src.metrics import write_json


RUNTIME = Path("experiments/database_lineage/runtime")
BASELINE = RUNTIME / "determinism_baseline"
ARTIFACT = Path("experiments/database_lineage/artifacts/tpch_first_run.json")


def main() -> int:
    BASELINE.mkdir(parents=True, exist_ok=True)
    summary = {
        "core_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    }
    for query in (1, 3, 6, 10):
        metric_path = RUNTIME / f"tpch_result_sf_0_01_q{query}.json"
        lineage_path = RUNTIME / f"core_lineage_sf_0_01_q{query}.json"
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        lineage_bytes = lineage_path.read_bytes()
        (BASELINE / metric_path.name).write_bytes(metric_path.read_bytes())
        (BASELINE / lineage_path.name).write_bytes(lineage_bytes)
        summary[f"q{query}"] = {
            "snapshot_id": metrics["snapshot"]["snapshot_id"],
            "binding_count": metrics["snapshot"]["binding_count"],
            "csv_sha256": metrics["output"]["csv_sha256"],
            "json_sha256": metrics["output"]["json_sha256"],
            "backward_lineage_file_sha256": hashlib.sha256(lineage_bytes).hexdigest(),
            "backward_lineage_file_bytes": len(lineage_bytes),
        }
    write_json(ARTIFACT, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
