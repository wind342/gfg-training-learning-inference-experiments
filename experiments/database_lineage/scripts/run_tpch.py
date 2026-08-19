from __future__ import annotations

import argparse
from pathlib import Path

from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.run_tpch_workload import run_query
from experiments.database_lineage.src.tpch_loader import scale_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime", type=Path, default=Path("experiments/database_lineage/runtime")
    )
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--query", type=int, action="append", required=True)
    args = parser.parse_args()
    name = scale_name(args.scale)
    database = args.runtime / f"tpch_sf_{name}.duckdb"
    results = {}
    failed = False
    for query in args.query:
        metrics, _lineage = run_query(
            database,
            scale_factor=args.scale,
            query_number=query,
            lineage_path=args.runtime / f"core_lineage_sf_{name}_q{query}.json",
            forward_lineage_path=args.runtime
            / f"core_forward_lineage_sf_{name}_q{query}.json",
        )
        results[f"q{query}"] = metrics
        failed = failed or not (
            metrics["output_exact_match_duckdb"]
            and metrics["official_answer_exact_match"]
            and metrics["output_orthogonality"]["csv_byte_identical"]
            and metrics["output_orthogonality"]["json_byte_identical"]
            and metrics["snapshot"]["validated"]
            and metrics["direct_structure_audit"]["passed"]
        )
        write_json(args.runtime / f"tpch_result_sf_{name}_q{query}.json", metrics)
    write_json(args.runtime / f"tpch_results_sf_{name}.json", results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
