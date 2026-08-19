from __future__ import annotations

import argparse
from pathlib import Path

from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.tpch_loader import generate_database, scale_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime", type=Path, default=Path("experiments/database_lineage/runtime")
    )
    parser.add_argument("--scale", type=float, action="append", default=[])
    args = parser.parse_args()
    scales = args.scale or [0.01, 0.1]
    manifests = {}
    for scale in scales:
        name = scale_name(scale)
        manifests[name] = generate_database(
            scale,
            args.runtime / f"tpch_sf_{name}.duckdb",
            args.runtime / f"tpch_sf_{name}_csv",
        )
    write_json(args.runtime / "tpch_generation_manifest.json", manifests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
