from __future__ import annotations

import argparse
from pathlib import Path

from experiments.opentelemetry_projection.src.run_experiment import (
    refresh_environment_and_manifest,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("experiments/database_lineage/runtime/tpch_sf_0_01.duckdb"),
    )
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument("--refresh-metadata-only", action="store_true")
    args = parser.parse_args()
    if args.refresh_metadata_only:
        refresh_environment_and_manifest()
        print("OpenTelemetry projection metadata refreshed")
        return
    metrics = run(
        database_path=args.database_path,
        include_formal=not args.skip_formal,
    )
    print(
        "OpenTelemetry projection experiment complete: "
        f"{metrics['negative_controls_fail_closed']} negative controls failed closed"
    )


if __name__ == "__main__":
    main()
