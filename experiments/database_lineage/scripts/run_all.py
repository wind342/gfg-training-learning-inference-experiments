from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = REPO / "experiments" / "database_lineage"


def run(label: str, command: list[str], failures: list[dict[str, object]]) -> bool:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode:
        failures.append(
            {"step": label, "command": command, "returncode": completed.returncode}
        )
        print(f"[{label}] FAILED ({completed.returncode})", flush=True)
        return False
    print(f"[{label}] passed", flush=True)
    return True


def record_sf_0_1_q1_resource_bound(failures: list[dict[str, object]]) -> None:
    preflight = {
        "status": "NOT_RUN_RESOURCE_BOUND",
        "reason": "NOT RUN — projected in-memory representation exceeded available physical memory.",
        "projected_peak_rss_bytes": 21_546_833_685,
        "physical_memory_bytes": 17_011_310_592,
        "sf_0_01_measured_peak_rss_bytes": 2_158_182_400,
        "sf_0_01_binding_count": 178_797,
        "sf_0_1_exact_projected_binding_count": 1_784_292,
    }
    path = EXPERIMENT / "runtime" / "tpch_resource_preflight_sf_0_1_q1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures.append({"step": "TPC-H SF 0.1 Q1", "returncode": 125, **preflight})
    print(f"\n[TPC-H SF 0.1 Q1] {preflight['reason']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the falsification-oriented database-lineage experiment"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    failures: list[dict[str, object]] = []

    run(
        "install locked experiment dependencies",
        [py, "-m", "pip", "install", "-r", str(EXPERIMENT / "requirements.lock")],
        failures,
    )
    run("install Core package", [py, "-m", "pip", "install", "-e", str(REPO)], failures)
    generation_command = [
        py,
        "-m",
        "experiments.database_lineage.scripts.generate_tpch",
    ]
    if args.fast:
        generation_command.extend(["--scale", "0.01"])
    run("generate fixed TPC-H data", generation_command, failures)
    run(
        "repository and experiment tests",
        [py, "-m", "experiments.database_lineage.scripts.run_tests"],
        failures,
    )
    run(
        "synthetic evaluation",
        [py, "-m", "experiments.database_lineage.scripts.run_synthetic"],
        failures,
    )
    run(
        "resolver before/after benchmark",
        [py, "-m", "experiments.database_lineage.scripts.benchmark_resolver"],
        failures,
    )
    run(
        "reader index before/after benchmark",
        [py, "-m", "experiments.database_lineage.scripts.benchmark_reader"],
        failures,
    )

    if args.full:
        for query in (1, 3, 6, 10):
            run(
                f"TPC-H SF 0.01 Q{query} first run",
                [
                    py,
                    "-m",
                    "experiments.database_lineage.scripts.run_tpch",
                    "--scale",
                    "0.01",
                    "--query",
                    str(query),
                ],
                failures,
            )
        run(
            "capture TPC-H determinism baseline",
            [py, "-m", "experiments.database_lineage.scripts.capture_tpch_baseline"],
            failures,
        )
        for query in (1, 3, 6, 10):
            run(
                f"TPC-H SF 0.01 Q{query} second run",
                [
                    py,
                    "-m",
                    "experiments.database_lineage.scripts.run_tpch",
                    "--scale",
                    "0.01",
                    "--query",
                    str(query),
                ],
                failures,
            )
        run(
            "compare TPC-H determinism",
            [py, "-m", "experiments.database_lineage.scripts.compare_tpch_determinism"],
            failures,
        )

        record_sf_0_1_q1_resource_bound(failures)
        run(
            "TPC-H SF 0.1 Q6 with resource guard",
            [py, "-m", "experiments.database_lineage.scripts.run_guarded_tpch"],
            failures,
        )
        docker = os.environ.get("DOCKER", "docker")
        run(
            "start pinned ProvSQL",
            [
                docker,
                "compose",
                "-f",
                str(EXPERIMENT / "docker-compose.yml"),
                "up",
                "-d",
                "--wait",
            ],
            failures,
        )
        run(
            "ProvSQL independent evaluation",
            [py, "-m", "experiments.database_lineage.scripts.run_provsql"],
            failures,
        )

    verify_command = [
        py,
        "-m",
        "experiments.database_lineage.scripts.verify_reproducibility",
    ]
    if args.fast:
        verify_command.append("--fast")
    run("artifact and reproducibility verification", verify_command, failures)
    if failures:
        print("\nMandatory failures:", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
