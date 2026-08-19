from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from experiments.database_lineage.src.metrics import write_json


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = REPO / "experiments" / "database_lineage"
DECISION_PATH = EXPERIMENT / "artifacts" / "resource_bound_decision.json"
MONITOR_PATH = EXPERIMENT / "runtime" / "sf_0_1_q6_resource_monitor.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full SF0.1/Q6 workload with an external RSS guard"
    )
    parser.add_argument("--rss-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-available-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not 0.80 <= args.rss_fraction <= 0.85:
        parser.error("--rss-fraction must remain within the mandated 0.80-0.85 range")

    physical = psutil.virtual_memory().total
    rss_limit = int(physical * args.rss_fraction)
    command = [
        sys.executable,
        "-m",
        "experiments.database_lineage.scripts.run_tpch",
        "--scale",
        "0.1",
        "--query",
        "6",
    ]
    started = time.perf_counter()
    child = subprocess.Popen(command, cwd=REPO)
    process = psutil.Process(child.pid)
    samples: list[dict[str, int | float]] = []
    stop_reason = None
    max_rss = 0
    minimum_available = physical
    try:
        while child.poll() is None:
            memory = psutil.virtual_memory()
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            elapsed = time.perf_counter() - started
            max_rss = max(max_rss, rss)
            minimum_available = min(minimum_available, memory.available)
            samples.append(
                {
                    "elapsed_seconds": elapsed,
                    "process_rss_bytes": rss,
                    "system_available_bytes": memory.available,
                }
            )
            if rss >= rss_limit:
                stop_reason = (
                    "process_rss_reached_configured_fraction_of_physical_memory"
                )
            elif memory.available < args.minimum_available_bytes:
                stop_reason = "system_available_memory_fell_below_safety_threshold"
            if stop_reason:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
                break
            time.sleep(args.sample_seconds)
    finally:
        returncode = child.poll()
        if returncode is None:
            child.terminate()
            try:
                returncode = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                returncode = child.wait(timeout=30)

    elapsed = time.perf_counter() - started
    completed = returncode == 0 and stop_reason is None
    result = {
        "status": "COMPLETED"
        if completed
        else "RESOURCE_LIMITED"
        if stop_reason
        else "FAILED",
        "command": command,
        "returncode": returncode,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "physical_memory_bytes": physical,
        "rss_limit_fraction": args.rss_fraction,
        "rss_limit_bytes": rss_limit,
        "minimum_available_memory_limit_bytes": args.minimum_available_bytes,
        "max_process_rss_bytes": max_rss,
        "minimum_system_available_bytes": minimum_available,
        "preflight": {
            "total_input_rows": 600572,
            "qualifying_rows": 11618,
            "estimated_bindings": 623808,
            "estimated_peak_rss_bytes": 10084621030,
            "estimated_in_memory_snapshot_peak_bytes": 10084621030,
            "estimated_snapshot_build_seconds": 1476.874697,
            "estimated_validation_seconds": 720.63368,
            "basis": "SF0.01/Q6 actual 62,557 bindings, 1,011,310,592 peak RSS, 148.104626 s build, 72.266917 s validation",
        },
        "standards_reduced": False,
        "partial_validation_used": False,
        "sampling_validation_used": False,
        "samples": samples,
    }
    write_json(MONITOR_PATH, result)
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    decision["sf_0_1_q6"] = {
        key: value for key, value in result.items() if key != "samples"
    }
    write_json(DECISION_PATH, decision)
    return 0 if completed else 125 if stop_reason else returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
