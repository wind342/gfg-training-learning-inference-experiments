from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def find_node() -> Path:
    explicit = os.environ.get("SOURCE_MAP_NODE")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    discovered = shutil.which("node")
    if discovered:
        return Path(discovered)
    user_profile = Path(os.environ.get("USERPROFILE", ""))
    bundled = user_profile / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
    if bundled.is_file():
        return bundled
    raise RuntimeError("NODE_RUNTIME_UNAVAILABLE")


def node_version(node: Path | None = None) -> str:
    return subprocess.check_output([str(node or find_node()), "--version"], text=True).strip()


def _run(script: str, *args: Path | str) -> None:
    rendered = [str(arg.resolve()) if isinstance(arg, Path) else str(arg) for arg in args]
    result = subprocess.run(
        [str(find_node()), str(EXPERIMENT_ROOT / "src" / script), *rendered],
        cwd=EXPERIMENT_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"NODE_TOOL_FAILED:{script}:{result.stderr.strip()}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def capture_native_map(result, output_path: Path) -> dict:
    input_path = output_path.with_suffix(".capture-input.json")
    receipts = [row for row in result.receipts]
    source_roots = {row.get("source_root") for row in receipts}
    if len(source_roots) != 1:
        raise ValueError("SOURCE_ROOT_CONFLICT")
    write_json(input_path, {
        "generated_file": result.generated_artifact,
        "source_root": next(iter(source_roots)),
        "receipts": receipts,
    })
    _run("native_source_map_capture.mjs", input_path, output_path)
    return json.loads(output_path.read_text(encoding="utf-8"))


def consumer_decode(map_path: Path, output_path: Path, *, base_url: str) -> dict:
    _run("source_map_consumer_oracle.mjs", "decode", map_path, base_url, output_path)
    return json.loads(output_path.read_text(encoding="utf-8"))


def consumer_queries(map_path: Path, cohort_path: Path, output_path: Path, *, base_url: str) -> list[dict]:
    _run("source_map_consumer_oracle.mjs", "query", map_path, base_url, cohort_path, output_path)
    return json.loads(output_path.read_text(encoding="utf-8"))


def native_compose(
    stage1_map: Path,
    stage2_map: Path,
    output_map: Path,
    output_report: Path,
    *,
    stage1_base_url: str,
    stage2_base_url: str,
) -> dict:
    _run(
        "native_map_composition.mjs",
        stage1_map,
        stage2_map,
        stage1_base_url,
        stage2_base_url,
        output_map,
        output_report,
    )
    return json.loads(output_report.read_text(encoding="utf-8"))
