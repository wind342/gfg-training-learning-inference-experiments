from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_report(
    *, repo_root: Path, base_commit: str, branch: str
) -> dict[str, Any]:
    distributions = {}
    for name in ("jsonschema", "pytest"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    identity_files = [repo_root / "pyproject.toml", repo_root / "requirements.lock"]
    return {
        "base_commit": base_commit,
        "branch": branch,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "dependencies": distributions,
        "dependency_file_sha256": {path.name: _sha256(path) for path in identity_files},
        "network_required": False,
        "native_domain_runtimes_used": [],
    }


def artifact_manifest(
    *,
    experiment_root: Path,
    artifact_paths: Iterable[Path],
    base_commit: str,
) -> dict[str, Any]:
    implementation_files = sorted(
        path
        for path in experiment_root.rglob("*")
        if path.is_file()
        and "artifacts" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    artifacts = sorted(
        (path for path in artifact_paths if path.name != "artifact_manifest.json"),
        key=lambda path: path.name,
    )
    return {
        "base_commit": base_commit,
        "implementation_files": [
            {
                "path": path.relative_to(experiment_root).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in implementation_files
        ],
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path), "size": path.stat().st_size}
            for path in artifacts
        ],
        "artifact_count": len(artifacts),
        "status": "SUPPORTED",
    }
