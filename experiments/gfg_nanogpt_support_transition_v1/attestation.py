from __future__ import annotations

import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    write_json,
)


def build_runtime_attestation(
    *,
    phase: str,
    repository: Path,
    trainer_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    repository = repository.resolve()
    trainer_root = trainer_root.resolve()
    code_paths = [
        repository / "experiments/gfg_nanogpt_support_transition_v1/capture_contract_v1.json",
        repository / "experiments/gfg_nanogpt_support_transition_v1/execution.py",
        repository / "experiments/gfg_nanogpt_support_transition_v1/runtime.py",
        repository / "experiments/gfg_nanogpt_support_transition_v1/selection.py",
        repository / "experiments/gfg_nanogpt_support_transition_v1/storage.py",
        repository / "experiments/gfg_nanogpt_support_transition_v1/transition_gfg.py",
        repository / "experiments/gfg_nanogpt_support_transition_v1/implementation_amendments_v1.json",
        repository / "experiments/gfg_nanogpt_support_redundancy_v1/runtime.py",
        repository / "experiments/gfg_nanogpt_autonomous_capability_discovery_v1/common.py",
        repository / "experiments/gfg_nanogpt_autonomous_capability_discovery_v1/training_gfg.py",
    ]
    repository_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        text=True,
    )
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("CST_ATTESTATION_CUDA_NOT_AVAILABLE")
    properties = torch.cuda.get_device_properties(0)
    material = {
        "code_files": {
            path.relative_to(repository).as_posix(): file_sha256(path)
            for path in code_paths
        },
        "cuda": {
            "available": cuda_available,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "device_name": properties.name,
            "device_total_memory": properties.total_memory,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "runtime": torch.version.cuda,
            "torch_version": torch.__version__,
        },
        "historical_rng_boundary": {
            "historical_cpu_and_cuda_rng_payload_materialized": False,
            "matched_branch_rng_restored_from_content_bound_current_runtime_seed": True,
            "stochastic_operator_in_admitted_path": False,
        },
        "model_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=trainer_root, text=True
        ).strip(),
        "model_py_sha256": file_sha256(trainer_root / "model.py"),
        "phase": phase,
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "repository_status_sha256": payload_sha256(repository_status),
        "repository_worktree_clean": not bool(repository_status.strip()),
        "schema": "nanogpt-support-transition-runtime-attestation-v1",
        "status": "PASS",
    }
    result = {**material, "attestation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
