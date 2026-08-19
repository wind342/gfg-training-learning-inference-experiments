from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

from .candidate_validator import runtime_gfg_prefix, runtime_value_failures
from .common import file_sha256, payload_sha256, write_json


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("SEALED_MECHANISM_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_and_seal_forecast(
    *,
    submission: Path,
    participant_repository: Path,
    prefix_directory: Path,
    prediction_cut_step: int,
    candidate_seal: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    prior_root = os.environ.get("NANOGPT_GFG_ROOT")
    prior_log = os.environ.get("GFG_QUERY_LOG")
    sys.path.insert(0, str(participant_repository))
    try:
        os.environ["NANOGPT_GFG_ROOT"] = str(prefix_directory)
        os.environ["GFG_QUERY_LOG"] = str(
            output_path.with_name("forecast_query_log.jsonl")
        )
        module = _load(
            submission / "mechanism.py",
            "sealed_capability_mechanism_forecast",
        )
        mechanism = module.CapabilityDynamicsMechanism()
        prefix = runtime_gfg_prefix(prefix_directory)
        try:
            state = mechanism.initialize(prefix)
            forecast = mechanism.forecast(state)
        finally:
            prefix.close()
        failures = [
            *runtime_value_failures(state, "$.state"),
            *runtime_value_failures(forecast, "$.forecast"),
        ]
        if failures:
            raise RuntimeError(
                "FORECAST_DYNAMIC_LOOKUP_MATERIAL:"
                + ",".join(sorted(failures))
            )
    finally:
        if sys.path[0] == str(participant_repository):
            sys.path.pop(0)
        if prior_root is None:
            os.environ.pop("NANOGPT_GFG_ROOT", None)
        else:
            os.environ["NANOGPT_GFG_ROOT"] = prior_root
        if prior_log is None:
            os.environ.pop("GFG_QUERY_LOG", None)
        else:
            os.environ["GFG_QUERY_LOG"] = prior_log
    state_sha = payload_sha256(state)
    forecast_sha = payload_sha256(forecast)
    material = {
        "candidate_seal_sha256": candidate_seal[
            "candidate_seal_sha256"
        ],
        "forecast": forecast,
        "forecast_sha256": forecast_sha,
        "future_gfg_reads": 0,
        "mechanism_state": state,
        "mechanism_state_sha256": state_sha,
        "prediction_cut_step": prediction_cut_step,
        "prefix_gfg_sha256": file_sha256(
            prefix_directory / "participant_gfg.sqlite3"
        ),
        "schema": "sealed-unseen-training-forecast-v2",
    }
    sealed = {
        **material,
        "forecast_seal_sha256": payload_sha256(material),
        "status": "SEALED_BEFORE_FUTURE",
    }
    write_json(output_path, sealed)
    return sealed
