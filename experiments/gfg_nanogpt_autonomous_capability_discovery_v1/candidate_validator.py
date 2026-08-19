from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

from .common import canonical_bytes, payload_sha256


REQUIRED_FILES = (
    "mechanism.py",
    "mechanism_spec.json",
    "state_schema.json",
    "forecast_spec.json",
    "intervention.py",
    "intervention_spec.json",
    "discovery_report.md",
    "query_log.jsonl",
)
RUNNER_CONTROL_FILES = {"FINAL_SUBMISSION_READY.json"}
FORBIDDEN_IMPORTS = {
    "aiohttp",
    "builtins",
    "glob",
    "httpx",
    "importlib",
    "io",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
FORBIDDEN_CALLS = {"compile", "eval", "exec", "__import__", "open"}


class RuntimeGFGPrefix(str):
    def __new__(cls, root: Path, client: Any) -> RuntimeGFGPrefix:
        value = super().__new__(cls, str(root))
        value._client = client
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def close(self) -> None:
        self._client.close()


def runtime_gfg_prefix(root: Path, max_step: int | None = None) -> RuntimeGFGPrefix:
    module = importlib.import_module("gfg_client")
    return RuntimeGFGPrefix(root, module.GFG(str(root), max_step=max_step))


def runtime_value_failures(value: Any, path: str = "$") -> list[str]:
    """Reject dynamic lookup material from executable candidate state.

    A mechanism may retain scientific identities and tensor hashes, but it
    must not smuggle the host prefix path, a run identifier, or wall-clock
    material into the state later supplied to the intervention.
    """

    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if "run_id" in lowered or "wall_clock" in lowered:
                failures.append("FORBIDDEN_DYNAMIC_LOOKUP_KEY:" + path)
            failures.extend(runtime_value_failures(child, f"{path}.{key_text}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            failures.extend(runtime_value_failures(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "run_id" in lowered or "wall_clock" in lowered:
            failures.append("FORBIDDEN_DYNAMIC_LOOKUP_VALUE:" + path)
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/"):
            failures.append("ABSOLUTE_RUNTIME_PATH:" + path)
        if re.search(r"\b20\d\d-\d\d-\d\d\b", value):
            failures.append("ABSOLUTE_RUNTIME_DATE:" + path)
    return failures


def _scan_python(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    failures.append("FORBIDDEN_IMPORT:" + alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                failures.append("FORBIDDEN_IMPORT:" + node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                failures.append("FORBIDDEN_CALL:" + node.func.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            lowered = value.lower()
            if "run_id" in lowered or "wall_clock" in lowered:
                failures.append("FORBIDDEN_LOOKUP_LITERAL")
            if re.search(r"\b20\d\d-\d\d-\d\d\b", value):
                failures.append("ABSOLUTE_DATE_LITERAL")
            if re.match(r"^[A-Za-z]:[\\/]", value):
                failures.append("ABSOLUTE_HOST_PATH_LITERAL")
    return sorted(set(failures))


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("CANDIDATE_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mechanism(module: Any) -> Any:
    candidate = getattr(module, "CapabilityDynamicsMechanism", None)
    if candidate is None:
        raise RuntimeError("MECHANISM_CLASS_MISSING")
    return candidate()


def _submodel_contract_gates(module: Any) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for class_name, prefix in (
        ("FormationDynamics", "formation"),
        ("StabilityDynamics", "stability"),
    ):
        submodel = getattr(module, class_name, None)
        gates[f"{prefix}_submodel_class_present"] = submodel is not None
        for method_name in ("initialize", "step", "output"):
            gates[f"{prefix}_submodel_{method_name}_present"] = (
                submodel is not None
                and callable(getattr(submodel, method_name, None))
            )
    unified = getattr(module, "CapabilityDynamicsMechanism", None)
    gates["unified_mechanism_class_present"] = unified is not None
    gates["unified_initialize_present"] = (
        unified is not None and callable(getattr(unified, "initialize", None))
    )
    gates["unified_forecast_present"] = (
        unified is not None and callable(getattr(unified, "forecast", None))
    )
    return gates


def _state_has_dual_components(state: Any) -> bool:
    return (
        isinstance(state, dict)
        and "formation_state" in state
        and "stability_state" in state
    )


def _forecast_gates(forecast: Any) -> dict[str, bool]:
    required = {
        "will_transition",
        "transition_step_low_200",
        "transition_step_high_200",
        "transition_step_low_500",
        "transition_step_high_500",
        "predicted_formation_curve",
        "predicted_stability_degradation_curve",
        "predicted_validation_curve",
        "mechanism_state",
        "post_formation_stability",
        "predicted_instability_intervals",
    }
    if not isinstance(forecast, dict):
        return {"forecast_is_object": False}
    curve = forecast.get("predicted_validation_curve")
    normalized_curve = (
        [normalize_forecast_curve_row(row) for row in curve]
        if isinstance(curve, list)
        else []
    )
    formation_curve = forecast.get("predicted_formation_curve")
    normalized_formation_curve = (
        [
            normalize_component_curve_row(row, "capability")
            for row in formation_curve
        ]
        if isinstance(formation_curve, list)
        else []
    )
    stability_curve = forecast.get("predicted_stability_degradation_curve")
    normalized_stability_curve = (
        [
            normalize_component_curve_row(row, "degradation")
            for row in stability_curve
        ]
        if isinstance(stability_curve, list)
        else []
    )

    def interval_valid(suffix: str, maximum_width: int) -> bool:
        low = forecast.get(f"transition_step_low_{suffix}")
        high = forecast.get(f"transition_step_high_{suffix}")
        if low is None and high is None:
            return True
        return (
            isinstance(low, int)
            and isinstance(high, int)
            and 0 <= low <= high <= 10000
            and high - low <= maximum_width
        )

    stability = forecast.get("post_formation_stability")
    instability_intervals = forecast.get("predicted_instability_intervals")
    mechanism_state = forecast.get("mechanism_state")
    state_trajectory = (
        mechanism_state.get("state_trajectory")
        if isinstance(mechanism_state, dict)
        else None
    )

    combined_points = (
        {point[0]: point[1] for point in normalized_curve if point is not None}
        if all(point is not None for point in normalized_curve)
        else {}
    )
    formation_points = (
        {
            point[0]: point[1]
            for point in normalized_formation_curve
            if point is not None
        }
        if all(point is not None for point in normalized_formation_curve)
        else {}
    )
    stability_points = (
        {
            point[0]: point[1]
            for point in normalized_stability_curve
            if point is not None
        }
        if all(point is not None for point in normalized_stability_curve)
        else {}
    )
    common_component_steps = (
        set(combined_points) == set(formation_points) == set(stability_points)
        and bool(combined_points)
        and len(combined_points) == len(normalized_curve)
        and len(formation_points) == len(normalized_formation_curve)
        and len(stability_points) == len(normalized_stability_curve)
    )
    composition_valid = common_component_steps and all(
        abs(
            combined_points[step]
            - max(0.0, min(1.0, formation_points[step] - stability_points[step]))
        )
        <= 1e-9
        for step in combined_points
    )

    def trajectory_valid() -> bool:
        if not isinstance(state_trajectory, list) or not state_trajectory:
            return False
        trajectory_steps: list[int] = []
        for row in state_trajectory:
            if not isinstance(row, dict):
                return False
            if not {"step", "formation_state", "stability_state"} <= set(row):
                return False
            if not isinstance(row["step"], int):
                return False
            trajectory_steps.append(row["step"])
        return (
            len(trajectory_steps) == len(set(trajectory_steps))
            and set(trajectory_steps) == set(combined_points)
        )

    def instability_intervals_valid() -> bool:
        if not isinstance(instability_intervals, list):
            return False
        for row in instability_intervals:
            if not isinstance(row, dict):
                return False
            low = row.get("step_low")
            high = row.get("step_high")
            if (
                not isinstance(low, int)
                or not isinstance(high, int)
                or not 0 <= low <= high <= 10000
            ):
                return False
        return True

    return {
        "forecast_is_object": True,
        "forecast_fields_complete": required <= set(forecast),
        "transition_boolean": isinstance(forecast.get("will_transition"), bool),
        "transition_interval_200_valid": interval_valid("200", 200),
        "transition_interval_500_valid": interval_valid("500", 500),
        "curve_nonempty": isinstance(curve, list) and bool(curve),
        "curve_rows_valid": (
            isinstance(curve, list) and all(row is not None for row in normalized_curve)
        ),
        "formation_curve_nonempty": (
            isinstance(formation_curve, list) and bool(formation_curve)
        ),
        "formation_curve_rows_valid": (
            isinstance(formation_curve, list)
            and all(row is not None for row in normalized_formation_curve)
        ),
        "stability_curve_nonempty": (
            isinstance(stability_curve, list) and bool(stability_curve)
        ),
        "stability_curve_rows_valid": (
            isinstance(stability_curve, list)
            and all(row is not None for row in normalized_stability_curve)
        ),
        "component_curve_steps_match": common_component_steps,
        "component_curve_composition_valid": composition_valid,
        "mechanism_state_has_dual_components": _state_has_dual_components(
            mechanism_state
        ),
        "state_trajectory_complete": trajectory_valid(),
        "post_formation_stability_valid": stability
        in {
            "STABLE",
            "TRANSIENT_DEGRADATION_RECOVERY",
            "PERSISTENT_DEGRADATION",
            "UNDETERMINED",
        },
        "predicted_instability_intervals_valid": (instability_intervals_valid()),
    }


def normalize_forecast_curve_row(
    row: Any,
) -> tuple[int, float] | None:
    if not isinstance(row, dict):
        return None
    step = row.get("step", row.get("optimizer_step"))
    accuracy = row.get("accuracy", row.get("validation_accuracy"))
    if (
        not isinstance(step, int)
        or not isinstance(accuracy, (int, float))
        or not 0 <= accuracy <= 1
    ):
        return None
    return step, float(accuracy)


def normalize_component_curve_row(
    row: Any,
    value_name: str,
) -> tuple[int, float] | None:
    if not isinstance(row, dict):
        return None
    step = row.get("step")
    value = row.get(value_name)
    if (
        not isinstance(step, int)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        return None
    return step, float(value)


def intervention_shift_interval(
    intervention_spec: dict[str, Any],
) -> tuple[int, int] | None:
    for low_name, high_name in (
        (
            "predicted_transition_shift_low",
            "predicted_transition_shift_high",
        ),
        ("transition_step_shift_low", "transition_step_shift_high"),
        ("predicted_shift_step_low", "predicted_shift_step_high"),
    ):
        low = intervention_spec.get(low_name)
        high = intervention_spec.get(high_name)
        if isinstance(low, int) and isinstance(high, int):
            return low, high
    for interval_name in (
        "transition_step_shift_interval",
        "predicted_shift_interval",
    ):
        interval = intervention_spec.get(interval_name)
        if isinstance(interval, dict):
            low = interval.get("step_low")
            high = interval.get("step_high")
            if isinstance(low, int) and isinstance(high, int):
                return low, high
    return None


def intervention_direction(
    intervention_spec: dict[str, Any],
) -> str | None:
    candidates: list[Any] = [
        intervention_spec.get("direction"),
        intervention_spec.get("predicted_direction"),
    ]
    prediction = intervention_spec.get("prediction")
    if isinstance(prediction, dict):
        candidates.append(prediction.get("direction"))
    declared = {value for value in candidates if value is not None}
    if len(declared) != 1:
        return None
    direction = declared.pop()
    if direction not in {"ADVANCE", "DELAY"}:
        return None
    return direction


def validate_candidate(
    *,
    submission: Path,
    interface_gfg_prefix: Path,
    participant_repository: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQUIRED_FILES:
        if not (submission / name).is_file():
            failures.append("MISSING_FILE:" + name)
    if failures:
        return {
            "failures": failures,
            "status": "CANDIDATE_EXECUTION_PLATFORM_FAILURE",
        }
    unexpected = [
        path.name
        for path in submission.iterdir()
        if path.name not in REQUIRED_FILES
        and path.name not in {"parameters", "__pycache__", *RUNNER_CONTROL_FILES}
    ]
    if unexpected:
        failures.extend("UNEXPECTED_FILE:" + name for name in unexpected)
    for name in ("mechanism.py", "intervention.py"):
        failures.extend(_scan_python(submission / name))
    json_documents: dict[str, Any] = {}
    for name in (
        "mechanism_spec.json",
        "state_schema.json",
        "forecast_spec.json",
        "intervention_spec.json",
    ):
        try:
            json_documents[name] = json.loads(
                (submission / name).read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            failures.append("INVALID_JSON:" + name)
    intervention_spec = json_documents.get("intervention_spec.json", {})
    if intervention_direction(intervention_spec) is None:
        failures.append("INTERVENTION_DIRECTION_INVALID")
    if intervention_shift_interval(intervention_spec) is None:
        failures.append("INTERVENTION_SHIFT_INTERVAL_MISSING")
    if intervention_spec.get("predicted_stability_effect") not in {
        "IMPROVE",
        "WORSEN",
        "NO_CHANGE",
    }:
        failures.append("INTERVENTION_STABILITY_EFFECT_INVALID")
    if (
        not isinstance(intervention_spec.get("stability_effect_rationale"), str)
        or not intervention_spec.get("stability_effect_rationale", "").strip()
    ):
        failures.append("INTERVENTION_STABILITY_RATIONALE_MISSING")
    total_bytes = sum(
        path.stat().st_size for path in submission.rglob("*") if path.is_file()
    )
    if total_bytes > 20 * 1024 * 1024:
        failures.append("CANDIDATE_SIZE_EXCEEDS_20_MIB")
    deterministic = False
    state: Any = None
    forecast: Any = None
    forecast_gates: dict[str, bool] = {}
    submodel_gates: dict[str, bool] = {}
    if not failures:
        prior_path = list(sys.path)
        prior_root = __import__("os").environ.get("NANOGPT_GFG_ROOT")
        prior_log = __import__("os").environ.get("GFG_QUERY_LOG")
        try:
            if participant_repository is not None:
                sys.path.insert(0, str(participant_repository))
            __import__("os").environ["NANOGPT_GFG_ROOT"] = str(interface_gfg_prefix)
            __import__("os").environ["GFG_QUERY_LOG"] = str(
                submission.parent / "candidate_validation_query_log.jsonl"
            )
            module = _load_module(
                submission / "mechanism.py",
                "sealed_capability_mechanism_validation",
            )
            submodel_gates = _submodel_contract_gates(module)
            failures.extend(
                name for name, passed in submodel_gates.items() if not passed
            )
            first = _mechanism(module)
            prefix_a = runtime_gfg_prefix(interface_gfg_prefix, max_step=500)
            try:
                state_a = first.initialize(prefix_a)
                forecast_a = first.forecast(state_a)
            finally:
                prefix_a.close()
            second = _mechanism(module)
            prefix_b = runtime_gfg_prefix(interface_gfg_prefix, max_step=500)
            try:
                state_b = second.initialize(prefix_b)
                forecast_b = second.forecast(state_b)
            finally:
                prefix_b.close()
            canonical_bytes(state_a)
            canonical_bytes(forecast_a)
            deterministic = canonical_bytes(state_a) == canonical_bytes(
                state_b
            ) and canonical_bytes(forecast_a) == canonical_bytes(forecast_b)
            state, forecast = state_a, forecast_a
            forecast_gates = _forecast_gates(forecast)
            if not _state_has_dual_components(state):
                failures.append("INITIAL_STATE_DUAL_COMPONENTS_MISSING")
            failures.extend(runtime_value_failures(state, "$.state"))
            failures.extend(runtime_value_failures(forecast, "$.forecast"))
            if not deterministic:
                failures.append("CANDIDATE_NONDETERMINISTIC")
            failures.extend(
                name for name, passed in forecast_gates.items() if not passed
            )
            intervention_module = _load_module(
                submission / "intervention.py",
                "sealed_training_intervention_validation",
            )
            intervention_class = getattr(
                intervention_module, "TrainingIntervention", None
            )
            if intervention_class is None:
                failures.append("INTERVENTION_CLASS_MISSING")
            else:
                intervention = intervention_class()
                if not callable(getattr(intervention, "initialize", None)):
                    failures.append("INTERVENTION_INITIALIZE_MISSING")
                if not callable(getattr(intervention, "apply", None)):
                    failures.append("INTERVENTION_APPLY_MISSING")
                else:
                    signature = inspect.signature(intervention.apply)
                    if len(signature.parameters) != 3:
                        failures.append("INTERVENTION_APPLY_SIGNATURE")
        except BaseException as exc:
            failures.append(
                "CANDIDATE_INTERFACE_EXCEPTION:"
                + type(exc).__name__
                + ":"
                + str(exc)[:300]
            )
        finally:
            sys.path[:] = prior_path
            environment = __import__("os").environ
            if prior_root is None:
                environment.pop("NANOGPT_GFG_ROOT", None)
            else:
                environment["NANOGPT_GFG_ROOT"] = prior_root
            if prior_log is None:
                environment.pop("GFG_QUERY_LOG", None)
            else:
                environment["GFG_QUERY_LOG"] = prior_log
    material = {
        "deterministic_replay": deterministic,
        "failures": sorted(set(failures)),
        "finite_canonical_state": state is not None,
        "forecast_gates": forecast_gates,
        "dual_submodel_contract": submodel_gates,
        "forecast_sha256": (payload_sha256(forecast) if forecast is not None else None),
        "schema": "candidate-interface-validation-v2",
        "state_sha256": (payload_sha256(state) if state is not None else None),
        "status": "PASS" if not failures else "FAIL",
        "submission_bytes": total_bytes,
    }
    material["validation_sha256"] = payload_sha256(material)
    return material
