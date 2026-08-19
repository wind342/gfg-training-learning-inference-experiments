from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from gfg_client import GFG


REQUIRED_FILES = {
    "mechanism.py",
    "mechanism_spec.json",
    "state_schema.json",
    "forecast_spec.json",
    "intervention.py",
    "intervention_spec.json",
    "discovery_report.md",
    "query_log.jsonl",
}


class Prefix(str):
    def __new__(cls, root: Path, client: GFG):
        value = super().__new__(cls, str(root))
        value._client = client
        return value

    def __getattr__(self, name):
        return getattr(self._client, name)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_precheck", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("mechanism.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python DUAL_DYNAMICS_SUBMISSION_CHECKER.py "
            "submission /evidence"
        )
    submission = Path(sys.argv[1]).resolve()
    evidence = Path(sys.argv[2]).resolve()
    failures = []
    missing = sorted(
        name for name in REQUIRED_FILES if not (submission / name).is_file()
    )
    failures.extend("missing file: " + name for name in missing)
    if failures:
        print(json.dumps({"failures": failures, "status": "FAIL"}, indent=2))
        raise SystemExit(1)

    module = load_module(submission / "mechanism.py")
    for class_name in (
        "FormationDynamics",
        "StabilityDynamics",
        "CapabilityDynamicsMechanism",
    ):
        if getattr(module, class_name, None) is None:
            failures.append("missing class: " + class_name)
    for class_name in ("FormationDynamics", "StabilityDynamics"):
        value = getattr(module, class_name, None)
        for method_name in ("initialize", "step", "output"):
            if value is None or not callable(getattr(value, method_name, None)):
                failures.append(f"missing method: {class_name}.{method_name}")
    unified = getattr(module, "CapabilityDynamicsMechanism", None)
    for method_name in ("initialize", "forecast"):
        if unified is None or not callable(getattr(unified, method_name, None)):
            failures.append(
                f"missing method: CapabilityDynamicsMechanism.{method_name}"
            )

    if not failures:
        mechanism = module.CapabilityDynamicsMechanism()
        client = GFG(str(evidence), max_step=500)
        try:
            state = mechanism.initialize(Prefix(evidence, client))
            forecast = mechanism.forecast(state)
        finally:
            client.close()
        json.dumps(state, sort_keys=True, allow_nan=False)
        json.dumps(forecast, sort_keys=True, allow_nan=False)
        if not isinstance(state, dict) or not {
            "formation_state",
            "stability_state",
        } <= set(state):
            failures.append("initialize state lacks formation_state/stability_state")

        required = {
            "predicted_formation_curve",
            "predicted_stability_degradation_curve",
            "predicted_validation_curve",
            "mechanism_state",
        }
        if not isinstance(forecast, dict) or not required <= set(forecast):
            failures.append("forecast lacks dual-dynamics fields")
        else:
            formation = {
                row["step"]: float(row["capability"])
                for row in forecast["predicted_formation_curve"]
            }
            degradation = {
                row["step"]: float(row["degradation"])
                for row in forecast["predicted_stability_degradation_curve"]
            }
            combined = {
                row.get("step", row.get("optimizer_step")): float(
                    row.get("accuracy", row.get("validation_accuracy"))
                )
                for row in forecast["predicted_validation_curve"]
            }
            if not formation or not (
                set(formation) == set(degradation) == set(combined)
            ):
                failures.append("component curve grids differ")
            elif any(
                abs(
                    combined[step]
                    - max(0.0, min(1.0, formation[step] - degradation[step]))
                )
                > 1e-9
                for step in combined
            ):
                failures.append("component curves do not compose exactly")
            mechanism_state = forecast["mechanism_state"]
            trajectory = (
                mechanism_state.get("state_trajectory")
                if isinstance(mechanism_state, dict)
                else None
            )
            if not isinstance(mechanism_state, dict) or not {
                "formation_state",
                "stability_state",
            } <= set(mechanism_state):
                failures.append("forecast mechanism_state lacks dual state")
            if not isinstance(trajectory, list) or {
                row.get("step") for row in trajectory if isinstance(row, dict)
            } != set(combined):
                failures.append("state_trajectory does not cover forecast grid")

    result = {"failures": failures, "status": "PASS" if not failures else "FAIL"}
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
