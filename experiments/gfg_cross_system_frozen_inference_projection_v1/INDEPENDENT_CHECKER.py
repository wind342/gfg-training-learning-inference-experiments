from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_tests(run: dict[str, Any], contract: dict[str, Any]) -> dict[str, bool]:
    system = run["system"]
    thresholds = contract["tests"]
    expected_components = {
        "resnet": {"layer1", "layer2", "layer3", "layer4"},
        "diffusion": {
            "high_resolution_skip",
            "low_resolution_skip",
            "bottleneck",
            "decoder_refinement",
        },
    }[system]
    source = run["source_identity"]
    checkpoint = source["checkpoint"]
    source_exact = (
        checkpoint["pass"]
        and checkpoint["actual_sha256"] == checkpoint["expected_sha256"]
        and source["initial_state_sha256"]
        == source["expected_initial_state_sha256"]
    )
    if system == "resnet":
        source_exact = source_exact and (
            source["trained_state_sha256"]
            == source["expected_trained_state_sha256"]
        )
    frozen = run["frozen_state"]
    repeat_error = (
        frozen["repeat_output_max_abs_error"]
        if system == "resnet"
        else max(
            frozen["one_step_repeat_max_abs_error"],
            frozen["complete_sample_repeat_max_abs_error"],
        )
    )
    components = run["component_output_rms"]
    gates = run["single_gate_complete_output_effects"]
    rollbacks = run["rollbacks"]
    rollback_field = (
        "complete_logit_max_abs_change"
        if system == "resnet"
        else "complete_sample_max_abs_change"
    )
    return {
        "source_version_identity": source_exact,
        "persistent_state_frozen": (
            frozen["model_before"] == frozen["model_after"]
            and frozen["optimizer_before"] == frozen["optimizer_after"]
        ),
        "repeat_inference_exact": repeat_error
        <= float(thresholds["repeat_output_tolerance"]),
        "components_called": set(components) == expected_components
        and all(
            value > float(thresholds["component_output_rms_minimum"])
            for value in components.values()
        ),
        "gate_changes_output": set(gates) == expected_components
        and max(gates.values(), default=0.0)
        > float(thresholds["gate_effect_minimum"]),
        "query_conditioned_support": run["support"]["maximum_query_profile_l1"]
        > float(thresholds["query_profile_l1_minimum"]),
        "nonadditive_combination": run["support"][
            "maximum_absolute_pair_interaction"
        ]
        > float(thresholds[f"{system}_interaction_minimum"]),
        "learned_version_dependence": set(rollbacks) == expected_components
        and max(
            (values[rollback_field] for values in rollbacks.values()), default=0.0
        )
        > float(thresholds["rollback_effect_minimum"]),
        "exact_restoration": run["maximum_restoration_output_error"]
        <= float(thresholds["restoration_tolerance"]),
    }


def _gfg_errors(gfg: dict[str, Any], results: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    facts = gfg.get("facts", [])
    occurrences = gfg.get("occurrences", [])
    fact_ids = [row.get("id") for row in facts]
    occurrence_ids = [row.get("id") for row in occurrences]
    if len(fact_ids) != len(set(fact_ids)):
        errors.append("DUPLICATE_FACT_ID")
    if len(occurrence_ids) != len(set(occurrence_ids)):
        errors.append("DUPLICATE_OCCURRENCE_ID")
    fact_by_id = {row.get("id"): row for row in facts}
    required = {"u", "tau", "omega", "z", "rho"}
    if any(not required.issubset(row) for row in facts):
        errors.append("INCOMPLETE_ATOMIC_FACT")
    for occurrence in occurrences:
        fact = fact_by_id.get(occurrence.get("realizes_fact"))
        if fact is None or fact.get("omega") != occurrence.get("id"):
            errors.append("OCCURRENCE_FACT_INCIDENCE_MISMATCH")
            break
    expected_occurrences: set[str] = set()
    for run in results["runs"]:
        system = run["system"]
        seed = run["seed"]
        expected_occurrences.add(f"{system}:{seed}:frozen_inference")
        expected_occurrences.add(f"{system}:{seed}:support_coalitions")
        for component in run["rollbacks"]:
            expected_occurrences.add(f"{system}:{seed}:rollback:{component}")
    if set(occurrence_ids) != expected_occurrences:
        errors.append("FORMAL_OCCURRENCE_SET_MISMATCH")
    validation = gfg.get("validation", {})
    if validation.get("status") != "PASS" or validation.get("errors"):
        errors.append("RECORDED_GFG_VALIDATION_FAILED")
    if validation.get("fact_count") != len(facts):
        errors.append("FACT_COUNT_MISMATCH")
    if validation.get("occurrence_count") != len(occurrences):
        errors.append("OCCURRENCE_COUNT_MISMATCH")
    return errors


def check(
    results_path: Path,
    gfg_path: Path,
    contract_path: Path = PACKAGE / "MODEL_CONTRACT.json",
) -> dict[str, Any]:
    results = _read(results_path)
    gfg = _read(gfg_path)
    contract = _read(contract_path)
    errors = _gfg_errors(gfg, results)
    expected_seeds = {
        system: {int(seed) for seed in spec["seeds"]}
        for system, spec in contract["systems"].items()
    }
    observed_seeds: dict[str, set[int]] = {"resnet": set(), "diffusion": set()}
    run_checks: list[dict[str, Any]] = []
    for run in results.get("runs", []):
        system = run.get("system")
        if system not in observed_seeds:
            errors.append(f"UNKNOWN_SYSTEM:{system}")
            continue
        observed_seeds[system].add(int(run["seed"]))
        recomputed = _expected_tests(run, contract)
        matches_recorded = recomputed == run.get("tests")
        passes = all(recomputed.values()) and matches_recorded and run.get("pass") is True
        if not passes:
            errors.append(f"RUN_FAILED:{system}:{run['seed']}")
        run_checks.append(
            {
                "system": system,
                "seed": run["seed"],
                "recomputed_tests": recomputed,
                "matches_recorded_tests": matches_recorded,
                "pass": passes,
            }
        )
    for system, seeds in expected_seeds.items():
        if observed_seeds[system] != seeds:
            errors.append(f"SEED_SET_MISMATCH:{system}")
        recorded = results.get("systems", {}).get(system, {})
        if recorded.get("seed_count") != len(seeds):
            errors.append(f"RECORDED_SEED_COUNT_MISMATCH:{system}")
        if recorded.get("passing_seeds") != len(seeds):
            errors.append(f"RECORDED_PASS_COUNT_MISMATCH:{system}")
        if recorded.get("all_seeds_pass") is not True:
            errors.append(f"RECORDED_SYSTEM_VERDICT_FAILED:{system}")
    expected_verdict = "CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED"
    if results.get("verdict") != expected_verdict:
        errors.append("FORMAL_VERDICT_MISMATCH")
    return {
        "schema": "gfg-cross-system-frozen-inference-independent-check-v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "formal_verdict": results.get("verdict"),
        "run_checks": run_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=PACKAGE / "FORMAL_RESULTS.json"
    )
    parser.add_argument("--gfg", type=Path, default=PACKAGE / "FORMAL_GFG.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.results, args.gfg)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
