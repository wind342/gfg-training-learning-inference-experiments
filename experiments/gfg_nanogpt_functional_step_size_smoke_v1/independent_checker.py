"""Recompute choices and metrics independently; never writes source evidence."""
from collections import Counter
import json

import numpy as np

from .common import (ALPHAS, CONFIRMATION, RULE_HASH, load_pre, parser,
                     pre_paths, read_json, require, rows, sha, verify_seal)
from .adjudicate import read_outcomes, check_tlp01


def close(left, right, label):
    require(set(left) == set(right), f"METRIC_KEYS:{label}")
    for k in left:
        require(abs(left[k] - right[k]) <= 1e-12, f"METRIC_MISMATCH:{label}:{k}")


def independent_metrics(values, base, pre):
    n = len(values)
    correct = sum(v > 0 for v in values)
    base_correct = sum(v > 0 for v in base)
    triples = list(zip(pre, base, values))
    return {
        "target_count": n, "correct_count": correct, "correct_rate": correct/n,
        "mean_minimum_margin": float(np.mean(values)), "median_minimum_margin": float(np.median(values)),
        "delta_correct_count_vs_native": correct-base_correct,
        "delta_correct_pp_vs_native": 100*(correct/n-base_correct/n),
        "prevented_correct_to_wrong": sum(x > 0 and b <= 0 and v > 0 for x, b, v in triples),
        "additional_wrong_to_correct": sum(x <= 0 and b <= 0 and v > 0 for x, b, v in triples),
        "newly_caused_correct_to_wrong": sum(x > 0 and b > 0 and v <= 0 for x, b, v in triples),
        "lost_wrong_to_correct": sum(x <= 0 and b > 0 and v <= 0 for x, b, v in triples),
    }


def check(args):
    root = args.output_root.resolve()
    seal = verify_seal(root)
    manifest = read_json(root / "MANIFEST.json")
    for name, value in manifest["files"].items():
        require(sha(root / name) == value, f"ARTIFACT_HASH:{name}")
    for path, value in manifest["outcome_source_hashes"].items():
        require(sha(path) == value, f"OUTCOME_SOURCE_HASH:{path}")
    expected_read_set = {str(p.resolve()) for p in pre_paths(args).values()}
    require(set(seal["evidence_files_opened_by_stage_a"]) == expected_read_set, "STAGE_A_READ_SET")
    g, f, j, h, identities, *_ = load_pre(args)
    record_index = {r["record_id"]: i for i, r in enumerate(identities)}
    ledger = rows(root / "SELECTION_LEDGER.jsonl")
    for r in ledger:
        index = [record_index[t["record_id"]] for t in r["targets"]]
        require(len(index) == 212 and len(set(index)) == 212, "SECTION_RECORDS")
        candidates = []
        for a in ALPHAS:
            # Separate implementation; no selector function used by this checker.
            predicted = g[index] + f[index]*a + j[index]*(a*a) + h[index]*(a*a)
            minima = np.min(predicted, axis=1)
            count, median = int(np.count_nonzero(minima > 0)), float(np.median(minima))
            require(r["scores"][str(a)] == {"predicted_correct_count": count,
                    "median_predicted_minimum_margin": median}, "PREDICTION_SCORE")
            candidates.append((count, median, -abs(a-1), a))
        require(r["selected_alpha"] == sorted(candidates)[-1][-1], "SECTION_SELECTION")
        require(r["selection_rule_sha256"] == RULE_HASH, "ROW_RULE_HASH")

    outcomes, _, ties = read_outcomes(args, ledger)
    require(outcomes == rows(root / "ADJUDICATION_LEDGER.jsonl"), "ADJUDICATION_LEDGER")
    result = read_json(root / "RESULTS.json")
    require(check_tlp01(args, ledger, outcomes) == result["tlp01_alpha1_crosscheck"], "TLP01_CHECK")
    require(result["strict_boundary_tie_count"] == ties, "TIE_COUNT")
    collected = {m: [] for m in result["pooled_descriptive"]}
    entry_ids = []
    for n, (selection, outcome) in enumerate(zip(ledger, outcomes)):
        margins = outcome["actual_minimum_margins"]
        rankings = [(sum(v > 0 for v in margins[str(a)]), float(np.median(margins[str(a)])),
                     -abs(a-1), a) for a in ALPHAS]
        oracle = sorted(rankings)[-1][-1]
        declared = result["sections"][n]
        require(declared["section_id"] == selection["section_id"] and declared["oracle_alpha"] == oracle,
                "ORACLE_SECTION")
        for method in collected:
            a = (selection["selected_alpha"] if method == "controller" else
                 oracle if method == "oracle" else 0.0 if method == "alpha0_reference"
                 else float(method.removeprefix("fixed_")))
            v = margins[str(a)]
            close(independent_metrics(v, margins["1.0"], margins["0.0"]),
                  declared["metrics"][method], f"section:{n}:{method}")
            collected[method].extend(v)
        entry_ids.extend([selection["entry_id"]]*212)
    entry_ids = np.asarray(entry_ids)
    arrays = {k: np.asarray(v) for k, v in collected.items()}
    masks = {"all": np.ones(len(entry_ids), bool),
             "development": ~np.isin(entry_ids, list(CONFIRMATION)),
             "confirmation": np.isin(entry_ids, list(CONFIRMATION))}
    masks.update({str(e): entry_ids == e for e in set(entry_ids)})
    for key, mask in masks.items():
        declared = (result["pooled_descriptive"] if key == "all" else
                    result["original_split_descriptive"][key] if key in ("development", "confirmation")
                    else result["per_run"][key])
        for method, a in arrays.items():
            close(independent_metrics(a[mask], arrays["fixed_1.0"][mask], arrays["alpha0_reference"][mask]),
                  declared[method], f"{key}:{method}")
    for a in ALPHAS:
        w = result["controller_vs_fixed_runwise"][str(a)]
        ds = {}
        for e in sorted(set(entry_ids)):
            mask = entry_ids == e
            ds[str(e)] = int((arrays["controller"][mask] > 0).sum() - (arrays[f"fixed_{a}"][mask] > 0).sum())
        require(ds == w["per_run_delta_correct"], "PER_RUN_DELTAS")
        require([w["wins"], w["ties"], w["losses"]] ==
                [sum(d > 0 for d in ds.values()), sum(d == 0 for d in ds.values()), sum(d < 0 for d in ds.values())], "RUN_WTL")
        require(abs(w["mean_run_delta_pp"] - np.mean([100*d/1272 for d in ds.values()])) <= 1e-12, "RUN_MEAN")
    freq = Counter(str(r["selected_alpha"]) for r in ledger)
    require(result["selected_alpha_frequency"] == {str(a): freq[str(a)] for a in ALPHAS}, "FREQUENCY")
    p = result["pooled_descriptive"]
    best = sorted((p[f"fixed_{a}"]["correct_count"], p[f"fixed_{a}"]["median_minimum_margin"], -abs(a-1), a)
                  for a in ALPHAS)[-1][-1]
    require(best == result["best_fixed_alpha_hindsight"], "BEST_FIXED")
    require(result["controller_minus_best_fixed_count"] == p["controller"]["correct_count"]-p[f"fixed_{best}"]["correct_count"], "BEST_FIXED_DELTA")
    require(result["oracle_minus_controller_count"] == p["oracle"]["correct_count"]-p["controller"]["correct_count"], "ORACLE_GAP")
    require(result["oracle_minus_controller_count"] >= 0, "ORACLE_UPPER_BOUND")
    verify_seal(root)
    return {"status": "PASS", "record_count": 15264, "section_count": 72, "run_count": 12,
            "checks": ["Stage A source-read allowlist", "sealed code/rule/ledger/source hashes",
                       "independently recomputed selections", "native raw parameter/update/input identities",
                       "all source outcomes and target identities", "all section/pooled/run/split metrics",
                       "fixed-alpha controls and shared-section oracle", "archived TL-P01 equivalence"]}


if __name__ == "__main__":
    print(json.dumps(check(parser().parse_args()), sort_keys=True))
