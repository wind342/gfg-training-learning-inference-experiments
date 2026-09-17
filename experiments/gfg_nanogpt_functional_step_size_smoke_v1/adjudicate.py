"""Stage B: existing native outcomes, after verifying the Stage A seal."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.gfg_nanogpt_adjacent_response_transport_v1.inventory import load_array, load_named_array
from experiments.gfg_nanogpt_adjacent_response_transport_v1.validator import _decision
from .common import (ALPHAS, CONFIRMATION, HERE, digest, load_pre, read_json,
                     require, rows, sha, verify_seal, write_json, write_rows)


def read_outcomes(args, ledger):
    # This is intentionally the first operation before opening any outcome file.
    verify_seal(args.output_root)
    inventory = {r["section_id"]: r for r in read_json(args.response_root / "RESOLVED_INVENTORY.json")["sections"]}
    manifest_path = args.response_root / "FINITE_AMPLITUDE_CURVES_MANIFEST.json"
    manifest = read_json(manifest_path)
    require(manifest["status"] == "PASS" and manifest["section_count"] == 72, "OUTCOME_MANIFEST")
    manifests = {r["section_id"]: r for r in manifest["sections"]}
    evidence_hashes = {str(manifest_path.resolve()): sha(manifest_path)}
    values = []
    ties = 0

    def document(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None:
            require(actual == expected, f"RAW_DOCUMENT_HASH:{path}")
        evidence_hashes[str(path.resolve())] = actual
        return read_json(path)

    def tensors(root, descriptor, named=True):
        # Existing validator checks file, packed raw and every named-child hash.
        result = load_named_array(root, descriptor) if named else load_array(root, descriptor)
        evidence_hashes[str((root / descriptor["locator"]).resolve())] = descriptor["file_sha256"]
        return result

    for record in ledger:
        sid = record["section_id"]
        inv, source = inventory[sid], record["source"]
        path = args.response_root / "sections" / f"{sid}.npz"
        expected = source["section_npz_sha256"]
        require(sha(path) == expected == manifests[sid]["data_file_sha256"], f"OUTCOME_HASH:{sid}")
        evidence_hashes[str(path.resolve())] = expected
        require(manifests[sid]["alpha1_parameter_exact"] and
                manifests[sid]["alpha1_forward_exact_count"] == 12, "NATIVE_ENDPOINT_MANIFEST")
        state = document(inv["receiver_state_path"], source["receiver_state_sha256"])
        transition = document(inv["transition_path"], source["transition_sha256"])
        endpoint = document(inv["native_endpoint_state_path"])
        eid = record["entry_id"]
        require(eid == state["entry_id"] == transition["entry_id"] == endpoint["entry_id"], "RAW_ENTRY")
        require(state["optimizer_step"] == transition["optimizer_step"] == source["optimizer_step"] and
                endpoint["optimizer_step"] == state["optimizer_step"] + 1, "RAW_STEP")
        require(record["window_id"] == state["window_id"] == transition["window_id"] == endpoint["window_id"], "RAW_WINDOW")
        require(state["state"]["parameters"]["raw_tensor_sha256"] == source["current_parameter_commitment"] and
                transition["step"]["parameter_update"]["raw_tensor_sha256"] == source["actual_update_commitment"], "RAW_COMMITMENT")
        root = Path(inv["receiver_state_path"]).parents[3]
        pre = tensors(root, state["state"]["parameters"])
        update = tensors(root, transition["step"]["parameter_update"])
        post = tensors(root, endpoint["state"]["parameters"])
        require(set(pre) == set(update) == set(post), "PARAMETER_NAMES")
        require(all(np.array_equal(np.subtract(post[k], pre[k], dtype=np.float32), update[k])
                    for k in pre), "NOT_EXACT_REALIZED_NATIVE_UPDATE")
        x_path = Path(inv["evaluation_input_path"])
        require(sha(x_path) == record["derivative_identity"]["evaluation_input_sha256"], "INPUT_FILE_IDENTITY")
        evidence_hashes[str(x_path.resolve())] = sha(x_path)
        inputs = np.load(x_path, allow_pickle=False)
        require(digest(np.ascontiguousarray(inputs).tobytes()) == inv["evaluation_input_sha256"], "INPUT_RAW_IDENTITY")
        order = np.asarray([r["evaluation_row_index"] for r in record["targets"]])
        require(sorted(order.tolist()) == list(range(212)), "ROW_COVERAGE")
        for r in record["targets"]:
            require(digest(np.ascontiguousarray(inputs[r["evaluation_row_index"]]).tobytes()) == r["row_content_sha256"], "INPUT_ROW_IDENTITY")
        with np.load(path, allow_pickle=False) as z:
            require(z["alphas"].tolist() == [-0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0], "OUTCOME_GRID")
            groups = z["groups"]
            logits = z["all_logits"]
            stored_margins = z["all_margins"]
            stored_predictions = z["all_predictions"]
        require(logits.shape == (7, 12, 212, 24), "LOGIT_SHAPE")
        require(groups[order].tolist() == [r["target_group"] for r in record["targets"]], "GROUP_ALIGNMENT")
        # Check the recorded ungated baseline against original native probes.
        # These historical arrays are adjudication evidence, never selector inputs.
        for name, index in (("receiver_probe_path", 1), ("native_endpoint_probe_path", 6)):
            probe = document(inv[name])
            require(probe["forwards"][0]["gate_components"] == [], "BASELINE_GATED")
            native = tensors(root, probe["forwards"][0]["logits"], named=False)
            require(np.array_equal(native, logits[index, 0]), "NATIVE_LOGIT_MISMATCH")
        actual = {}
        for alpha, index in ((0.0, 1), (0.25, 3), (0.5, 4), (0.75, 5), (1.0, 6)):
            margins, predictions, *_ = _decision(logits[index, 0], groups)
            require(np.array_equal(margins, stored_margins[index, 0]) and
                    np.array_equal(predictions, stored_predictions[index, 0]), "DERIVED_OUTCOME_MISMATCH")
            # Use exact double-precision differences of the stored logits.
            l = logits[index, 0].astype(np.float64)
            incorrect = l.copy()
            incorrect[np.arange(212), groups] = -np.inf
            m = l[np.arange(212), groups] - incorrect.max(axis=1)
            require(np.isfinite(m).all(), "NONFINITE_OUTCOME")
            ties += int((m == 0).sum())
            actual[str(alpha)] = m[order].tolist()
        values.append({"section_id": sid, "entry_id": eid,
                       "record_ids": [r["record_id"] for r in record["targets"]],
                       "actual_minimum_margins": actual,
                       "source_section_sha256": expected,
                       "transition_id": transition["transition_id"],
                       "receiver_state_id": state["state"]["state_id"],
                       "endpoint_state_id": endpoint["state"]["state_id"],
                       "occurrence_decomposition": transition["occurrence_decomposition"]})
    return values, evidence_hashes, ties


def metric(margin, native, start):
    correct, base, initial = margin > 0, native > 0, start > 0
    return {
        "target_count": len(margin), "correct_count": int(correct.sum()),
        "correct_rate": float(correct.mean()),
        "mean_minimum_margin": float(np.mean(margin)),
        "median_minimum_margin": float(np.median(margin)),
        "delta_correct_count_vs_native": int(correct.sum() - base.sum()),
        "delta_correct_pp_vs_native": float(100 * (correct.mean() - base.mean())),
        "prevented_correct_to_wrong": int((initial & ~base & correct).sum()),
        "additional_wrong_to_correct": int((~initial & ~base & correct).sum()),
        "newly_caused_correct_to_wrong": int((initial & base & ~correct).sum()),
        "lost_wrong_to_correct": int((~initial & base & ~correct).sum()),
    }


def summarize(ledger, outcomes):
    require([r["section_id"] for r in ledger] == [r["section_id"] for r in outcomes], "OUTCOME_SECTION_ORDER")
    methods = [f"fixed_{a}" for a in ALPHAS] + ["controller", "oracle", "alpha0_reference"]
    collected = {m: [] for m in methods}
    entries, sections = [], []
    for selected, outcome in zip(ledger, outcomes):
        actual = {a: np.asarray(v, dtype=np.float64) for a, v in outcome["actual_minimum_margins"].items()}
        oracle = max(ALPHAS, key=lambda a: (int((actual[str(a)] > 0).sum()),
                                          float(np.median(actual[str(a)])), -abs(a-1)))
        choices = {f"fixed_{a}": a for a in ALPHAS}
        choices.update(controller=selected["selected_alpha"], oracle=oracle, alpha0_reference=0.0)
        for method, a in choices.items():
            collected[method].append(actual[str(a)])
        entries.extend([selected["entry_id"]] * 212)
        sections.append({"section_id": selected["section_id"], "entry_id": selected["entry_id"],
                         "selected_alpha": selected["selected_alpha"], "oracle_alpha": oracle,
                         "metrics": {m: metric(actual[str(a)], actual["1.0"], actual["0.0"])
                                     for m, a in choices.items()}})
    arrays = {m: np.concatenate(v) for m, v in collected.items()}
    entries = np.asarray(entries)

    def subset(mask):
        return {m: metric(v[mask], arrays["fixed_1.0"][mask], arrays["alpha0_reference"][mask])
                for m, v in arrays.items()}
    pooled = subset(np.ones(len(entries), dtype=bool))
    per_run = {str(e): subset(entries == e) for e in sorted(set(entries))}
    versus = {}
    for a in ALPHAS:
        deltas = {e: v["controller"]["correct_count"] - v[f"fixed_{a}"]["correct_count"]
                  for e, v in per_run.items()}
        versus[str(a)] = {"per_run_delta_correct": deltas,
                           "wins": sum(d > 0 for d in deltas.values()),
                           "ties": sum(d == 0 for d in deltas.values()),
                           "losses": sum(d < 0 for d in deltas.values()),
                           "mean_run_delta_pp": float(np.mean([100*d/1272 for d in deltas.values()]))}
    best = max(ALPHAS, key=lambda a: (pooled[f"fixed_{a}"]["correct_count"],
                                    pooled[f"fixed_{a}"]["median_minimum_margin"], -abs(a-1)))
    return {
        "schema": "functional-step-size-smoke-results-v1", "status": "PASS",
        "evidence_status": "EXPLORATORY_REUSE_NOT_FRESH_PROSPECTIVE_CONFIRMATION",
        "record_count": len(entries), "section_count": len(sections), "run_count": len(per_run),
        "outer_statistical_unit": "run", "new_training_or_inference": False,
        "selected_alpha_frequency": {str(a): sum(r["selected_alpha"] == a for r in ledger) for a in ALPHAS},
        "pooled_descriptive": pooled, "per_run": per_run, "sections": sections,
        "controller_vs_fixed_runwise": versus,
        "original_split_descriptive": {
            "development": subset(~np.isin(entries, list(CONFIRMATION))),
            "confirmation": subset(np.isin(entries, list(CONFIRMATION))),
        },
        "best_fixed_alpha_hindsight": best,
        "controller_minus_best_fixed_count": pooled["controller"]["correct_count"] - pooled[f"fixed_{best}"]["correct_count"],
        "oracle_minus_controller_count": pooled["oracle"]["correct_count"] - pooled["controller"]["correct_count"],
        "oracle_minus_controller_pp": 100 * (pooled["oracle"]["correct_rate"] - pooled["controller"]["correct_rate"]),
        "oracle_scope": "one shared candidate amplitude per section; hindsight correctness upper bound only",
    }


def check_tlp01(args, ledger, outcomes):
    manifest = read_json(args.boundary_root / "MANIFEST.json")
    path = args.boundary_root / "BOUNDARY_PREDICTIONS.jsonl.gz"
    require(sha(path) == manifest["files"][path.name], "TLP01_LEDGER_HASH")
    original = {r["record_id"]: r for r in rows(path)}
    g, f, j, h, identities, *_ = load_pre(args)
    # Literal original TL-P01 expression, independent of generalized predictor.
    minima = (g + f + j + h).min(axis=1)
    require(set(original) == {r["record_id"] for r in identities}, "TLP01_RECORD_SET")
    maximum = 0.0
    for row, m in zip(identities, minima):
        old = original[row["record_id"]]
        require(all(row[k] == old[k] for k in ("entry_id", "section_id", "evaluation_unit_id", "target_group")), "TLP01_IDENTITY")
        require(bool(m > 0) == old["predicted_final_correct"]["quadratic_complete"], "TLP01_PREDICTION_CHANGED")
        maximum = max(maximum, abs(float(m) - old["minimum_predicted_endpoint_gap"]["quadratic_complete"]))
    require(maximum <= 1e-7, f"TLP01_ENDPOINT_CHANGED:{maximum}")
    g_by_id = {r["record_id"]: g[i] for i, r in enumerate(identities)}
    for selected, outcome in zip(ledger, outcomes):
        pre = np.asarray(outcome["actual_minimum_margins"]["0.0"])
        post = np.asarray(outcome["actual_minimum_margins"]["1.0"])
        for row, m0, m1 in zip(selected["targets"], pre, post):
            old = original[row["record_id"]]
            require(bool(m0 > 0) == old["start_correct"] and bool(m1 > 0) == old["truth_final_correct"], "TLP01_TRUTH_CHANGED")
            require(abs(float(g_by_id[row["record_id"]].min()) - m0) <= 1e-12, "ALPHA_ZERO_GAP_CHANGED")
    return {"status": "PASS", "record_count": len(original), "maximum_predicted_minimum_gap_error": maximum,
            "all_alpha1_predictions_identical": True, "all_native_correctness_identical": True,
            "source_file": str(path.resolve()), "source_sha256": sha(path)}


def report(result):
    p = result["pooled_descriptive"]
    lines = ["# Functional step-size smoke: actual adjudication", "",
             "Exploratory historical-data reuse; not fresh prospective confirmation.", "",
             f"Population: {result['run_count']} runs, {result['section_count']} sections, {result['record_count']} targets.", "",
             f"Selected amplitudes (sections): `{result['selected_alpha_frequency']}`.", "",
             "| Method | Correct / 15264 | Correct % | Mean margin | Median margin | Delta vs native (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in p.items():
        lines.append(f"| {name} | {m['correct_count']} | {100*m['correct_rate']:.6f} | {m['mean_minimum_margin']:.8f} | {m['median_minimum_margin']:.8f} | {m['delta_correct_pp_vs_native']:+.6f} |")
    c = p["controller"]
    lines += ["", f"Controller vs native: prevented correct-to-wrong {c['prevented_correct_to_wrong']}; additional wrong-to-correct {c['additional_wrong_to_correct']}; newly caused correct-to-wrong {c['newly_caused_correct_to_wrong']}; lost wrong-to-correct {c['lost_wrong_to_correct']}.", "",
              f"Best fixed candidate (hindsight): {result['best_fixed_alpha_hindsight']}. Controller minus best fixed: {result['controller_minus_best_fixed_count']:+d} targets.",
              f"Section-level oracle minus controller: {result['oracle_minus_controller_count']} targets ({result['oracle_minus_controller_pp']:.6f} pp).", "",
              "## Run-level comparisons", "", "| Run | Controller correct | Native correct | Delta | Delta pp |", "|---|---:|---:|---:|---:|"]
    for e, v in result["per_run"].items():
        m = v["controller"]
        lines.append(f"| {e} | {m['correct_count']} | {v['fixed_1.0']['correct_count']} | {m['delta_correct_count_vs_native']:+d} | {m['delta_correct_pp_vs_native']:+.6f} |")
    lines += ["", "| Against fixed alpha | Run wins | Ties | Losses | Mean run delta pp |", "|---|---:|---:|---:|---:|"]
    for a, v in result["controller_vs_fixed_runwise"].items():
        lines.append(f"| {a} | {v['wins']} | {v['ties']} | {v['losses']} | {v['mean_run_delta_pp']:+.6f} |")
    lines += ["", "## Boundaries", "",
              "Alpha 0 is not a candidate. The oracle uses one shared amplitude per section, not one per target.",
              "Targets are pooled descriptions; runs are the outer units. The original 8/4 split is descriptive only.",
              "This tests one-step counterfactual outcomes along already formed updates, not an optimizer or a sequential controller.",
              "No claim of superior long-term training, generalization, support organization, or other architectures is justified.",
              "Full fixed-alpha, transition, split, source-hash and section results are retained in RESULTS.json and the ledgers.",
              "No candidate or selector rule was changed after outcome access.", ""]
    return "\n".join(lines)


def run(args):
    root = args.output_root.resolve()
    seal = verify_seal(root)
    require(not (root / "RESULTS.json").exists(), "RESULTS_ALREADY_EXIST")
    ledger = rows(root / "SELECTION_LEDGER.jsonl")
    started = datetime.now(timezone.utc).isoformat()
    outcomes, evidence_hashes, ties = read_outcomes(args, ledger)
    tl = check_tlp01(args, ledger, outcomes)
    result = summarize(ledger, outcomes)
    result.update(tlp01_alpha1_crosscheck=tl, strict_boundary_tie_count=ties,
                  adjudication_started_utc=started, selection_ledger_sha256=seal["ledger_sha256"])
    evidence_hashes[tl["source_file"]] = tl["source_sha256"]
    # Verify read-only sources remained unchanged throughout execution.
    verify_seal(root)
    for path, value in evidence_hashes.items():
        require(sha(path) == value, f"OUTCOME_SOURCE_CHANGED:{path}")
    write_rows(root / "ADJUDICATION_LEDGER.jsonl", outcomes)
    write_json(root / "RESULTS.json", result)
    with (root / "RESULTS.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write(report(result))
    write_json(root / "MANIFEST.json", {
        "schema": "functional-step-size-smoke-manifest-v1",
        "frozen_commit": seal["frozen_commit"], "frozen_tag": seal["frozen_tag"],
        "files": {name: sha(root / name) for name in (
            "SELECTION_LEDGER.jsonl", "SELECTION_SEAL.json", "SELECTION_SEAL.sha256",
            "ADJUDICATION_LEDGER.jsonl", "RESULTS.json", "RESULTS.md")},
        "code_hashes": seal["code_hashes"], "pre_source_hashes": seal["pre_source_hashes"],
        "outcome_source_hashes": evidence_hashes,
        "existing_source_evidence_unchanged": True,
    })
    print("STAGE_B_COMPLETE", flush=True)
