"""Stage A: coordinates -> one sealed common amplitude per section."""
from datetime import datetime, timezone

import numpy as np

from .common import (ALPHAS, FROZEN, RULE, RULE_HASH, TAG, code_hashes,
                     install_read_guard, load_pre, pre_paths, require, sha,
                     write_json, write_rows)


def endpoints(g, f, j, h, alpha):
    # Exact existing TL-P01 addition order at alpha=1; only Taylor-degree scaling.
    return ((g + alpha * f) + alpha**2 * j) + alpha**2 * h


def choose(scores):
    return max(ALPHAS, key=lambda a: (scores[str(a)]["predicted_correct_count"],
                                      scores[str(a)]["median_predicted_minimum_margin"],
                                      -abs(a - 1.0)))


def run(args):
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "SELECTION_LEDGER.jsonl").exists(), "SELECTION_ALREADY_EXISTS")
    paths = pre_paths(args)
    opened = install_read_guard(args, paths.values())
    g, f, j, h, ids, sources, audit, inv = load_pre(args)
    require(np.array_equal(endpoints(g, f, j, h, 1.0), g + f + j + h), "ALPHA_ONE_FORMULA")
    section_ids = np.asarray([r["section_id"] for r in ids])
    ledger = []
    for sid in sorted(set(section_ids)):
        indices = np.flatnonzero(section_ids == sid)
        require(len(indices) == 212, "SECTION_TARGET_COUNT")
        scores = {}
        for a in ALPHAS:
            m = endpoints(g[indices], f[indices], j[indices], h[indices], a).min(axis=1)
            scores[str(a)] = {"predicted_correct_count": int((m > 0).sum()),
                              "median_predicted_minimum_margin": float(np.median(m))}
        ledger.append({
            "section_id": str(sid), "entry_id": ids[indices[0]]["entry_id"],
            "source": sources[sid], "derivative_identity": audit[sid],
            "source_paths": {k: inv[sid][k] for k in
                             ("receiver_state_path", "transition_path", "evaluation_input_path")},
            "observation_id": inv[sid]["observation_id"], "pair_id": inv[sid]["pair_id"],
            "window_id": inv[sid]["window_id"],
            "targets": [ids[i] for i in indices], "scores": scores,
            "selected_alpha": choose(scores), "selection_rule_sha256": RULE_HASH,
        })
    write_rows(root / "SELECTION_LEDGER.jsonl", ledger)
    pre_hashes = {str(p.resolve()): sha(p) for p in paths.values()}
    write_json(root / "SELECTION_SEAL.json", {
        "schema": "functional-step-size-selection-seal-v1", "status": "SEALED_BEFORE_OUTCOME_ACCESS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "frozen_commit": FROZEN, "frozen_tag": TAG,
        "rule": RULE, "rule_sha256": RULE_HASH, "code_hashes": code_hashes(),
        "ledger_sha256": sha(root / "SELECTION_LEDGER.jsonl"), "pre_source_hashes": pre_hashes,
        "evidence_files_opened_by_stage_a": sorted(opened),
        "record_count": 15264, "section_count": 72, "run_count": 12,
        "positive_alpha_outcome_access": False,
    })
    with (root / "SELECTION_SEAL.sha256").open("x", encoding="ascii") as out:
        out.write(sha(root / "SELECTION_SEAL.json") + "\n")
    print("STAGE_A_SEALED " + sha(root / "SELECTION_LEDGER.jsonl"), flush=True)
