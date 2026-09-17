"""Small read-only source adapter; contains no outcome loader."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile  # load before installing the Stage A evidence-read guard

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = "36dab5ce347dbbdac157ef23205f556606d18294"
TAG = "paper-experiments-cross-system-feedback-release"
ALPHAS = (0.25, 0.5, 0.75, 1.0)
CONFIRMATION = frozenset({
    "entry-4ed462761347d6b87e61", "entry-d5b80ca9b9cd18fa343f",
    "entry-786d0a3628f6f791399f", "entry-481b86f81d58d496a687",
})
RULE = {
    "version": "functional-step-size-smoke-v1", "alphas": list(ALPHAS),
    "formula": "((g0+a*first_gap)+a*a*joint_rotation)+a*a*hidden_curvature",
    "score": ["predicted_correct_count", "median_predicted_minimum_margin", "-abs(alpha-1)"],
    "correctness": "all_competitor_gaps_strictly_positive", "unit": "section",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


RULE_HASH = digest(canonical(RULE).encode())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_rows(path, values):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--reports-root", type=Path, default=Path(
        "D:/codex/source-information-continuity-gfg-end-to-end-prediction-v1/experiments/"
        "gfg_nanogpt_cumulative_scientist_v1/reports"))
    p.add_argument("--response-root", type=Path, default=Path(
        "E:/gfg-evidence/nanogpt-adjacent-response-transport-v1/submission"))
    p.add_argument("--boundary-root", type=Path, default=Path(
        "E:/gfg-evidence/nanogpt-actual-update-boundary-v1/submission"))
    p.add_argument("--output-root", type=Path, default=HERE)
    return p


def pre_paths(args):
    joint = args.reports_root / "joint_rotation_branch_v1"
    comp = args.reports_root / "competitor_handoff_outcome_v1"
    return {
        "joint_manifest": joint / "FINAL_MANIFEST.json",
        "joint": joint / "JOINT_ROTATION_COORDINATES.npz",
        "identity": joint / "COORDINATE_IDENTITY_LEDGER.jsonl.gz",
        "derivative_audit": joint / "DERIVATIVE_AUDIT.json",
        "comp_manifest": comp / "FINAL_MANIFEST.json",
        "comp": comp / "COMPETITOR_COORDINATES.npz",
        "comp_identity": comp / "COORDINATE_IDENTITY_LEDGER.jsonl.gz",
        "sources": comp / "SOURCE_SECTION_LEDGER.jsonl.gz",
        "inventory": args.response_root / "RESOLVED_INVENTORY.json",
        "target_identity": args.response_root / "IDENTITY_MATERIAL.json",
    }


def install_read_guard(args, allowed):
    """Enforce the Stage A source boundary, not merely an unused-field promise."""
    allowed = {p.resolve() for p in allowed}
    roots = [args.reports_root.resolve(), args.response_root.resolve(), args.boundary_root.resolve()]
    opened = set()

    def hook(event, values):
        if event != "open" or not isinstance(values[0], (str, bytes)):
            return
        p = Path(values[0]).resolve()
        if any(p.is_relative_to(root) for root in roots):
            require(p in allowed, f"STAGE_A_FORBIDDEN_SOURCE:{p}")
            require(not any(c in str(values[1]) for c in "wax+"), "SOURCE_WRITE_FORBIDDEN")
            opened.add(str(p))
    sys.addaudithook(hook)
    return opened


def load_pre(args):
    paths = pre_paths(args)
    for path in paths.values():
        require(path.is_file(), f"MISSING_PRE_SOURCE:{path}")
    for key, children in (("joint_manifest", ("joint", "identity", "derivative_audit")),
                          ("comp_manifest", ("comp", "comp_identity", "sources"))):
        manifest = read_json(paths[key])
        require(manifest["status"] == "PASS", f"SOURCE_MANIFEST_FAIL:{key}")
        for child in children:
            require(sha(paths[child]) == manifest["output_hashes"][paths[child].name],
                    f"SOURCE_HASH_MISMATCH:{child}")
    identities = rows(paths["identity"])
    other = rows(paths["comp_identity"])
    fields = ("record_id", "entry_id", "section_id", "evaluation_unit_id", "evaluation_row_index", "target_group")
    require([{k: r[k] for k in fields} for r in identities] ==
            [{k: r[k] for k in fields} for r in other], "COORDINATE_IDENTITY_MISMATCH")
    ids = [r["record_id"] for r in identities]
    require(len(ids) == len(set(ids)) == 15264, "TARGET_POPULATION_MISMATCH")
    with np.load(paths["joint"], allow_pickle=False) as z:
        require(z["record_ids"].tolist() == ids, "JOINT_ORDER_MISMATCH")
        f, j, h = (np.asarray(z[k], dtype=np.float64) for k in
                   ("first_gap", "joint_rotation", "hidden_curvature"))
        stored_g = z["gaps"].copy()
    with np.load(paths["comp"], allow_pickle=False) as z:
        require(z["record_ids"].tolist() == ids, "GAP_ORDER_MISMATCH")
        g = np.asarray(z["all_competitor_gaps"], dtype=np.float64)
    require(np.array_equal(g.astype(np.float32), stored_g), "GAP_CACHE_DISAGREEMENT")
    require(all(x.shape == (15264, 23) and np.isfinite(x).all() for x in (g, f, j, h)),
            "INVALID_COORDINATE_ARRAY")
    sources = {r["section_id"]: r for r in rows(paths["sources"])}
    audit = {r["section_id"]: r for r in read_json(paths["derivative_audit"])["sections"]}
    inventory = {r["section_id"]: r for r in read_json(paths["inventory"])["sections"]}
    units = read_json(paths["target_identity"])["entries"]
    sections = sorted({r["section_id"] for r in identities})
    require(len(sections) == len(sources) == len(audit) == len(inventory) == 72, "SECTION_COUNT")
    require(set(sections) == set(sources) == set(audit) == set(inventory), "SECTION_IDENTITY")
    counts = Counter(r["entry_id"] for r in identities)
    require(len(counts) == 12 and set(counts.values()) == {1272}, "RUN_COUNTS")
    require(CONFIRMATION <= set(counts), "CONFIRMATION_IDENTITY")
    for r in identities:
        sid, eid = r["section_id"], r["entry_id"]
        src, a, inv = sources[sid], audit[sid], inventory[sid]
        require(eid == src["entry_id"] == a["entry_id"] == inv["entry_id_audit_only"], "ENTRY_IDENTITY")
        require(src["alpha_value_used"] == 0, "GAPS_NOT_PRE_UPDATE")
        require(src["receiver_state_sha256"] == a["receiver_state_sha256"] and
                src["transition_sha256"] == a["actual_update_sha256"], "DERIVATIVE_SOURCE_IDENTITY")
        require(a["coordinate_time_boundary"] == "PRE_UPDATE_STATE_PLUS_FORMED_ACTUAL_UPDATE", "TIME_BOUNDARY")
        u = units[eid][r["evaluation_row_index"]]
        require(r["evaluation_unit_id"] == u["evaluation_unit_id"] and
                r["target_group"] == u["target_group"], "TARGET_IDENTITY")
        require(r["record_id"] == "factorunit-" + digest(f"{sid}:{r['evaluation_unit_id']}".encode())[:32],
                "FACTOR_IDENTITY")
        r.update({k: u[k] for k in ("row_content_sha256", "upstream_element_identity")})
    return g, f, j, h, identities, sources, audit, inventory


def code_hashes():
    return {p.name: sha(p) for p in sorted(HERE.iterdir()) if
            p.suffix == ".py" or p.name == "PROTOCOL_FREEZE.md"}


def verify_seal(root, check_sources=True):
    seal = read_json(root / "SELECTION_SEAL.json")
    require(sha(root / "SELECTION_SEAL.json") == (root / "SELECTION_SEAL.sha256").read_text().strip(), "SEAL_HASH")
    require(sha(root / "SELECTION_LEDGER.jsonl") == seal["ledger_sha256"], "LEDGER_CHANGED")
    require(seal["rule_sha256"] == RULE_HASH and seal["code_hashes"] == code_hashes(), "RULE_OR_CODE_CHANGED")
    if check_sources:
        for path, value in seal["pre_source_hashes"].items():
            require(sha(path) == value, f"PRE_SOURCE_CHANGED:{path}")
    return seal
