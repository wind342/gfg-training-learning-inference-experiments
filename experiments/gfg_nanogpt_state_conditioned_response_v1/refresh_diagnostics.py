from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .dataset import build_dataset, load_records
from .model import file_sha256
from .runner import (
    ALL_MODELS,
    DEFAULT_OUTPUT,
    SOURCE_ROOT,
    assessment_text,
    challenge_audit,
    failure_audit,
    read_json,
    write_gzip_jsonl,
    write_json,
)


def refresh(output_root: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    dataset = build_dataset(load_records(SOURCE_ROOT / "PRETARGET_FACTOR_RECORDS.jsonl.gz"))
    with gzip.open(output_root / "RESPONSE_CURVE_PREDICTIONS.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    predictions = {
        name: np.asarray([row["predictions"][name]["displacement"] for row in rows], dtype=np.float64)
        for name in ALL_MODELS
    }
    normalizers = np.asarray([row["outer_normalization_scale"] for row in rows], dtype=np.float64)
    outer = {"predictions": predictions, "normalizers": normalizers}
    failure_summary, failures = failure_audit(dataset, outer)
    failure_path = output_root / "FAILURE_CASE_LEDGER.jsonl.gz"
    write_gzip_jsonl(failure_path, failures)
    failure_summary.update({"ledger": failure_path.name, "ledger_sha256": file_sha256(failure_path)})
    challenge = challenge_audit(SOURCE_ROOT / "SURVIVING_CONDITIONAL_COUNTEREXAMPLES.json", rows)
    challenge_rows = challenge.pop("rows")
    challenge_path = output_root / "FROZEN_CHALLENGE_AUDIT.jsonl.gz"
    write_gzip_jsonl(challenge_path, challenge_rows)
    challenge.update({"ledger": challenge_path.name, "ledger_sha256": file_sha256(challenge_path)})
    failure_summary["frozen_1477_pair_challenge"] = challenge
    write_json(output_root / "FAILURE_CASE_ANALYSIS.json", failure_summary)
    result = read_json(output_root / "NONLINEAR_RESPONSE_MODEL_RESULTS.json")
    ablation = read_json(output_root / "ABLATION_RESULTS.json")
    (output_root / "SCIENTIFIC_ASSESSMENT.md").write_text(
        assessment_text(result, ablation, challenge, failure_summary), encoding="utf-8", newline="\n"
    )
    manifest = read_json(output_root / "MANIFEST.json")
    manifest["status"] = "MODEL_COMPLETE_PENDING_INDEPENDENT_CHECK_AND_GFG"
    manifest["deliverables"] = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name not in {"MANIFEST.json", "INDEPENDENT_CHECK.json"}:
            manifest["deliverables"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json(output_root / "MANIFEST.json", manifest)
    return {"failure_count": failure_summary["failure_count"], "support_state_count": failure_summary["provisional_category_counts"].get("SUPPORT_STATE_MISSING", 0)}


if __name__ == "__main__":
    print(json.dumps(refresh(), sort_keys=True))
