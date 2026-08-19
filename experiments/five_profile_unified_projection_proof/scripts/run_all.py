from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..src.runner import run_all, write_failure


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed unified reproduction of five frozen P1/P2 projection proofs.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifacts = args.artifacts.resolve()
    try:
        result = run_all(repo, artifacts)
    except BaseException as error:
        write_failure(artifacts, error)
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    summary = result["summary"]
    print(json.dumps({"status": summary["status"], "canonical_summary_hash": summary["determinism"]["run_1_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
