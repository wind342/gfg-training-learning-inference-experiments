from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import aggregate
from .numeric import sha256_file


def check_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_paths = sorted(root.rglob("MANIFEST.json")) + sorted(
        root.rglob("FINAL_MANIFEST.json")
    )
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        for name, expected in manifest["files"].items():
            path = base / name
            if not path.is_file():
                errors.append(f"MISSING:{path}")
                continue
            if path.stat().st_size != expected["bytes"]:
                errors.append(f"SIZE:{path}")
            if sha256_file(path) != expected["sha256"]:
                errors.append(f"SHA256:{path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    errors = check_manifest(root)
    recomputed = aggregate(root)
    recorded_path = root / "GENERALIZATION_RESULTS.json"
    recorded = (
        json.loads(recorded_path.read_text(encoding="utf-8"))
        if recorded_path.is_file()
        else None
    )
    if recorded is not None and recomputed != recorded:
        errors.append("RESULT_RECOMPUTATION_MISMATCH")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "recomputed_verdict": recomputed["verdict"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
