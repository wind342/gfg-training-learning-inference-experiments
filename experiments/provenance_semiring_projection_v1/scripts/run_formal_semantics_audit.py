from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.formal_semantics import (
    build_formal_target_semantics_audit,
    render_formal_target_semantics_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit formal lower-target semantics")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    audit = build_formal_target_semantics_audit()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "formal_target_semantics_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.artifact_root / "formal_target_semantics_audit.md").write_text(
        render_formal_target_semantics_markdown(audit),
        encoding="utf-8",
        newline="\n",
    )
    return 0 if audit["formal_algebraic_target_count"] >= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
