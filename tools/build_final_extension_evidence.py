from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.build_cross_system_evidence_v4 import (
    copy_source_tree,
    finalize_bundle,
    sha256,
    write_json,
    write_text,
)


CONCEPT_DOI = "10.5281/zenodo.22005307"
PREVIOUS_PUBLISHED_VERSION_DOI = "10.5281/zenodo.22015318"
DEFAULT_PUBLICATION_TAG = "paper-experiments-cross-system-feedback-release"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_prior_archives(previous_root: Path, output: Path) -> dict[str, dict[str, Any]]:
    manifest_path = previous_root / "ARCHIVE_MANIFEST.json"
    manifest = read_json(manifest_path)
    copied: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(manifest["archives"].items()):
        source = previous_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != expected["bytes"] or sha256(source) != expected["sha256"]:
            raise RuntimeError(f"prior archive identity mismatch: {name}")
        with zipfile.ZipFile(source) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"prior archive CRC failed: {name}:{bad}")
            file_count = len(archive.infolist())
        target = output / name
        shutil.copy2(source, target)
        copied[name] = {
            "file_count": file_count,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "carried_forward_from_manifest_sha256": sha256(manifest_path),
        }
    return copied


def verify_rl_authority(experiment_source: Path, formal_root: Path) -> dict[str, Any]:
    source_manifest = read_json(experiment_source / "ARTIFACT_MANIFEST.json")
    formal_result = formal_root / "FORMAL_RESULT.json"
    if not formal_result.is_file():
        raise FileNotFoundError(formal_result)
    actual = sha256(formal_result)
    expected = source_manifest["formal_result_sha256"]
    if actual != expected:
        raise RuntimeError(
            f"formal result authority mismatch: {experiment_source.name}:{actual}:{expected}"
        )
    return {
        "formal_result_sha256": actual,
        "formal_file_count": sum(1 for path in formal_root.rglob("*") if path.is_file()),
        "formal_uncompressed_bytes": sum(
            path.stat().st_size for path in formal_root.rglob("*") if path.is_file()
        ),
    }


def build_rl_bundle(
    *,
    experiment_id: str,
    experiment_source: Path,
    formal_root: Path,
    output: Path,
    archive_root: str,
    publication_tag: str,
    publication_commit: str,
    generated_at: str,
    public_readme: str,
    audit_summary: Path | None = None,
) -> dict[str, Any]:
    authority = verify_rl_authority(experiment_source, formal_root)
    with tempfile.TemporaryDirectory(prefix=f"{experiment_id.lower()}-public-") as temporary:
        root = Path(temporary)
        copy_source_tree(experiment_source, root / "experiment")
        shutil.copytree(formal_root, root / "formal")
        if audit_summary is not None:
            if not audit_summary.is_file():
                raise FileNotFoundError(audit_summary)
            shutil.copy2(audit_summary, root / "FRESH_NATIVE_REPLAY_AUDIT.json")
        write_json(root / "FORMAL_AUTHORITY.json", authority)
        write_text(root / "PUBLIC_README.md", public_readme)
        return finalize_bundle(
            root,
            output,
            archive_root,
            f"{experiment_id.lower()}-public-evidence-v1",
            experiment_id,
            publication_tag,
            publication_commit,
            generated_at,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--previous-archive-root", type=Path, required=True)
    parser.add_argument("--rl-e05-formal-root", type=Path, required=True)
    parser.add_argument("--rl-e06-formal-root", type=Path, required=True)
    parser.add_argument("--rl-e06-audit-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--publication-tag", default=DEFAULT_PUBLICATION_TAG)
    args = parser.parse_args()

    repository = args.repository.resolve()
    previous_root = args.previous_archive_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    archives = copy_prior_archives(previous_root, output)

    e05_name = "rl_e05_selective_feedback_evidence_v1.zip"
    archives[e05_name] = build_rl_bundle(
        experiment_id="RL-E05",
        experiment_source=repository
        / "experiments/gfg_rl_selective_positive_feedback_support_concentration_v1",
        formal_root=args.rl_e05_formal_root.resolve(),
        output=output / e05_name,
        archive_root="rl_e05_selective_feedback_evidence_v1",
        publication_tag=args.publication_tag,
        publication_commit=args.publication_commit,
        generated_at=generated_at,
        public_readme="""# RL-E05 selective-feedback evidence

This bundle preserves all twelve formal shared-policy executions, their exact
selective, balanced and frozen feedback ledgers, checkpoints, exhaustive
component-version rollbacks, Generation-Fact Graphs and the independent checker.
The preregistered composite verdict remains NOT_SUPPORTED because the large-event
temporal-precedence gate failed; the narrower concentration and crowding
relations are reported separately and remain directly checkable.
""",
    )

    e06_name = "rl_e06_dose_recovery_evidence_v1.zip"
    archives[e06_name] = build_rl_bundle(
        experiment_id="RL-E06",
        experiment_source=repository
        / "experiments/gfg_rl_selective_positive_feedback_dose_recovery_v1",
        formal_root=args.rl_e06_formal_root.resolve(),
        output=output / e06_name,
        archive_root="rl_e06_dose_recovery_evidence_v1",
        publication_tag=args.publication_tag,
        publication_commit=args.publication_commit,
        generated_at=generated_at,
        audit_summary=args.rl_e06_audit_summary.resolve()
        if args.rl_e06_audit_summary
        else None,
        public_readme="""# RL-E06 dose, duration and recovery evidence

This bundle preserves all twelve formal shared-policy executions, 211,200 real
optimizer updates, the complete dose and recovery ledgers, checkpoints,
Generation-Fact Graphs and the frozen independent checker. The observed balanced
recovery from the common update-800 fork is 29.17 percentage points; the same
endpoint is 38.54 points above the matched branch that continued exclusive
feedback through update 3,200. These are distinct comparisons.
""",
    )

    for name in (
        "PUBLIC_EVIDENCE_MATRIX.md",
        "PUBLIC_ARCHIVE.md",
        "FULL_REPRODUCTION_AUDIT.md",
        "CROSS_SYSTEM_EVIDENCE_CHAIN.md",
        "RL_EVIDENCE_CHAIN.md",
        "FINAL_EXTENSION_RELEASE_AUDIT.md",
    ):
        shutil.copy2(repository / name, output / name)

    write_text(
        output / "README.txt",
        f"""Generation-fact evidence and executable experiments: cross-system and feedback extension

This upload carries forward the previously validated publication and cross-system
bundles and adds the complete RL-E05 and RL-E06 formal evidence. It therefore
contains the cross-system training-learning and frozen-inference tests together
with the selective positive-feedback concentration, dose, duration and recovery
tests.

Git tag: {args.publication_tag}
Git commit: {args.publication_commit}
Stable archive-series DOI: https://doi.org/{CONCEPT_DOI}
Repository: https://github.com/wind342/gfg-training-learning-inference-experiments

Run `python tools/verify_final_extension_evidence.py <this-directory>` from the
matching Git release before use. The verifier checks every top-level hash, every
ZIP path and CRC, all cross-system results and checkpoint authorities, and both
reinforcement-feedback evidence bundles.
""",
    )

    top_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "ARCHIVE_MANIFEST.json"
    }
    manifest = {
        "schema_version": "gfg-publication-evidence-final-extension-v1",
        "status": "UPLOAD_CANDIDATE_NOT_YET_PUBLISHED",
        "generated_at_utc": generated_at,
        "repository": "https://github.com/wind342/gfg-training-learning-inference-experiments",
        "publication_tag": args.publication_tag,
        "publication_commit": args.publication_commit,
        "zenodo_concept_doi": CONCEPT_DOI,
        "previous_published_version_doi": PREVIOUS_PUBLISHED_VERSION_DOI,
        "archives": archives,
        "files": top_files,
    }
    write_json(output / "ARCHIVE_MANIFEST.json", manifest)
    print(
        json.dumps(
            {"status": "BUILT", "output": str(output), "archives": archives},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
