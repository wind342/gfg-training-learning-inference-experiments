from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile


PUBLICATION_TAG = "paper-experiments-v3"
CONCEPT_DOI = "10.5281/zenodo.22005307"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def zip_directory(source: Path, destination: Path, archive_root: str) -> int:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(f"{archive_root}/{relative}", date_time=(2026, 8, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return len(files)


def copy_required(source: Path, destination: Path, name: str) -> Path:
    path = source / name
    if not path.is_file():
        raise FileNotFoundError(path)
    target = destination / name
    shutil.copy2(path, target)
    return target


def build_tl_bundle(source: Path, output: Path) -> dict[str, Any]:
    required = {
        "BOUNDARY_PREDICTIONS.jsonl.gz",
        "BOUNDARY_RESULTS.json",
        "DERIVATIVE_AUDIT.json",
        "EXPERIMENT_CONTEXT.json",
        "INDEPENDENT_CHECK.json",
        "MANIFEST.json",
        "PROTOCOL_FREEZE.md",
        "SCIENTIFIC_ASSESSMENT.md",
        "SOURCE_MANIFEST.json",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"TL-P01 missing files: {missing}")
    with tempfile.TemporaryDirectory(prefix="tl-p01-public-") as temporary:
        root = Path(temporary)
        for name in sorted(required):
            shutil.copy2(source / name, root / name)
        write_text(
            root / "PUBLIC_README.md",
            """# TL-P01 actual-update target-boundary evidence

This bundle preserves every one of the 15,264 target-update records, the
frozen result tables, derivative audit and source identities used by TL-P01.
The frozen confirmation split contains four complete held-out runs and 5,088
records.  From the matching Git release run:

    python -m experiments.gfg_nanogpt_actual_update_boundary_v1.INDEPENDENT_CHECKER .

The checker validates the internal manifest and independently recomputes both
the all-run and confirmation metrics.  Native absolute path strings in
SOURCE_MANIFEST.json are immutable provenance labels; the public checker does
not dereference them.
""",
        )
        count = zip_directory(root, output, "tl_p01_actual_update_boundary_v1")
    return {"file_count": count, "bytes": output.stat().st_size, "sha256": sha256(output)}


def build_inf_bundle(source: Path, output: Path) -> dict[str, Any]:
    graph = source / "gfg"
    audit = source / "strict-audit.stdout.json"
    if not (graph / "ARCHIVE_MANIFEST.json").is_file() or not audit.is_file():
        raise FileNotFoundError("INF-E01 graph or strict audit is missing")
    with tempfile.TemporaryDirectory(prefix="inf-e01-public-") as temporary:
        root = Path(temporary)
        shutil.copytree(graph, root / "gfg")
        shutil.copy2(audit, root / "STRICT_LOGIT_LEVEL_AUDIT.json")
        write_text(
            root / "PUBLIC_README.md",
            """# INF-E01 frozen-inference GFG evidence

This bundle contains 13 validated derived GFGs covering 52 checkpoint phases,
their content-addressed tensor payloads and the frozen strict logit-level
audit. From the matching Git release run:

    python -m experiments.gfg_nanogpt_training_learning_inference_projection_v1.PUBLIC_EVIDENCE_CHECKER .

The checker validates the per-run manifest, validation, SQLite and tensor
hashes, recomputes the logit-level component interactions and
query-conditioned support-profile distances, and compares them with the
frozen result at a 1e-12 tolerance. Regenerating these GFGs from native model
forwards requires the original historical parameter checkpoints.
""",
        )
        count = zip_directory(root, output, "inf_e01_frozen_inference_projection_v1")
    return {"file_count": count, "bytes": output.stat().st_size, "sha256": sha256(output)}


def build_instrument_bundle(
    source: Path,
    repository: Path,
    output: Path,
    publication_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    original_manifest_path = source / "INSTRUMENT_ARCHIVE_MANIFEST.json"
    if not original_manifest_path.is_file() or not (source / "source").is_dir():
        raise FileNotFoundError("instrument source archive is incomplete")
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="instruments-public-") as temporary:
        root = Path(temporary)
        shutil.copytree(source / "source", root / "source")
        # Overlay every archived path that exists in the corrected repository.
        for entry in original["files"]:
            relative = Path(entry["relative_path"])
            if relative.parts[0] != "source":
                continue
            repository_path = repository.joinpath(*relative.parts[1:])
            if repository_path.is_file():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repository_path, target)
        files = []
        for path in sorted((root / "source").rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        manifest = dict(original)
        manifest.update(
            {
                "schema_version": "training-learning-instruments-evidence-v2",
                "generated_at_utc": generated_at,
                "publication_tag": PUBLICATION_TAG,
                "publication_commit": publication_commit,
                "source_file_count": len(files),
                "source_total_bytes": sum(entry["bytes"] for entry in files),
                "files": files,
            }
        )
        write_json(root / "INSTRUMENT_ARCHIVE_MANIFEST.json", manifest)
        write_text(
            root / "README.md",
            f"""# Training-learning experimental instruments evidence v2

This archive freezes the four reusable experimental instruments described in
the manuscript Methods at Git tag `{PUBLICATION_TAG}` and commit
`{publication_commit}`: CSRG-4C (TL-E06), realized-update causal forks
(TL-E03), finite-amplitude update paths (TL-E04), and the identity-aligned
target-boundary ledger (TL-E08).

`INSTRUMENT_ARCHIVE_MANIFEST.json` records every archived source file, byte
length and SHA-256 identity. The corrected source index uses repository-relative
links, so it does not depend on a deleted or renamed historical tag.

This package preserves source, protocols and validators. It does not claim to
contain every native training checkpoint or full tensor trajectory; those
public verification boundaries are stated in `PUBLIC_EVIDENCE_MATRIX.md`.
""",
        )
        count = zip_directory(root, output, "training_learning_instruments_evidence_v2")
    return {"file_count": count, "bytes": output.stat().st_size, "sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--previous-archive-root", type=Path, required=True)
    parser.add_argument("--instrument-source", type=Path, required=True)
    parser.add_argument("--tl-p01-source", type=Path, required=True)
    parser.add_argument("--inf-e01-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-commit", required=True)
    args = parser.parse_args()

    repository = args.repository.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    archives: dict[str, dict[str, Any]] = {}
    for name in (
        "nanogpt_base_gfg_evidence_v1.zip",
        "rl_e02_temporal_credit_formal_evidence_v1.zip",
    ):
        path = copy_required(args.previous_archive_root.resolve(), output, name)
        with zipfile.ZipFile(path) as archive:
            archive.testzip()
            file_count = len(archive.infolist())
        archives[name] = {"file_count": file_count, "bytes": path.stat().st_size, "sha256": sha256(path)}

    tl_name = "tl_p01_actual_update_boundary_evidence_v1.zip"
    archives[tl_name] = build_tl_bundle(args.tl_p01_source.resolve(), output / tl_name)
    inf_name = "inf_e01_frozen_inference_gfg_evidence_v1.zip"
    archives[inf_name] = build_inf_bundle(args.inf_e01_source.resolve(), output / inf_name)
    instruments_name = "training_learning_instruments_evidence_v2.zip"
    archives[instruments_name] = build_instrument_bundle(
        args.instrument_source.resolve(),
        repository,
        output / instruments_name,
        args.publication_commit,
        generated_at,
    )

    shutil.copy2(repository / "PUBLIC_EVIDENCE_MATRIX.md", output / "PUBLIC_EVIDENCE_MATRIX.md")
    shutil.copy2(repository / "PUBLIC_ARCHIVE.md", output / "PUBLIC_ARCHIVE.md")
    write_text(
        output / "README.txt",
        f"""Generation-fact evidence and executable experiments

This is an upload candidate for Git tag {PUBLICATION_TAG} and commit
{args.publication_commit}. It corrects repository pointers and adds the
complete TL-P01 prediction ledger and INF-E01 derived inference GFGs. Existing
Zenodo versions remain immutable.

Stable archive-series DOI: https://doi.org/{CONCEPT_DOI}
Repository: https://github.com/wind342/gfg-training-learning-inference-experiments

Run `python tools/verify_publication_evidence_v3.py <this-directory>` from the
matching Git release before publication. The evidence boundary for every
manuscript experiment is stated in PUBLIC_EVIDENCE_MATRIX.md.
""",
    )
    write_text(
        output / "EDITORIAL_CORRECTION_TEMPLATE.md",
        f"""Subject: Updated public evidence release for manuscript 2026-08-25037

Dear Editors,

We have issued a corrected public evidence release for the submitted
manuscript. The new release preserves the frozen experimental results while
correcting obsolete repository pointers and adding the complete TL-P01
prediction ledger and INF-E01 derived inference GFG evidence.

Git tag: {PUBLICATION_TAG}
Git commit: {args.publication_commit}
Zenodo concept DOI: https://doi.org/{CONCEPT_DOI}
Zenodo version DOI: [INSERT AFTER PUBLICATION]

No manuscript result, protocol, experimental threshold or scientific
conclusion was changed. Earlier Git commits, tags and Zenodo versions remain
immutable as part of the audit history.

Sincerely,
Mian Wang
""",
    )

    ordinary_files = sorted(path for path in output.iterdir() if path.name != "ARCHIVE_MANIFEST.json")
    top_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in ordinary_files
        if path.is_file()
    }
    manifest = {
        "schema_version": "gfg-publication-evidence-archive-v3",
        "status": "UPLOAD_CANDIDATE_NOT_YET_PUBLISHED",
        "generated_at_utc": generated_at,
        "repository": "https://github.com/wind342/gfg-training-learning-inference-experiments",
        "publication_tag": PUBLICATION_TAG,
        "publication_commit": args.publication_commit,
        "zenodo_concept_doi": CONCEPT_DOI,
        "archives": archives,
        "files": top_files,
    }
    write_json(output / "ARCHIVE_MANIFEST.json", manifest)
    print(json.dumps({"status": "BUILT", "output": str(output), "archives": archives}, indent=2))


if __name__ == "__main__":
    main()
