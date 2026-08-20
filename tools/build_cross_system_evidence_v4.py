from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable
import zipfile


CONCEPT_DOI = "10.5281/zenodo.22005307"
DEFAULT_PUBLICATION_TAG = "paper-experiments-v4-cross-system"
ZIP_TIMESTAMP = (2026, 8, 21, 0, 0, 0)


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


def included_source_files(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def copy_source_tree(source: Path, destination: Path) -> None:
    for path in included_source_files(source):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def file_manifest(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def zip_directory(source: Path, destination: Path, archive_root: str) -> int:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(f"{archive_root}/{relative}", date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return len(files)


def finalize_bundle(
    root: Path,
    output: Path,
    archive_root: str,
    schema: str,
    experiment_id: str,
    publication_tag: str,
    publication_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    files = file_manifest(root, {"BUNDLE_MANIFEST.json"})
    manifest = {
        "schema": schema,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "publication_tag": publication_tag,
        "publication_commit": publication_commit,
        "file_count": len(files),
        "total_uncompressed_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }
    write_json(root / "BUNDLE_MANIFEST.json", manifest)
    count = zip_directory(root, output, archive_root)
    return {
        "file_count": count,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def build_training_learning_bundle(
    experiment_id: str,
    experiment_source: Path,
    runtime_source: Path,
    output: Path,
    archive_root: str,
    publication_tag: str,
    publication_commit: str,
    generated_at: str,
    readme: str,
) -> dict[str, Any]:
    required_runtime = {
        "FINAL_MANIFEST.json",
        "GENERALIZATION_RESULTS.json",
        "SCIENTIFIC_ASSESSMENT.md",
    }
    missing = sorted(name for name in required_runtime if not (runtime_source / name).is_file())
    if missing:
        raise FileNotFoundError(f"{experiment_id} runtime missing: {missing}")
    with tempfile.TemporaryDirectory(prefix=f"{experiment_id.lower()}-public-") as temporary:
        root = Path(temporary)
        copy_source_tree(experiment_source, root / "experiment")
        shutil.copytree(runtime_source, root / "runtime")
        write_text(root / "PUBLIC_README.md", readme)
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


def checkpoint_authorities(results_path: Path) -> dict[str, Any]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    bundle_by_system = {
        "resnet": "tl_g01_resnet_cross_system_evidence_v1.zip",
        "diffusion": "tl_g02_diffusion_cross_system_evidence_v1.zip",
    }
    root_by_system = {
        "resnet": "tl_g01_resnet_cross_system_evidence_v1/runtime",
        "diffusion": "tl_g02_diffusion_cross_system_evidence_v1/runtime",
    }
    rows = []
    for run in results["runs"]:
        checkpoint = run["source_identity"]["checkpoint"]
        relative_source = Path(checkpoint["checkpoint"])
        seed_name = f"seed_{run['seed']}"
        rows.append(
            {
                "system": run["system"],
                "seed": run["seed"],
                "source_experiment": "TL-G01" if run["system"] == "resnet" else "TL-G02",
                "authority_bundle": bundle_by_system[run["system"]],
                "authority_member": f"{root_by_system[run['system']]}/{seed_name}/FINAL_CHECKPOINT.pt",
                "native_source_label": relative_source.as_posix(),
                "bytes": checkpoint["byte_count"],
                "sha256": checkpoint["actual_sha256"],
                "identity_passed_in_native_execution": checkpoint["pass"],
            }
        )
    return {
        "schema": "inf-g01-cross-bundle-checkpoint-authorities-v1",
        "relationship": "INF-G01 uses the exact trained checkpoints established by TL-G01 and TL-G02",
        "authorities": rows,
    }


def build_inference_bundle(
    experiment_source: Path,
    runtime_source: Path,
    output: Path,
    publication_tag: str,
    publication_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    required = {"FORMAL_RESULTS.json", "FORMAL_GFG.json"}
    missing = sorted(name for name in required if not (runtime_source / name).is_file())
    if missing:
        raise FileNotFoundError(f"INF-G01 runtime missing: {missing}")
    with tempfile.TemporaryDirectory(prefix="inf-g01-public-") as temporary:
        root = Path(temporary)
        copy_source_tree(experiment_source, root / "experiment")
        shutil.copy2(runtime_source / "FORMAL_RESULTS.json", root / "FORMAL_RESULTS.json")
        shutil.copy2(runtime_source / "FORMAL_GFG.json", root / "FORMAL_GFG.json")
        if (runtime_source / "RUN_RESULTS.json").is_file():
            shutil.copy2(runtime_source / "RUN_RESULTS.json", root / "RUN_RESULTS.json")
        authorities = checkpoint_authorities(root / "FORMAL_RESULTS.json")
        write_json(root / "CHECKPOINT_AUTHORITIES.json", authorities)
        write_text(
            root / "PUBLIC_README.md",
            """# INF-G01 cross-system frozen-inference evidence

This bundle preserves the formal ResNet and diffusion frozen-inference results,
their validated Generation-Fact Graph, the frozen protocol, implementation and
independent checker. The six exact trained checkpoints are not duplicated here.
`CHECKPOINT_AUTHORITIES.json` identifies their byte lengths and SHA-256 values
inside the TL-G01 and TL-G02 evidence bundles.

From the matching Git release, verify the formal results with:

    python -m experiments.gfg_cross_system_frozen_inference_projection_v1.INDEPENDENT_CHECKER \
      --results <bundle>/FORMAL_RESULTS.json --gfg <bundle>/FORMAL_GFG.json

The top-level archive verifier additionally resolves all six cross-bundle
checkpoint authorities and checks their bytes and SHA-256 identities.
""",
        )
        return finalize_bundle(
            root,
            output,
            "inf_g01_cross_system_inference_evidence_v1",
            "inf-g01-public-evidence-v1",
            "INF-G01",
            publication_tag,
            publication_commit,
            generated_at,
        )


def copy_previous_archives(previous_root: Path, output: Path) -> dict[str, dict[str, Any]]:
    previous_manifest = json.loads(
        (previous_root / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8")
    )
    copied: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(previous_manifest["archives"].items()):
        source = previous_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != expected["bytes"] or sha256(source) != expected["sha256"]:
            raise RuntimeError(f"previous archive identity mismatch: {name}")
        target = output / name
        shutil.copy2(source, target)
        with zipfile.ZipFile(target) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"previous archive CRC failed: {name}:{bad}")
            file_count = len(archive.infolist())
        copied[name] = {
            "file_count": file_count,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "carried_forward_from_version_doi": "10.5281/zenodo.22015318",
        }
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--previous-archive-root", type=Path, required=True)
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

    archives = copy_previous_archives(previous_root, output)

    resnet_name = "tl_g01_resnet_cross_system_evidence_v1.zip"
    archives[resnet_name] = build_training_learning_bundle(
        "TL-G01",
        repository / "experiments/gfg_resnet_cifar_training_learning_generalization_v1",
        repository / "runtime/gfg_resnet_cifar_generalization_v1_formal",
        output / resnet_name,
        "tl_g01_resnet_cross_system_evidence_v1",
        args.publication_tag,
        args.publication_commit,
        generated_at,
        """# TL-G01 ResNet cross-system training-learning evidence

This bundle preserves three formal ResNet-18/CIFAR-100/SGD-momentum runs,
including their realized training records, compact GFGs, frozen checkpoints,
cross-run results, protocol, implementation and independent checker. It changes
architecture, modality, task and optimizer from the original nanoGPT system.

From the matching Git release, recompute the formal result with:

    python -m experiments.gfg_resnet_cifar_training_learning_generalization_v1.INDEPENDENT_CHECKER <bundle>/runtime

The CIFAR-100 dataset is not redistributed. Fresh native re-execution requires
the documented PyTorch environment and the standard third-party dataset.
""",
    )

    diffusion_name = "tl_g02_diffusion_cross_system_evidence_v1.zip"
    archives[diffusion_name] = build_training_learning_bundle(
        "TL-G02",
        repository / "experiments/gfg_ddpm_cifar_training_learning_generalization_v1",
        repository / "runtime/gfg_ddpm_cifar_generalization_v1/formal",
        output / diffusion_name,
        "tl_g02_diffusion_cross_system_evidence_v1",
        args.publication_tag,
        args.publication_commit,
        generated_at,
        """# TL-G02 diffusion cross-system training-learning evidence

This bundle preserves three formal time-conditioned U-Net/CIFAR-10/AdamW runs,
including realized training records, compact GFGs, frozen checkpoints, cross-run
results, protocol, implementation and independent checker. It changes the task
from classification to diffusion residual prediction and uses occurrence-level
image--timestep--noise targets.

From the matching Git release, recompute the formal result with:

    python -m experiments.gfg_ddpm_cifar_training_learning_generalization_v1.INDEPENDENT_CHECKER <bundle>/runtime

The CIFAR-10 dataset is not redistributed. Fresh native re-execution requires
the documented PyTorch environment and the standard third-party dataset.
""",
    )

    inference_name = "inf_g01_cross_system_inference_evidence_v1.zip"
    archives[inference_name] = build_inference_bundle(
        repository / "experiments/gfg_cross_system_frozen_inference_projection_v1",
        repository / "runtime/gfg_cross_system_frozen_inference_projection_v1/formal",
        output / inference_name,
        args.publication_tag,
        args.publication_commit,
        generated_at,
    )

    for name in (
        "PUBLIC_EVIDENCE_MATRIX.md",
        "PUBLIC_ARCHIVE.md",
        "FULL_REPRODUCTION_AUDIT.md",
        "CROSS_SYSTEM_EVIDENCE_CHAIN.md",
    ):
        shutil.copy2(repository / name, output / name)
    write_text(
        output / "README.txt",
        f"""Generation-fact evidence and executable experiments: cross-system extension

This upload candidate carries forward every archive bundle from Zenodo version
10.5281/zenodo.22015318 and adds the formal TL-G01, TL-G02 and INF-G01 evidence.
The new experiments change architecture, modality, objective and optimizer,
then retest both training-learning dynamics and frozen-inference projection.

Git tag: {args.publication_tag}
Git commit: {args.publication_commit}
Stable archive-series DOI: https://doi.org/{CONCEPT_DOI}
Repository: https://github.com/wind342/gfg-training-learning-inference-experiments

Run `python tools/verify_cross_system_evidence_v4.py <this-directory>` from the
matching Git release before publication. The verifier checks every top-level
hash, every ZIP path and CRC, the three independent scientific checkers and the
six cross-bundle checkpoint identities.
""",
    )

    ordinary_files = sorted(path for path in output.iterdir() if path.name != "ARCHIVE_MANIFEST.json")
    top_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in ordinary_files
        if path.is_file()
    }
    manifest = {
        "schema_version": "gfg-publication-evidence-archive-v4-cross-system",
        "status": "UPLOAD_CANDIDATE_NOT_YET_PUBLISHED",
        "generated_at_utc": generated_at,
        "repository": "https://github.com/wind342/gfg-training-learning-inference-experiments",
        "publication_tag": args.publication_tag,
        "publication_commit": args.publication_commit,
        "zenodo_concept_doi": CONCEPT_DOI,
        "previous_version_doi": "10.5281/zenodo.22015318",
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
