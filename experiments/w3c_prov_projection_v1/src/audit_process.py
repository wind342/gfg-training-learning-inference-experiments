from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
PYTHON_PREFIX = Path(sys.prefix).resolve()
TEMP_ROOT = Path(os.environ.get("TEMP", os.environ.get("TMP", str(REPO_ROOT / "runtime")))).resolve()


class AuditTrace:
    def __init__(self) -> None:
        self.opened: dict[str, set[str]] = {}
        self.imported_modules: set[str] = set()
        self.subprocesses: set[str] = set()
        self.sockets: set[str] = set()

    @staticmethod
    def normalize_path(value: object) -> str | None:
        if isinstance(value, bytes):
            value = os.fsdecode(value)
        if not isinstance(value, str) or not value:
            return None
        try:
            absolute = Path(os.path.abspath(value))
        except (OSError, ValueError):
            return None
        roots = (
            (REPO_ROOT, "{repo}"),
            (PYTHON_PREFIX, "{python_prefix}"),
            (TEMP_ROOT, "{temp}"),
        )
        for root, label in roots:
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                continue
            return label + "/" + relative.as_posix()
        return "{external}/" + absolute.name

    def hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args:
            path = self.normalize_path(args[0])
            if path is not None:
                mode = str(args[1]) if len(args) > 1 else "unknown"
                self.opened.setdefault(path, set()).add(mode)
        elif event == "import" and args:
            self.imported_modules.add(str(args[0]))
        elif event == "subprocess.Popen" and args:
            self.subprocesses.add(str(args[0]))
        elif event.startswith("socket."):
            self.sockets.add(event)

    def value(self) -> dict[str, Any]:
        return {
            "environment_variable_reads": {
                "reliably_observable": False,
                "values": [],
            },
            "imported_modules": sorted(self.imported_modules),
            "opened_file_paths": [
                {"modes": sorted(modes), "path": path}
                for path, modes in sorted(self.opened.items())
            ],
            "sockets": sorted(self.sockets),
            "subprocesses": sorted(self.subprocesses),
        }


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _output_sha(output: Any, canonical_json_bytes: Any) -> str:
    return _sha(canonical_json_bytes({name: value.hex() for name, value in sorted(output.files.items())}))


def _candidate_run() -> dict[str, Any]:
    from experiments.w3c_prov_projection_v1.src.candidate_projection import project_snapshot
    from experiments.w3c_prov_projection_v1.src.core_capture import CoreCaptureCollector
    from experiments.w3c_prov_projection_v1.src.generator import run_generator
    from experiments.w3c_prov_projection_v1.src.provn import parse_provn, serialize_provn
    from experiments.w3c_prov_projection_v1.src.record_model import canonical_json_bytes

    core = CoreCaptureCollector()
    output = run_generator([core])
    snapshot = core.validated_snapshot()
    records = project_snapshot(snapshot)
    provn = serialize_provn(records)
    parsed = parse_provn(provn)
    if records != parsed:
        raise AssertionError("candidate PROV-N round trip differs")
    return {
        "current_candidate_provn_byte_input_count": 1,
        "input_artifact_roles": [
            "frozen_generator_fixture",
            "validated_core_snapshot",
            "current_candidate_provn_bytes",
        ],
        "normalized_record_count": len(parsed),
        "normalized_record_sha256": _sha(canonical_json_bytes(parsed)),
        "ordinary_output_sha256": _output_sha(output, canonical_json_bytes),
        "output_artifact_roles": [
            "current_candidate_provn_bytes",
            "current_candidate_normalized_records",
        ],
        "prov_bytes_sha256": _sha(provn),
        "validated_snapshot_input_count": 1,
        "validated_snapshot_id": snapshot.snapshot_id,
    }


def _native_run() -> dict[str, Any]:
    from experiments.w3c_prov_projection_v1.src.generator import run_generator
    from experiments.w3c_prov_projection_v1.src.native_reference import NativeProvCollector
    from experiments.w3c_prov_projection_v1.src.provo_normalizer import normalize_provo
    from experiments.w3c_prov_projection_v1.src.record_model import canonical_json_bytes

    native = NativeProvCollector()
    output = run_generator([native])
    ttl = native.qualified_provo()
    records = normalize_provo(ttl)
    return {
        "actual_generator_callback_input_count": 1,
        "input_artifact_roles": [
            "frozen_generator_fixture",
            "actual_generator_callbacks",
            "current_native_provo_bytes",
        ],
        "normalized_record_count": len(records),
        "normalized_record_sha256": _sha(canonical_json_bytes(records)),
        "ordinary_output_sha256": _output_sha(output, canonical_json_bytes),
        "output_artifact_roles": [
            "current_native_provo_bytes",
            "current_native_normalized_records",
        ],
        "prov_bytes_sha256": _sha(ttl),
        "validated_snapshot_input_count": 0,
    }


def _mutation_run(mutation_id: str, target: Path) -> dict[str, Any]:
    if mutation_id == "hidden_exchange_write":
        target.write_text("hidden relation answer\n", encoding="utf-8")
        action = "write"
    else:
        target.read_bytes()
        action = "read"
    return {
        "action": action,
        "input_artifact_roles": ["mutation_fixture"],
        "mutation_id": mutation_id,
        "normalized_record_count": 0,
        "normalized_record_sha256": _sha(b""),
        "output_artifact_roles": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("candidate", "native", "mutation"), required=True)
    parser.add_argument("--mutation-id")
    parser.add_argument("--target")
    args = parser.parse_args()

    trace = AuditTrace()
    sys.addaudithook(trace.hook)
    sys.path.insert(0, str(REPO_ROOT))

    if args.mode == "candidate":
        result = _candidate_run()
    elif args.mode == "native":
        result = _native_run()
    else:
        if not args.mutation_id or not args.target:
            raise ValueError("mutation mode requires --mutation-id and --target")
        result = _mutation_run(args.mutation_id, Path(args.target))
    value = {
        "mode": args.mode,
        "result": result,
        "trace": trace.value(),
    }
    sys.stdout.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
