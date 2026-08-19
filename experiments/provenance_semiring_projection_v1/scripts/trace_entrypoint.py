from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace one isolated experiment entry point")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    trace_path = args.trace.resolve()
    events: list[dict[str, Any]] = []
    called_symbols: set[tuple[str, str]] = set()

    def normalized(value: object) -> str:
        text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
        try:
            resolved = Path(text).resolve()
            if resolved == trace_path:
                return "TRACE_OUTPUT"
            return resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            return text

    def audit_hook(event: str, hook_args: tuple[object, ...]) -> None:
        if event == "open" and hook_args:
            path = normalized(hook_args[0])
            if path != "TRACE_OUTPUT" and path.startswith("experiments/provenance_semiring_projection_v1/"):
                events.append({"event": "file_open", "path": path, "mode": str(hook_args[1]) if len(hook_args) > 1 else None})
        elif event.startswith("subprocess"):
            events.append({
                "event": "subprocess",
                "detail": event,
                "executable": str(hook_args[0]) if hook_args else None,
                "argv": [str(item) for item in hook_args[1]] if len(hook_args) > 1 and isinstance(hook_args[1], (list, tuple)) else [],
                "command_line": str(hook_args[1]) if len(hook_args) > 1 else "",
            })
        elif event.startswith("socket"):
            events.append({"event": "socket", "detail": event})

    def profiler(frame: Any, event: str, _arg: object) -> None:
        if event != "call":
            return
        filename = normalized(frame.f_code.co_filename)
        if filename.startswith("experiments/provenance_semiring_projection_v1/"):
            called_symbols.add((filename, frame.f_code.co_name))

    sys.addaudithook(audit_hook)
    sys.setprofile(profiler)
    module = importlib.import_module(args.module)
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    sys.argv = [args.module, *forwarded]
    return_code = int(module.main())
    sys.setprofile(None)
    document = {
        "schema_version": "isolated-entrypoint-trace-v1",
        "module": args.module,
        "return_code": return_code,
        "events": events,
        "called_symbols": [
            {"file": filename, "symbol": symbol}
            for filename, symbol in sorted(called_symbols)
        ],
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
