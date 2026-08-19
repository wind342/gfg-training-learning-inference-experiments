from pathlib import Path
import json
import sys


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "experiments" / "gfg_nanogpt_native_prebranch_left_history_v1").is_dir():
            return candidate
    raise RuntimeError("REPOSITORY_ROOT_NOT_FOUND")


ROOT = _repository_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gfg_nanogpt_native_prebranch_left_history_v1.independent import check


if __name__ == "__main__":
    result = check(Path(__file__).resolve().parent)
    print(json.dumps({"status": result["status"], "check_count": result["check_count"]}, sort_keys=True))
