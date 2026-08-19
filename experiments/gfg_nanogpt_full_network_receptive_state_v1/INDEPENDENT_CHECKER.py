from __future__ import annotations

import argparse
import json
from pathlib import Path

from .independent import check
from .runner import DEFAULT_REPORT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    print(json.dumps(check(args.report_root.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
