from __future__ import annotations

import subprocess
import sys


def main() -> None:
    commands = [
        [sys.executable, "-m", "pytest", "experiments/opentelemetry_projection/tests"],
        [sys.executable, "-m", "pytest", "tests"],
        [sys.executable, "-m", "pytest", "experiments/database_lineage/tests"],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
