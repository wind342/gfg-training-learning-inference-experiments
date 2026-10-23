from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE = "experiments.gfg_moe_cifar_training_learning_inference_generalization_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            "-m",
            f"{PACKAGE}.runner",
            "--mode",
            args.mode,
            "--data-root",
            str(args.data_root),
            "--output-root",
            str(args.output_root),
        ],
        check=True,
    )
    if args.mode == "formal":
        subprocess.run([sys.executable, "-m", f"{PACKAGE}.aggregate", str(args.output_root)], check=True)
        subprocess.run([sys.executable, "-m", f"{PACKAGE}.INDEPENDENT_CHECKER", str(args.output_root)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
