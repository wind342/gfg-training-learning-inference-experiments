from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-data-root", type=Path, required=True)
    parser.add_argument("--diffusion-data-root", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint-root", type=Path)
    parser.add_argument("--diffusion-checkpoint-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "experiments.gfg_cross_system_frozen_inference_projection_v1.runtime",
        "--resnet-data-root",
        str(args.resnet_data_root),
        "--diffusion-data-root",
        str(args.diffusion_data_root),
        "--output-root",
        str(args.output_root),
    ]
    if args.resnet_checkpoint_root:
        command.extend(
            ["--resnet-checkpoint-root", str(args.resnet_checkpoint_root)]
        )
    if args.diffusion_checkpoint_root:
        command.extend(
            ["--diffusion-checkpoint-root", str(args.diffusion_checkpoint_root)]
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
