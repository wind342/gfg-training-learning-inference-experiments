import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gfg_nanogpt_local_branch_coordinate_v1.independent import check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    print(check(args.report_root.resolve()))
