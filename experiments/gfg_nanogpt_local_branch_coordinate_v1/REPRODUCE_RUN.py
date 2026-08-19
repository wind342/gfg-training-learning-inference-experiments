from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gfg_nanogpt_local_branch_coordinate_v1.runner import main


if __name__ == "__main__":
    main()
