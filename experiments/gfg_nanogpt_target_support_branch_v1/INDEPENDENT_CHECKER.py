from pathlib import Path

from experiments.gfg_nanogpt_target_support_branch_v1.independent import check


if __name__ == "__main__":
    check(Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "target_support_branch_v1")
