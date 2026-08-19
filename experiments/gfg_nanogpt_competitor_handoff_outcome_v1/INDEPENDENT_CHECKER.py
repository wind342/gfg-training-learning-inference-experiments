from pathlib import Path

from experiments.gfg_nanogpt_competitor_handoff_outcome_v1.independent import check


if __name__ == "__main__":
    check(Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports" / "competitor_handoff_outcome_v1")
