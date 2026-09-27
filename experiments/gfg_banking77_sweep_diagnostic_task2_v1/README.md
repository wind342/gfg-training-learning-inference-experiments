# Task2 first-actual-safe sweep diagnostic

Source: accepted Stage1 A/B/C from f0b6aaf. Every original sweep is observed; early-safe candidates are never committed. Both original trajectories, native proposals, optimizer moments, predictor outputs, complete solver decisions and final checkpoint states are unchanged. First-safe and original terminal candidates are independently reconstructed and forwarded; committed heads are independently forwarded.

| Quantity | 210 | 630 |
|---|---:|---:|
|Unsafe native updates|208|618|
|Original local solves|307|618|
|Original sweeps|48686|64285|
|First-safe prefix sweeps|5173|648|
|Potential avoided sweeps on fixed path|43513|63637|
|Local solves safe before original stop|267|618|
|Safe → unsafe transitions|167|418|
|Ever-safe but final-unsafe local solves|59|0|
|Cap solves|100|121|
|Cap solves ever safe|91|121|
|Earlier candidate differs from executed head|207|616|
|First-safe sweep per update median / mean / p90|1.00 / 24.87 / 147.00|1.00 / 1.05 / 1.00|
|Potential sweep reduction|89.37%|98.99%|

| Cost on fixed update states (seconds) | 210 | 630 |
|---|---:|---:|
|Plain original solver, measured|2.748149|3.024291|
|First-safe prefix solver, estimated|0.258257|0.020309|
|Solver time saved, estimated|2.489893|3.003982|
|Check every sweep until first safe, measured checks on original states|11.954889|1.972039|
|Original one terminal check per outer, matched probes|0.822103|1.578183|
|Additional check cost|11.132786|0.393856|
|Net saving, estimate (positive saves time)|-8.642894|2.610126|
|Full candidate check median (ms)|2.527950|2.301100|
|Full candidate check mean (ms)|2.742993|2.488303|

Each candidate check includes CPU reconstruction, FP32 conversion, parameter transfers and exact buffer256 forward/counts. Audit/log/hash work is excluded. The matched plain solver is run without callbacks; its total normalizes observed per-sweep CPU profiles. GPU/CPU interleaving and cache effects make savings estimates approximate. Skipped outer geometry/diagnostic work is not credited.

These are per-update counterfactuals on the fixed historical states, not a measured early-stop controller. Different early candidates can change later native updates, floors and learning. No check interval, residual threshold, early-stop policy or new solver was selected or deployed.

All original committed endpoints remain floor-safe; old CE backward is zero. Frozen BERT/cache/input/source hashes remain unchanged. Test was never accessed, and tasks3–11 were not run.

Full per-sweep observations, per-update summaries and audit identities: `E:\gfg-evidence\banking77-sweep-diagnostic-task2-v1`. Compact evidence is in `results/`.

## Interpretation

On the original states, first-sweep candidates are already floor-safe for 158/208 unsafe updates in the 210 arm and 603/618 in the 630 arm. All 121 capped local solves in the 630 arm are safe at sweep1. This directly establishes substantial solver work after an earlier functionally safe candidate.

The 210 arm has 40 local solves that never become safe on their original inner path. Exhaustively checking them adds substantial cost before original outer continuation. Its first-safe cumulative sweep median is 1 but its mean is 24.87 and p90 is 147. The 630 arm first-safe cumulative maximum is 3.

Safety is non-monotone: 167/418 safe-to-unsafe transitions are observed across local paths. In the 210 arm, 59 local solves are safe at an earlier point but unsafe at their original terminal point. The observer never uses this information to alter stopping or continuation.

For checking every sweep until first safe on these fixed update states: estimated solver savings are 2.49/3.00s, while additional candidate-check costs are 11.13/0.39s. Net estimates therefore worsen by 8.64s for 210 and improve by 2.61s for 630. This is not a measured training speedup, and skipped outer geometry/diagnostics are not credited. The full check includes reconstruction and transfers, which dominate the small endpoint alone.

First-safe candidates differ from the original executed head on 207/208 and 616/618 unsafe updates. Early commitment would create a different training trajectory. The diagnostic supports testing a separate stopping policy; it does not establish its learnability, future floors or end-to-end performance. No such policy is implemented here.
