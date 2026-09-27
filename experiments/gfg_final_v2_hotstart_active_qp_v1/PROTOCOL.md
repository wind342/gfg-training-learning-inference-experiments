# B: warm-start active-set QP actuator, independent of endpoint experiment A

Base `f357dfdb9d13759dc07a5703d6d11301a3adb946`; preserve cached Hildreth and
all earlier evidence. FewRel task10, 210 updates only, identical task9 inputs,
optimizer, LR, frozen features and relation floors. No Banking77, old-loss
replay, floor relaxation, or endpoint arithmetic changes. Use original fixed256.

Before implementation, analyze all 209 adjacent transitions of cached
Hildreth's 210-step log: discovered/final-active Jaccard, previous retention,
added/deleted constraints and common-pair dual continuity. Also distinguish
positive-dual support from the complete discovered set. Proceed only if the
observed continuity is substantial; preserve analysis even if not proceeding.

Solve min 0.5*||P-P_native||^2 for the currently discovered capability
constraints A P >= EPS + unchanged pair reserve. The half factor changes no
minimizer. Native deficit selection and closest-wrong tie-breaking remain
unchanged. Previous final pairs/duals warm-start their matching current
constraints; previously known pairs on currently selected rows are eligible.
Do not carry protection constraints for rows no longer selected merely because
they appeared in the previous update. Record retained versus filtered pairs.

Persist the constraint registry and FP64 Gram entries across updates. Register
new pairs incrementally; calculate only missing Gram entries in requested
working submatrices. Update RHS at each native head/reserve, never reuse stale
RHS. A dual nonnegative block active-set solver uses inherited multipliers and
their working support; solve passive principal systems, deterministically pivot
violations, and check full primal/dual/complementarity residuals. No tolerance
or algorithm search after seeing outcomes. Failed solver/outer iterations are
recorded and roll back to the receiving head; no unverified endpoint executes.

Every solve is followed by the original full fixed256 relation-floor endpoint.
If deficient, select/add further constraints and solve again. Limit outer
rounds to 100 and dual pivots to 200; FP64 scaled KKT tolerance 1e-10.
Numerical tests include binding/nonbinding/negative-RHS cases and comparisons
to independent SciPy NNLS on fixed synthetic SPD Gram problems.

For each observed QP in validation, independently solve the identical G,b from
zero (cold) outside algorithm timing. Compare KKT/objective/solutions and
iterations; never feed cold solutions into the warm actuator. This ablation
tests warm-start benefit on the same QPs without confounding trajectories.

After validation, one standalone warm replay and one component profile;
compare to a fresh original cached-Hildreth run, not an A endpoint variant.
Report QP pivots separately from outer solves/endpoint checks, discovered and
positive-dual constraints, cache reuse, solver/control wall, final old/new/all
and actual floor violations. If slow or unsuccessful, retain the negative
result and stop; no solver tuning or larger task extension.
