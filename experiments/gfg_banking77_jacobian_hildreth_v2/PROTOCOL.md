# Banking77 nonlinear V2 cutter v2: exact-check approximate local candidates

Follow-up authorized after v1 smoke and posthoc diagnosis. Preserve all v1
source/evidence byte-for-byte. This is a new trajectory, not a byte-identical
engineering speedup and not a fresh unseen-benchmark claim.

All v1 model/data/source/feature hashes, seed2021, 11x7 class order, reserved
10 controls/intent, batch16 schedules, AdamW settings and moments semantics,
fixed task-entry K, deficit selection, EPS1e-6, numerical guard, trust rule
(rho=2*native displacement, half on poor actual repair, floor rho_initial/16),
outer cap8, inner cap200 and convergence tolerances are inherited exactly.
Both budgets210/630 are fixed before this new smoke. No test tuning/access.

The only actuator change: an INNER_LIMIT local iterate that is finite and
respects the frozen trust-ball tolerance is reconstructed/materialized and
actual FP32 endpoint-checked. If C>=K, first-safe commit. Otherwise relinearize
at this candidate, update selected constraints from its actual margins, and
continue up to the original outer cap. Invalid numerical candidates or outer
cap exhaustion roll back, keeping exactly one native moments advancement.
Convergence status and remaining local residuals are recorded transparently:
accepted approximate local iterates are not claimed exact QP minimizers.
No floor relaxation, scalar alpha grid, old CE backward, replay loss, stale
Jacobian, or solver substitution. The objective anchor remains native.

Engineering remains analytic structured Jacobian/Gram, cached immutable CPU
H/abs(H)/input Gram/IDs/labels/relations, persistent fixed256 GPU bank buffers.
Native-safe executes with no Jacobian. No new threshold tuning or sweep-cap
increase. An exact check is required before every commit including rollback.

Smoke BOTH210/630 with unchanged gates: floor violations0 and exact optimizer
semantics; >=20 unsafe natives; unsafe repair>=90%; rollback<=10% scheduled;
current train macro>=50%; final train CE<=90% entry; median executed/native
norm ratio on triggered steps>=0.1 (rollbacks count as0). Native task2 remains
a byte-checked replay of the sealed original native reference.

If both smoke gates pass, independently replay all840 controlled steps before
continuing tasks3-11 for BOTH budgets. Full formal trajectories retain these
fixed settings; later task learning metrics do not trigger retuning. Stop on
actual safety/state failure. Reuse unchanged sealed Native/ER/RanPAC baselines;
RanPAC RP10000 cumulative only. New final test evaluation occurs only after
both full streams are sealed, with no protocol changes based on test results.
If either smoke fails, preserve negative results and stop formal expansion.

Report v1 and v2 separately: approximate local commits, cap rechecks, remaining
QP residuals, floor, repair, learning, sweeps, endpoints, old CE backward0,
derivative/Gram/cutter FLOP estimates and instrumented wall. Do not infer a new
trajectory's accuracy by counting posthoc v1 safe candidates. Include all
common setup/controller wall and state what cost estimates omit.
