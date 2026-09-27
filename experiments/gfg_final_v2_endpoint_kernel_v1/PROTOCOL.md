# A: fixed endpoint numerical implementations, task10 only

Base `f357dfdb9d13759dc07a5703d6d11301a3adb946`. Preserve all old code and
evidence. FewRel task10 only, 210 updates, same task9 V2-210 checkpoint,
AdamW state, features, labels, K, LR5e-5 and seed2021. No Banking77.

Arms in fixed order: original 256, preallocated loop256, preallocated batched
256/512/1024/2048. Immutable feature bank is padded/copied once at task entry;
the optimized forwards reuse output storage and never allocate/copy/cat per
tile. Batched variants use one baddbmm API invocation for all contiguous tiles.
Report API calls separately from the number of constituent tile matrices:
a batched API call is not a claim of one physical GPU kernel launch.

Each arm passes numerical preflight before its 210-update validation. Compare
identical heads and feature rows with the original fixed256 arithmetic, record
logit/margin/sign discrepancies and repeatability. Do not force byte identity
or change tile choice after observing results. K is fixed; refuse a variant
that fails its initial receiving floor. Keep Hildreth coordinates, reserves,
selection and stop rule unchanged. An altered trajectory is a new fixed
numerical implementation, never labelled a byte-identical acceleration.

At every candidate, an independent read-only fixed256 observer compares margin
signs, counts and closest-wrong choices. Check every actual executed floor under
the arm AND fixed256. Check head/moment hashes and control decisions against
the frozen cached trajectory, without feeding observer results to the arm.
Test is read only after each training replay; report old/new/all macro metrics
under both the arm and fixed256 evaluation. Preserve any failure.

Separate validation, one standalone timing per arm, and one component profile
per arm. Count setup/padding and actual commit/recheck in algorithm wall;
exclude external observers, verification hashes, serialization and final test.
Time only one GPU workload at a time. No best-of selection or confidence interval.
B's actuator is not used in A. No combination of A and B performance claims.
