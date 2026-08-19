# Experiment Report: Provenance Semirings as a Strict Hierarchical Projection

## Outcome

Within the frozen positive relational-algebra profile, the machine evidence supports this exact hierarchy:

```text
Gamma_G(omega)
  -- exact strict projection --> N[X]
  -- exact algebraic projections --> {bag N, Boolean B, PosBool(X)}
  -- exact task projections --> {flat source support, Vars(N[X]), existing Database which-lineage}
```

The final machine status is `PROVENANCE_SEMIRING_STRICT_HIERARCHY_FORMAL_SEMANTICS_HARDENING_SUPPORTED`. This is a bounded construction over the declared finite fixtures and positive operators, not a claim about all provenance semirings, arbitrary SQL, all DBMSs, or universal query equivalence.

The primary algebraic authority is Green, Karvounarakis, and Tannen, “Provenance Semirings,” PODS 2007, DOI 10.1145/1265530.1265535. The frozen author PDF has SHA-256 `74e092702db58518afeaf909e1d3380848165b2cb9ae75dc6822b04f66aa5be0`; the commutative-semiring definition is on proceedings page 33, and the N[X], homomorphism, and flat P(X) surfaces are on proceedings page 34. Witness-basis Why provenance was checked separately against Buneman, Khanna, and Tan, ICDT 2001, paper pages 8–10; that frozen PDF has SHA-256 `6c244258ab44a229957a1a16605787f1b9ade11bee0ef6eac8086eb6905e1087`.

<!-- BEGIN MACHINE-GENERATED REPORT STATISTICS -->
## Machine-derived experiment statistics

This block is generated from final machine artifacts; it is not maintained by hand.

| Metric | Value |
| --- | ---: |
| Frozen P1 cases | 13 |
| Source-variable observations | 135 |
| Unique source identities | 131 |
| Logical outputs | 42 |
| Polynomial terms | 197 |
| Coefficient observations | 197 |
| Monomial factors | 332 |
| Exponent observations | 332 |
| Strictness pairs | 5 |
| Real strictness executions | 10 |
| All classified lower comparisons | 52 |
| Formal algebraic comparisons | 39 |
| Task-projection comparisons | 13 |
| Negative controls | 70 |
| Passing bottom-up test areas | 50 |
| Manifested files | 139 |
| Manifested artifact files | 63 |
<!-- END MACHINE-GENERATED REPORT STATISTICS -->

## Formal-semantics hardening conclusion

PR #19's frozen P1 and P2 conclusions are preserved. All 13 P1 cases remain exact with zero mismatch and zero repair, including W12; all five P2 witness pairs remain supported across ten real executions. `artifacts/v1_result_preservation.json` compares the current conclusions with frozen PR #19 head `f20ff57501b754b111be893565092c0e107c8b73`.

The flat P(X) calculation uses finite variable sets with union for both addition and multiplication. Here, 0 and 1 are both the empty set, and the empty set does not annihilate nonempty values under multiplication. The implementation is therefore named `flat_source_support_view` and classified as `PARTIAL_NONZERO_SUPPORT_VIEW`: it agrees exactly with `Vars(N[X])` on frozen nonzero output support but is not claimed as a whole-carrier commutative-semiring homomorphism.

The flat support view is not witness-basis Why provenance. `Which(X)`, `Trio(X)`, and witness-basis Why remain `NOT_EVALUATED` because the frozen authorities do not justify inventing those algebraic structures here. Existing tuple-level which-lineage is separately classified as `NON_SEMIRING_TASK_PROJECTION`.

The verified algebraic targets are bag N, Boolean B, and PosBool(X). Their direct K-relation executions agree with the corresponding N[X]-derived projections in all 39 comparisons. The 13 flat-support comparisons and the existing Database which-lineage bridge are reported separately as task projections.

Native N[X] now uses `src/native_polynomial_oracle.py`, with its own immutable monomial representation, zero, one, addition, multiplication, coefficient aggregation, exponent aggregation, canonical serialization, and local SHA-256 variable mapping. Candidate retains the frozen `NXPolynomial` implementation. The comparison process reads only their final canonical JSON documents. Shared algebra-helper and shared variable-helper counts are both zero.

Every number in the generated block is recomputed from `nx_exact_comparison.json`, `nx_field_coverage.json`, `native_nx_polynomials.json`, `nx_strictness_counterexamples.json`, `hierarchical_projection_exact_comparison.json`, `negative_controls.json`, the hardening test result, and `artifact_manifest.json`. `report_artifact_consistency.json` fails closed if persisted statistics or the generated report block drift. The manifest's stable count summary breaks self-reference; its complete per-file size and SHA-256 table remains independently rehashed.

Within the frozen positive relational-algebra profile, complete generation facts exactly and strictly project to the canonical N[X] provenance polynomial. An independently implemented native polynomial oracle and a Core-only Candidate agree coefficient-by-coefficient and exponent-by-exponent. N[X] then yields exact algebraic projections to the formally verified target semirings, while flat source-support and tuple-level which-lineage are reported separately as exact task projections. Thus the hierarchical unification remains supported without conflating semiring homomorphisms with non-semiring views.

在冻结的正关系代数 profile 范围内，完整生成事实能够精确且严格地投影为规范 N[X] provenance 多项式。完全独立实现的原生多项式 Oracle 与仅依赖 Core 的 Candidate 在每个系数和指数上精确一致。N[X] 进一步精确投影至经形式核验的目标半环；平坦来源支持和元组级 which-lineage 则被单独报告为任务投影。因此，该纵向统一无需混淆半环同态与非半环视图即可成立。

修复后的结论不是减少了统一范围，而是第一次把代数投影与任务投影的边界划清，同时以完全独立的多项式 Oracle 重新证明了 Γ→N[X]。

## Evidence and boundaries

1. **P1/P2 preservation.** P1 remains 13/13 exact, with zero variable-, output-, coefficient-, exponent-, and canonical-polynomial mismatch. P2 retains physical-occurrence, evidence, environment, disposition, and operation-result witnesses. Equal N[X] therefore cannot reconstruct complete Gamma.

2. **Independent Native and Candidate.** Native reads the frozen relations and RA AST and uses only its independent polynomial oracle. Candidate reads a validated Core Snapshot and continues to use the frozen Candidate algebra. Native does not read Core, Candidate artifacts, expected answers, or comparison results; Candidate does not read Native artifacts.

3. **Coefficients, exponents, and identities.** W4 aggregates duplicate derivations as coefficient 2. W6 aggregates repeated use of one source identity as exponent 2. Variables derive from complete source identities rather than tuple values, so equal-valued distinct sources remain distinct.

4. **GeneratedOrigin recursion.** Candidate follows each GeneratedOrigin to its adjacent `prior_support_id`, multiplies jointly required inbound origins, and adds alternative producer occurrences for a logical output. Missing bridges, missing inputs, ambiguous producers, and cycles fail closed.

5. **Algebraic targets versus task projections.** Bag N, Boolean B, and PosBool(X) pass the frozen formal checks and exact comparisons. Flat source support, Vars(N[X]), and existing tuple-level which-lineage are exact task projections; they are not jointly relabeled as semiring homomorphic images.

6. **Lower strictness.** Coefficient, balanced-bag, and exponent constructions show that equality in each lower domain does not imply N[X] equality. A joint exponent witness maps `x` and `x^2` to equal lower results while the N[X] polynomials remain different.

7. **Database which-lineage.** For every declared output, `Vars` extracts variables from N[X], maps them back to frozen source identities, and agrees with both the existing Database native result and its Core Candidate result. This is a tuple-level task projection, not a new Which(X) semiring claim.

8. **Report consistency.** The report generator recomputes all displayed counts from machine artifacts. Wrong output, term, observation, test, negative-control, or manifest counts block the consistency gate; no artifact is repaired to accommodate report text.

9. **Core remains unchanged.** The Core runtime, protocol, compatibility code, Core tests, and existing Database experiment retain the frozen PR #19 tree hashes. No semiring-specific field was added to Core, and all changes remain within this experiment directory.

10. **Unification of unification.** Complete generation facts exactly and strictly yield N[X]; N[X] exactly yields three verified algebraic targets and the separately classified task projections. This establishes the vertical hierarchy without requiring every lower view to be a commutative semiring.

11. **W3C PROV boundary.** `nx_w3c_relation_scope.json` remains `NOT_EVALUATED`. Direct derivability, equivalence, hierarchy, or incomparability between W3C PROV and N[X] requires a separate paired experiment.

12. **Scope.** Included operators are selection, projection, rename, union, natural/equi-join, and self-join over finite base relations, including identity-distinct equal values and multistage composition. Difference, negation, antijoin, NOT EXISTS, aggregation, recursion/Datalog, windows, NULL three-valued logic, outer joins, arbitrary SQL, probabilistic evaluation, arbitrary semirings, and universal query equivalence are excluded.

## Falsification and reproducibility

- The original 45 negative controls and the 25 added formal-semantics/independence controls are counted separately; all 70 execute exactly once, use unique mutation fingerprints, fail closed with exact reason codes, and perform no automatic repair.
- Native, Candidate, direct lower K, N[X]-derived lower evaluation, comparison, and report-statistics processes record imports, called symbols, file opens, subprocesses, sockets, artifact reads, and profile reads.
- Two complete hardening executions regenerate the scientific evidence and all 50 bottom-up tests. Their scientific reports and test results are byte-identical, with no scientific field excluded.
- The final artifact manifest records and independently rehashes every experiment file except its unavoidable self-reference.

Provenance semirings are not replaced by the complete model. Their success is explained: they retain exactly the algebraic derivation facts needed for annotated positive relational evaluation. Their boundary is also explained: they intentionally erase occurrence-specific generation facts that are unnecessary for that task.

完整模型并不取代 Provenance Semiring，而是解释它为何成功：它精确保留带注释正关系求值所需的代数推导事实；同时也解释其边界，因为它有意舍弃该任务不需要的发生级生成事实。
