# Experiment Report: Provenance Semirings as a Strict Hierarchical Projection

## Outcome

Within the frozen positive relational-algebra profile, the machine evidence supports the exact hierarchy

```text
Gamma_G(omega)
  -- exact strict projection --> N[X]
  -- exact algebraic projections --> {bag N, Boolean B, PosBool(X)}
  -- exact task projections --> {flat source support, Vars(N[X]), existing Database which-lineage}
```

The final machine status is `PROVENANCE_SEMIRING_AS_STRICT_HIERARCHICAL_PROJECTION_SUPPORTED`. This is a bounded construction over the declared finite fixtures and operators, not a claim about all provenance semirings, all database provenance, arbitrary SQL, or universal query equivalence.

The primary theoretical authority is Green, Karvounarakis, and Tannen, “Provenance Semirings,” PODS 2007, DOI 10.1145/1265530.1265535. The frozen author PDF has SHA-256 `74e092702db58518afeaf909e1d3380848165b2cb9ae75dc6822b04f66aa5be0`; Definitions 3.1/3.2 are on proceedings page 33, while Definition 4.1, Figure 5, and the homomorphism surface are on proceedings page 34.

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
| Negative controls | 45 |
| Passing bottom-up test areas | 35 |
| Manifested files | 93 |
| Manifested artifact files | 37 |
<!-- END MACHINE-GENERATED REPORT STATISTICS -->

## Required questions

1. **What does N[X] represent here?** It is the canonical natural-number polynomial annotation for each logical output value. Variables identify registered source tuples; multiplication records joint use in one derivation; addition records alternative derivations; coefficients preserve repeated identical monomials; exponents preserve repeated use of one source identity.

2. **How is N[X] recursively derived from GenerationBinding?** A registered source maps to its stable variable. A GeneratedOrigin must recurse to its `prior_support_id`. All inbound origins bound to one occurrence/support are multiplied. Terminal supports with one `logical_output_key` are added. Missing bridges, missing inputs, multiple producer occurrences for one support, and cycles fail closed.

3. **Why multiplication inside an occurrence and addition between occurrences?** Inputs jointly required by a single generation occurrence form a conjunction and therefore a semiring product. Distinct producer occurrences that yield the same logical value are alternatives and therefore a semiring sum. This is the frozen positive-RA/K-relation interpretation.

4. **How does a duplicate derivation form a coefficient?** W4 executes two distinct branches that both produce the same monomial. Canonical addition merges them as coefficient 2 rather than silently deduplicating them.

5. **How does self-join form an exponent?** W6 binds the same source identity through two distinct join slots in one occurrence. Multiplication combines the two equal factors as exponent 2.

6. **How are duplicate-valued identities preserved?** Variables are derived from the complete `SourceInformationRecord.source_identity`, never from tuple values. W5 therefore retains two variables even though the ordinary values are identical.

7. **How does GeneratedOrigin support multistage composition?** Every later-stage input that was generated earlier is represented by a GeneratedOrigin pointing only to the adjacent prior support. Candidate recursion follows that bridge. No original-source-to-final-output shortcut is created.

8. **Where does Native N[X] come from?** The Native process reads only the frozen base relations and RA AST, then independently evaluates selection, projection, rename, union, equi-join, and self-join with N[X] addition and multiplication. It does not read Core, Snapshot, Candidate, existing lineage, or expected answers.

9. **Does Candidate read only ValidatedSnapshot?** Yes. Its semantic inputs are a matching validated Snapshot, its validation result, the frozen N[X] profile/crosswalk, and structural canonicalization. Static and runtime audits report Candidate Native reads = 0 and Candidate fixture/AST reads = 0.

10. **Are Native and Candidate exact term by term?** Yes. Every frozen case, source-variable identity, output identity, coefficient, and exponent is compared in the machine-generated statistics block and the v2 exact-comparison artifact. All compared fields have zero mismatch and zero repair.

11. **Why can equal N[X] not reconstruct complete Gamma?** Ten real executions form five paired fibers with equal ordinary/Native/Candidate N[X] but different valid Snapshot identities. The pairs vary physical occurrence structure, evidence, environment, disposition, and operation result, witnessing non-injectivity within the frozen execution domain.

12. **Why is ExplicitDisposition strictness evidence?** W1 retains the excluded source and a valid selection disposition in Gamma, while that source contributes no positive output term. Changing disposition facts can change Gamma without changing N[X].

13. **Which database semantics are lower projections?** Direct K-relation execution and N[X] homomorphisms agree exactly for bag naturals, Boolean existence, and canonical positive-Boolean lineage. Flat source support is separately exact as a non-semiring task projection. The frozen existing Database which-lineage is connected through `Vars(N[X])` as another task projection.

14. **How is which-lineage derived from N[X]?** `Vars` takes the union of variables appearing in the polynomial and maps each variable back to its frozen source identity. For every declared existing backward-lineage output, the result equals both the existing native oracle and existing Core Candidate source-tuple set.

15. **Which lower projections are strict?** Real coefficient, balanced-bag, and exponent constructions show that equality in each evaluated algebraic or task domain does not imply N[X] equality. Coefficients, source alternatives, and exponents are independently forgotten by different projections.

16. **Can the joint lower projections still fail to reconstruct N[X]?** Yes. The exponent construction maps `x` and `x^2` to the same bag, Boolean, flat-support, and PosBool results while retaining different N[X] polynomials. The joint status is therefore supported, not assumed.

17. **Does this experiment prove a direct relationship between W3C PROV and N[X]?** No. `nx_w3c_relation_scope.json` remains `NOT_EVALUATED`. They are treated as different projections from complete facts; direct derivability, equivalence, hierarchy, or incomparability requires a separate paired experiment.

18. **Was Core changed for semiring support?** No. Core runtime, protocol, compatibility code, and Core tests retain their frozen tree hashes. All new implementation and evidence are confined to this experiment directory, and no semiring-specific Core field was added.

19. **Did ordinary database results change?** No. The write-only collector cannot return control information, and capture-on/off canonical bytes are equal in all 13 cases across both complete scientific runs.

20. **What is the exact scope?** Included: finite base relations, selection, projection, rename, union, natural/equi-join, self-join, identity-distinct equal values, and multistage composition. Excluded: difference, negation, antijoin, NOT EXISTS, aggregation, recursion/Datalog, windows, NULL three-valued logic, outer join, arbitrary SQL, probabilistic evaluation, arbitrary semirings, universal query equivalence, all DBMSs, and a re-proof of the complete provenance literature.

## Falsification and reproducibility

- P1 exactness, field coverage, P2 strictness, classified lower comparisons, negative controls, tests, and manifest sizes are reported only in the machine-generated statistics block above.
- Existing Database native/Core records and every declared backward which row remain exact against `Vars(N[X])`.
- Lower strictness retains both per-domain witnesses and a joint witness.
- Every prohibited isolation target remains zero; authorized read-only Core implementation-identity checks are reported separately from unauthorized subprocesses.
- Complete repeated-run evidence includes every scientific field required by the hardening protocol and independently rehashes all materialized components.

Provenance semirings are not replaced by the complete model. Their success is explained: they retain exactly the algebraic derivation facts needed for annotated positive relational evaluation. Their boundary is also explained: they intentionally erase occurrence-specific generation facts that are unnecessary for that task.

完整模型并不取代 Provenance Semiring，而是解释了它为何成功：它精确保留带注释正关系求值所需的代数推导事实；同时也解释了它的边界，因为它有意舍弃了该任务不需要的发生级生成事实。

完整生成事实不仅统一了多个数据库 provenance 机制；它还把统一这些机制的 N[X] 理论本身，确定为完整发生级事实的一个严格投影。

Complete generation facts form a higher-level occurrence-specific structure from which the universal N[X] provenance polynomial is exactly derivable. The polynomial is strict because equal N[X] results can arise from unequal generation occurrences. The established database provenance semantics are in turn exact lower projections of N[X]. Thus a framework that already unifies multiple database provenance semantics is itself a strict projection of complete generation facts.
