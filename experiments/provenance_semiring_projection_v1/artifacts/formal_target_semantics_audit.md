# Formal target semantics audit

Machine status: `FORMAL_TARGET_SEMANTICS_CLASSIFIED`.

| Domain | Classification | 0 | 1 | Addition | Multiplication | Zero annihilates | Algebraic target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bag_naturals` | `COMMUTATIVE_SEMIRING_TARGET` | `0` | `1` | natural addition | natural multiplication | True | True |
| `boolean` | `COMMUTATIVE_SEMIRING_TARGET` | `false` | `true` | logical OR | logical AND | True | True |
| `positive_boolean_lineage` | `SEMIRING_QUOTIENT_OR_HOMOMORPHIC_IMAGE` | `{'terms': []}` | `{'terms': [[]]}` | OR with canonical absorption | AND with canonical absorption | True | True |
| `flat_source_support_view` | `PARTIAL_NONZERO_SUPPORT_VIEW` | `{'variables': []}` | `{'variables': []}` | set union | set union | False | False |

The former `why_powerset` is now `flat_source_support_view`. Its empty set is both additive and multiplicative identity, and it fails multiplicative-zero annihilation for every nonempty value. It is therefore an exact task projection on frozen nonzero output support, not a complete commutative-semiring homomorphism target.

`Which(X)`, `Trio(X)`, and witness-basis Why provenance remain `NOT_EVALUATED`; no algebraic structure is invented from names alone. Existing tuple-level which-lineage is classified separately as `NON_SEMIRING_TASK_PROJECTION`.

Authority locations: Green-Karvounarakis-Tannen, PODS 2007, proceedings pages 33-34; Buneman-Khanna-Tan, ICDT 2001, paper pages 8-10.
