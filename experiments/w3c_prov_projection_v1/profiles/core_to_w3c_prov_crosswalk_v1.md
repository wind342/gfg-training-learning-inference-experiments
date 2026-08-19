# Core to W3C PROV crosswalk v1

The machine-readable JSON file is authoritative for this experiment.

`SourceInformationRecord`, `PerceptualSupportRecord`, and `ExplicitDisposition` project to selected Entity subtypes. `GenerationOccurrence` projects to Activity. `GeneratorManifest` projects to SoftwareAgent and is associated with each Activity. A GeneratedOrigin bridge resolves to its prior support Entity and does not create an independent PROV Entity.

A binding `(origin, occurrence, outcome, role, ordinal)` projects to an identified Usage, the single identified Generation for `(outcome, occurrence)`, and an identified expanded Derivation that cites all five participants. The role is carried by `prov:role`; the ordinal is carried by `ex:relationOrdinal`. No implicit pairing is created.

Evidence, operation results, environment, Snapshot validation, relation-evidence closure, full payloads, registries, and undeclared context are outside the profile.
