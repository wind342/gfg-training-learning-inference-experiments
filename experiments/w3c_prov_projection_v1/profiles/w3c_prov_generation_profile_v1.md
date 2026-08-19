# W3C PROV generation profile v1

Profile ID: `w3c-prov-generation-profile-v1`. Status: frozen before implementation.

The profile includes PROV Entity, Activity, SoftwareAgent, identified Usage, identified Generation, identified expanded Derivation, identified Association, `prov:type`, `prov:role`, the closed `ex:` attribute allowlist in the JSON profile, deterministic PROV-N, and the corresponding qualified PROV-O representation.

Every Core `GenerationBinding` becomes its own Usage and expanded Derivation. All bindings for the same generated Entity and Activity share one Generation. The base role and a deterministic ordinal preserve legal multiplicity; bindings are never reconstructed as a Cartesian product.

Bundle, provenance of provenance, Collection, Dictionary, Alternate, Specialization, Delegation, Communication, Start, End, temporal ordering, Invalidation, arbitrary extensions, arbitrary RDF entailment, the complete PROV ecosystem, and other serializations are `EXCLUDED` or `NOT_EVALUATED`.

Only declared stable domain fields may enter `ex:` attributes. Complete Core objects, tables, Snapshots, payloads, Evidence, operation closure, environments, opaque blobs, and mechanical Core-ID copies are prohibited.
