# Source manifest

This companion repository preserves experimental files imported from frozen
commits of `wind342/source-information-continuity` together with subsequent
experiments developed and frozen directly in this repository.

| Imported scope | Source commit | Relevant Git tree identity |
|---|---|---|
| Core v3, GFG implementation, five-profile projection, nanoGPT training–learning experiments, direct prediction and frozen inference | `0b03a0b65b24dfce00e6f70610efa6b566c6bd3b` | GF-P01 tree `1b375ce37f43d98b753034c68268582c366381c5`; TL-P01 tree `ab25eaf2b3b304bb7f6cea52ebc4a4037043b791` |
| Provenance-semiring projection and formal-semantics hardening | `17768eb8921ecbb28c0cc2ce7d5e013b1ce396e8` | Frozen source tree `469db7011ce90ab62f302f1b06aab42fd9716042`; portable companion tree `039968664ad24f2b3e24e0235f45dc30c0f945fb` |
| Reinforcement-learning feedback closure | `d186816eb9d972577208d1273d7d92247a02056e` | RL-E01 tree `6616e2399cb66c05f9eb13b865929187bda3bbe3` |
| GFG-guided long-delay temporal-credit discovery | `5ba05f6adfa00e76baac80e7c62832e2281a4e54` | RL-E02 source tree `a80761308c23e49920ec312e002751b0ebe8c082`; portable companion tree `4fabf634ac16984f08d4f065c14f6ebef6a2d959` |
| Long-chain temporal credit and recursive optimization of credit discovery | companion commit `b06539f` | RL-E03 tree `bd037254fff9e5ea7ba24d4efb4e4bbdfd07e76b` |
| Stochastic long-chain temporal credit | companion commit `bd5dca0` | RL-E04 tree `d0b0c6dd6ff9601d20e97165d350d8fc4aee6811` |

The GF-P01 tree at source commit `0b03a0b...` is byte-identical to the hardened
replay tree at commit `c407076c73b7f8de9679d7cd59f885b1bd64e69a`.

Files were imported by Git object identity rather than copied from mutable
runtime outputs. RL-E02 received portability-only documentation normalization
after import: machine-local artifact paths were replaced with logical external
artifact locators; its frozen executable source hashes, result values and
artifact hashes were unchanged. RL-E03 and RL-E04 retain their frozen contracts,
compact formal artifacts and independent checkers at the tree identities above.
No manuscript document, private data directory, API token, credential or
machine-specific path is intentionally included.

The portable GF-P02 tree differs from the frozen source tree only by omission
of the two third-party author-version PDFs listed in
`THIRD_PARTY_AUTHORITIES.md`. A path-by-path Git-object comparison found no
other added, removed or modified GF-P02 file. The citations, persistent links
and audited hashes remain public; the papers themselves are not redistributed.

