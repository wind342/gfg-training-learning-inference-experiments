# Ownership and lifetime contract

| Object | Ownership / shape | Materialize | Release | Collective / alias |
|---|---|---|---|---|
| Receiving parameters | FSDP local shard; emulation slices one full model | caller model construction | caller teardown | emulation proxies alias full model |
| Gradients / native moments | local coordinates | backward / native optimizer | native optimizer lifecycle | real FSDP gradient collectives are framework-owned |
| Candidate | one local-parameter-sized buffer | controller creation; fill before native step | reused each proposal | no all-gather; bounded 65536-element swap scratch |
| Actual delta | one local-parameter-sized buffer | rounded native candidate minus receiving | reused as scaled endpoint after selection | never reconstruct candidate by inverse arithmetic |
| Receiving block + delta block | ONE stage full tensors | directional stage context enter | context exit; FSDP reshard | receiving unshard + delta DTensor.full_tensor; emulation concatenation |
| h, h', h'' | probe activation bank | embedding then chain rule per block | superseded by next block | no persistent model-size copy |
| Local readout / update rows | vocabulary shard | boundary decision | after decision | emulation gathers each local row interval; not a full-vocab logits gather |
| Target rows / delta rows / scores | probe-specific | boundary decision | after decision | payload is O(N*H), additional to minima, not O(N) alone |
| FP64 boundary differences | target tile x local competitor tile x hidden | tile loop | next tile / return | no communication |
| Per-target minima | 4 x N doubles | decision | selection / audit retention | global MIN then exact median; 72-byte host summary |

Recorded physical CUDA peaks in the tiny gate include the independent oracle and
audit snapshots, so they are not per-rank production-memory measurements. Local
ownership byte counts and peak block materialization are measured separately.
Emulation has no network traffic and no aggregate GPU-memory reduction. CPU Gloo
collective validation is separate and cannot certify CUDA/NCCL execution.
