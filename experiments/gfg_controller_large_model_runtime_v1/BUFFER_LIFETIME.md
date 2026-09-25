# Buffer lifetimes (single GPU, FP32)

Let P be parameter count, N target count, V vocabulary, H hidden width; target
tile B=32 and competitor tile C=8 in this exact reference-order implementation.

| Logical object | Physical allocation | Lifetime / alias rule |
|---|---:|---|
| Receiving parameters | 4P bytes, existing model | Never overwritten before commit; temporarily detached from Parameter.data only during candidate optimizer call |
| Gradients | 4P, existing training | Backward through proposal; standard optimizer zero_grad lifetime |
| Adam first/second moments | 8P, existing optimizer | Persistent native recurrence, never scaled by alpha |
| Candidate parameters | 4P extra | Copied from receiving state; native optimizer writes here; retained until commit |
| Actual delta | 4P extra | Candidate minus receiving state; used for directional response; reused as scaled endpoint ONLY after adjudication |
| Directional activations h/h'/h'' | Architecture/sequence dependent | Certified global nested JVP; experimental blockwise implementation separately checked |
| Boundary differences | O(B*C*H), CPU float64 | One target/competitor tile at a time |
| Four minimum margins | O(4N), CPU float64 | Across all competitor tiles until global selector |
| Full pre-logits | O(NV), CPU float64 | Still retained for reference-compatible order; not fully vocabulary-streamed |
| Head weights and delta | O(VH), CPU | Still transferred; no tensor parallel certification |
| Full coordinates | O(NV) | AUDIT mode only |
| Parameter snapshots | CPU copies | External AUDIT/test harness only, no resident GPU audit buffer |

Two resident control slabs replace five prior slabs: 8P versus 20P bytes, a 60%
reduction in that category. This is NOT a claim of 60% lower total GPU peak or a
speed claim. Parameter+gradient+Adam+two slabs total 24P bytes before activations,
AD temporaries, CUDA workspace, allocator reserve and Python/framework overhead.

One GPU/rank only. No FSDP, tensor parallel, CPU optimizer offload or communication
implementation is claimed. Moving .data does not move optimizer-state ownership:
the same Parameter object remains the key. Exceptions restore pointers but may
leave advanced moments, therefore failed steps must terminate the run.
