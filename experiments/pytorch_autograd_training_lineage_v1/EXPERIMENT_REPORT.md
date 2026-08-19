# PyTorch Autograd Projection and Bidirectional Training-Update Lineage v1

Final status: `PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED` (22/22 gates).

Scope: **the frozen PyTorch Autograd dependency profile over the declared deterministic workloads**. This is not a claim about every PyTorch program, operator, device, optimizer, checkpoint implementation, compiled graph, distributed execution, or neural causal attribution.

## Result

- Official PyTorch `2.13.0+cpu` executed on CPU with float64, one intra-op and one inter-op thread, seed 424242, and deterministic algorithms enabled.
- Candidate and Native graphs are exact for 5/5 workloads: 33 native nodes, 33 ordered edge slots, and zero node, edge, slot, shared-node, leaf, root, or multiplicity mismatches.
- Three independently constructed pairs have different complete generation facts but equal Native Autograd graphs.
- 20/20 standard forward queries and 29/29 standard reverse queries exactly match the independent receipt reference; FP, FN, role, occurrence, path, and multiplicity mismatches are all zero. The checkpoint forward and reverse queries also match.
- All 32 negative controls fail closed with unique mutation fingerprints and honest depths.
- Two complete scientific reports have the same SHA-256 `270d8901041778ee2a60cb493cbf5b99591db032213c32a7e12462099d5b0e17`.
- Tests passed twice: 34 experiment tests and 24 unchanged Core tests per execution.

## Direct answers to the required questions

1. **Where does the Native graph come from?** From actual eager PyTorch execution. After the real forward loss exists and before backward, the independent observer starts at `loss.grad_fn` and traverses public `Node.name()` and ordered `next_functions`.
2. **Does the Candidate read only a ValidatedSnapshot?** Yes. Its only authority inputs are a `ValidatedSnapshot`, matching `SnapshotValidation`, frozen profile, frozen crosswalk, and structural canonicalizer. Static dependency audit found no `grad_fn`, `next_functions`, Native artifact, receipt, object-ID, or reference read.
3. **Are Native and Candidate exact node-by-node and edge-by-edge?** Yes within scope: 33/33 nodes and 33/33 ordered edge slots, with canonical bytes exact for every workload.
4. **Which Γ facts does Autograd select?** The frozen graph selects backward Function node types, reachable differentiable leaves, ordered dependency slots, `output_nr`, shared-node identity, multiplicity, and root topology needed for reverse AD.
5. **Which occurrence facts does it omit jointly?** Training-sample identity, evidence/environment, concrete forward versus recomputation occurrence identity, external checkpoint state version, tensor outcomes, gradient values, optimizer occurrence/state, parameter semantic versions, and explicit dispositions.
6. **Does graph equality imply equal training occurrence?** No. All three strictness pairs have equal graphs and different validated Γ snapshots.
7. **Can a training sample be followed forward to actual parameter updates?** Yes. Every declared training source query returns its actual activations, loss, gradient, SGD update, optimizer state, parameter version, roles, ordinals, and path multiplicity.
8. **Can a parameter update be traced backward to actual training sources?** Yes. Every declared parameter-after support was reverse-queried through optimizer, gradient, backward, optional recomputation, forward, and source records.
9. **Are forward, recomputation, backward, and optimizer distinct?** Yes. They use different content-addressed occurrences and stages; the checkpoint fixture records original and recomputation operations separately.
10. **Is there any fabricated sample→parameter shortcut?** No. The direct-shortcut count is zero; paths cross actual GeneratedOrigin stage bridges.
11. **Where does checkpoint divergence occur?** At the backward recomputation occurrence: original forward reads external scale 1, while the divergent recomputation reads a replacement float64 tensor with scale 2.
12. **Is the wrong gradient finite and undetected by the default check?** Yes. The gradient is finite, backward raises no exception, and `determinism_check="default"` does not reject the same-shape/dtype/device state change.
13. **Can the Native graph distinguish stable and divergent runs?** No. Their canonical Native graph bytes are exactly equal.
14. **How does Γ localize scale=2?** Reverse lineage from the divergent parameter version reaches the registered recomputation-scale source through the concrete recomputation activation and gradient-production occurrence.
15. **How does Γ list affected gradients and parameter versions?** Forward lineage from the scale-2 source returns the recomputed activation, parameter gradient, optimizer state after, and parameter-after support with complete paths.
16. **Are zero gradient and nonparticipation distinct?** Yes. `p_zero` has a real path and a zero-valued gradient support; `p_unused.grad is None`, has no participation path, and has an explicit `UNUSED_IN_THIS_LOSS_OCCURRENCE` disposition.
17. **Does capture change training output?** No. For every standard workload, output-only, Native-only, Core-only, and dual ordinary bytes are identical. Native-only equals dual-Native, Core-only equals dual-Candidate, and dual Native equals dual Candidate. Output-only deliberately does not traverse a graph; topology equivalence follows through the single shared training entrypoint and those transitive observations.
18. **Did Core change for PyTorch?** No. Core runtime, protocol/schema, compat/v2, and tests/core tree SHAs equal the frozen base; PyTorch-specific Core field count is zero.
19. **What scope supports the conclusion?** CPU float64 eager reverse-mode AD for the seven frozen standard wrappers and five declared workloads, plus official non-reentrant checkpoint and SGD fixtures under the frozen deterministic settings.
20. **What was not evaluated?** CUDA/XPU/MPS, AMP, distributed execution, `torch.compile`, sparse tensors, forward-mode AD, higher-order gradients, complex alias/view behavior, in-place operations, custom C++ operators, non-SGD optimizers, reentrant checkpoint, and neural causal attribution.

## Scientific statement

Within the frozen PyTorch profile, the Autograd computational graph is an exact strict projection of complete training-generation facts. The complete model additionally provides authoritative bidirectional lineage between training sources and parameter versions across forward, checkpoint recomputation, backward and optimizer occurrences. In a documented activation-checkpoint divergence construction, the native Autograd graph remains unchanged while recomputation reads a different external state; generation-fact queries identify the exact divergent occurrence and all downstream gradients and parameter updates affected by it.

在冻结的 PyTorch profile 范围内，Autograd 计算图是完整训练生成事实的精确严格投影。完整模型进一步建立训练来源与参数版本之间跨前向、checkpoint 重计算、反向和优化器发生的权威双向追踪。在 activation checkpoint 状态偏差构造中，原生 Autograd 图保持不变，而重计算读取了不同的外部状态；生成事实查询能够精确定位发生偏差的具体重计算，并返回其影响的全部梯度与参数更新。

**Autograd 交付的是求导所需的计算图；完整生成事实交付的是一次模型更新究竟如何发生。**


## Gradient-dependency oracle hardening

Final hardening status: `PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_EVIDENCE_HARDENING_SUPPORTED` (20/20 gates).

The v1 Core capture and receipt reference separately encoded semantically overlapping local reverse rules. That does not overturn the exact Autograd projection, checkpoint graph equality, finite gradient divergence, parameter-update divergence, or v1 lineage results; it limited how independently the gradient-value dependency paths had been validated.

The v2 evidence removes that shared semantic assumption from the reference path. A native-only PyTorch runner uses public `Node.name()`, ordered `Node.next_functions`, node pre/post hooks, leaf tensor hooks, and `saved_tensors_hooks` to observe real backward execution and saved-tensor retrieval. It then performs 75 predetermined single-token interventions and 72 registered-source replay interventions. No Core snapshot, Candidate relation, receipt dependency answer, operation-specific derivative table, or persistent Python object identity is available to that oracle process.

The independently observed native relation has 29 relations; Core has 29. False positives, false negatives, source-identity mismatches, target-gradient mismatches, topology mismatches, duplicate-identity collapses, multiplicity mismatches, missing witnesses, unsupported Core dependencies, and unrepresented native dependencies are all zero. Every declared Core `gradient_value_dependency` therefore has an actual saved-tensor intervention witness, a registered-source replay witness, or both.

For the checkpoint flagship case, the registered recomputation source `source:external:scale:recomputation` with value 2 is independently replayed and changes `step_0:gradient:parameter:p`; real backward node execution and recomputation saved-tensor activity are present. The unchanged stable/divergent native graph is not used as the dependency answer. The v2 forward and reverse comparisons remain exact (34/34 and 41/41).

Candidate/Core and native oracle run in separate processes and do not read each other's evidence. The v2 receipt reference supplies only forward, backward-boundary, and optimizer edges, accepting gradient-value edges from the native oracle; its operation-specific gradient-rule count is zero. The original 32 negative controls remain preserved, and 20 new controls execute once each, fail closed, have unique fingerprints, and perform no automatic repair.

Two complete hardening runs produced byte-identical scientific artifacts and reports with no excluded scientific fields. Each test execution passed 56 experiment tests and 24 unchanged Core tests.

This profile does not evaluate semantic importance, causal contribution magnitude, sample importance, arbitrary PyTorch operators, CUDA or other accelerators, mixed precision, distributed or compiled execution, higher-order/forward-mode AD, in-place and complex alias/view behavior, custom C++ operators, non-SGD optimizers, or reentrant checkpoint. Interventions establish value dependency only for the frozen deterministic workloads and perturbations.

“The original experiment derived gradient-value dependencies in both the Core capture and receipt reference through separately implemented but semantically shared local reverse rules. The hardened experiment removes that shared semantic assumption from the reference path. A native PyTorch oracle observes actual backward-node execution and saved-tensor retrieval, then independently intervenes on saved tensors and registered sources. Within the frozen profile, the resulting native dependency relation is exactly equal to the gradient-value dependency relation delivered by complete generation facts.”

“原实验的Core捕获与receipt参考路径分别实现了语义相同的局部反向规则，因此仍可能共享同类错误。加固实验从参考路径中移除了这一共同假设：原生PyTorch oracle直接观测实际backward节点执行与保存张量取回，并分别对保存张量和注册来源实施独立干预。在冻结profile范围内，由此得到的原生梯度依赖关系与完整生成事实交付的gradient-value dependency关系精确一致。”

**梯度来源关系不再由两套相同规则相互证明，而由真实PyTorch backward发生独立证明。**
