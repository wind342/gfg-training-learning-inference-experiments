# Fixed relation-capability floor: task8 only

Base115534a09cd554fcd75fd46dde711fdb98988519. One new arm from Final v1 task7
end head/AdamW moments. Frozen/eval BERT,unchanged80-class linear head,64 seen
classes,task8,seed2021,LR5e-5,batch16,210 updates,original order,one epoch,
clip1,fused AdamW(.9,.999),weight_decay0. No replay,Router,features or test tuning.
Original baseline is referenced,never rerun. Only this new directory may change.

Use exact original task8 protection bank:7833 old validation controls,56 relations,
plus80 new audit-only rows. K[r] is the correct count(margin>0) at task8 entry,
after expanding readout to64 classes. K remains fixed through all210 updates.
Old controls may exchange correctness,or drop from a temporary improvement to K.

At each step form the actual native AdamW head. If all native relation counts
meet K,execute unchanged. Otherwise,for each deficient relation select its d=K-C
currently wrong(margin<=0) samples with highest target-versus-strongest-competitor
margin; ties use ascending original dataset ID. Selection may include a previously
selected-but-still-wrong sample. Maintain selected IDs/dual constraints only within
this update,not across steps. For selected controls add all currently violating
target/competitor pairs using the original EPS1e-6 rule.

Reuse original coordinate(),materialize(),reserve formula,FP32 affine forward,
pair ordering and dual state. Each sweep updates existing active constraints,
materializes FP32 and checks ALL56 relation floors. Stop at the FIRST floor-safe
endpoint,even if an earlier selected recovery sample remains incorrect; selected
constraints are a search device,not a second acceptance criterion. If floors still
fail,select deficit samples again and add violating pairs on the accumulated
selected controls. Maximum100 sweeps TOTAL per scheduled update; no reset when
adding constraints. If still floor-unsafe,skip to the receiving head. Moments
advance once per scheduled step,even on skip,projection never changes them.

Reserve is frozen from receiving/native envelopes with original dimension k1539.
No modified tolerance,convergence rule,loss,optimizer,bank or LR. Correctness for
counts is strictly margin>0; selected pair targets remain EPS+original reserve.

Record fixed K,all per-relation counts/deficits,selections,active pairs,dual/primal
hashes,sweeps,skips,flips/gains,geometry,timing. Define same-step sample migration
as a relation with BOTH correct-to-wrong and wrong-to-correct in one executed
update; also report task-entry-to-end membership replacement separately. These
are correctness-set changes,not direct measurements of internal neural support.
Report non-migration count slack use separately: relation count falls versus
receiving but remains at/above K. No claim that all benefits are migration alone.

Preflight tests: swap accepted with fixed floor; K not ratcheted after temporary
gain; deterministic nearest/tie selection; recovery projection; hard global sweep
cap/skip; original primitives and initial state identity. Training/test/data
isolation remains identical; test only after training result is sealed. Independent
checker reconstructs all210 native updates,selections,Hildreth sweeps and margins,
verifies all11760 executed relation-step floors and optimizer continuity.

Interpretation: test improvement with control floors preserved supports the
combined relaxed protection policy,not population-wide old-capability guarantees,
unique causation or internal-support migration by itself. Failure of this greedy
selection does not prove infeasibility. No outcome-driven rerun or extra arm/task.
