"""Full fixed-trajectory replay with read-only per-sweep functional observations."""
from .common import *
from .preflight import install,verify
from .solver import observe_solver
from experiments.gfg_banking77_cutter_dedup_task2_v1 import optimized as cutter
from experiments.gfg_banking77_cutter_dedup_task2_v1 import audit as independent
from experiments.gfg_banking77_cutter_dedup_task2_v1.predictor import Predictor
from experiments.gfg_banking77_cutter_dedup_task2_v1.compare import state_equal

def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]

class Observer:
    def __init__(self,bank,K,output):
        self.bank=bank;self.K=K;self.output=output;self.plain=cutter.ball_hildreth
        self.total_probes=0;self.independent_checks=0;self.total_wall=0.;self.audit_wall=0.
    def step(self,index,reference,native):
        self.index=index;self.reference=reference;self.native=native;self.native64=cpu(native)
        self.rounds=[];self.outer=0;self.first_candidate=None;self.first_location=None
    def geometry(self,geometry,current):
        self.outer+=1;self.geo=geometry;self.current=current
        self.record=self.reference['decision']['rounds'][self.outer-1]
        assert nhash(current)==self.record['current_sha256'] and ahash(geometry.G)==self.record['gram_sha256']
        self.offset=None if self.outer==1 else tuple(n-p for n,p in zip(self.native64,current))
        self.independent_factors=None
    def check_independently(self,coef,dual,rounded,z):
        start=time.perf_counter()
        if self.independent_factors is None:
            self.independent_factors=independent.factors(self.bank.H,self.bank.labels,self.current,self.geo.ordered,
                # Independent factors use the exact original local endpoint, not a probe endpoint.
                self.local_z)
            assert ahash(self.independent_factors['G'])==ahash(self.geo.G)
        offset=tuple(n-p for n,p in zip(self.native64,self.current))
        other,_,_=independent.rebuild(self.independent_factors,coef,dual,offset,self.current)
        assert nhash(other)==nhash(rounded)
        params=tuple(torch.from_numpy(a).cuda() for a in other)
        actual=independent.endpoint(params,self.bank.x,self.bank.n)
        assert ahash(actual)==ahash(z)
        assert independent.count(actual,self.bank.labels).tolist()==self.bank.counts(z).tolist()
        self.independent_checks+=1;self.audit_wall+=time.perf_counter()-start
    def probe(self,sweep,coef,dual,residual,length,change,core_seconds):
        stamp=time.perf_counter();unchanged=dual.tobytes();geo=self.geo
        # Timing contains the full necessary candidate check, excluding logs/hashes/audit.
        begin=time.perf_counter();corr=geo.reconstruct(dual)
        d=corr if self.offset is None else tuple(coef*u+v for u,v in zip(self.offset,corr))
        next64=tuple(p+v for p,v in zip(self.current,d));recon_end=time.perf_counter()
        rounded=tuple(v.astype(np.float32) for v in next64)
        candidate=tuple(torch.from_numpy(v).to(t.device) for v,t in zip(rounded,self.native));sync();material_end=time.perf_counter()
        z=self.bank.exact(candidate);counts=self.bank.counts(z);safe=bool(np.all(counts>=self.K));finish=time.perf_counter()
        check=finish-begin;row=dict(step=self.index,outer=self.outer,sweep=sweep,counts=counts.tolist(),safe=safe,
            solver_core_seconds=core_seconds,check_seconds=check,reconstruction_seconds=recon_end-begin,
            materialization_seconds=material_end-recon_end,endpoint_counts_seconds=finish-material_end,
            halfspace_violation=residual,cycle_change=change,ball_norm=length)
        self.observations.append(row);self.total_probes+=1
        first=safe and self.first_safe is None
        if first:
            self.first_safe=sweep;self.first_safe_counts=counts.tolist();self.first_safe_sha=nhash(rounded)
            if self.first_candidate is None:
                self.first_candidate=tuple(v.copy() for v in rounded);self.first_location=(self.outer,sweep)
        terminal=sweep==self.record['solver']['sweeps']
        if first or terminal:self.check_independently(coef,dual,rounded,z)
        if terminal:
            assert nhash(rounded)==self.record['candidate_sha256']
            assert ahash(z)==self.record['endpoint_sha256'] and counts.tolist()==self.record['endpoint_counts']
        assert dual.tobytes()==unchanged
        self.output.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        self.total_wall+=time.perf_counter()-stamp
    def solve(self,G,b,Ad0,d0norm2,rho,limit=200):
        signature=(ahash(G),ahash(b),ahash(Ad0))
        assert signature[1]==self.record['rhs_sha256'] and rho==self.record['rho']
        self.observations=[];self.first_safe=None;self.first_safe_counts=None;self.first_safe_sha=None
        start=time.perf_counter();plain=self.plain(G,b,Ad0,d0norm2,rho,limit);plain_seconds=time.perf_counter()-start
        result=observe_solver(G,b,Ad0,d0norm2,rho,limit,observe=self.probe)
        coef,dual,info=result
        assert coef==plain[0] and dual.tobytes()==plain[1].tobytes() and info==plain[2]
        assert coef==self.record['coef'] and ahash(dual)==self.record['dual_sha256'] and info==self.record['solver']
        assert signature==(ahash(G),ahash(b),ahash(Ad0))
        rows=self.observations;assert len(rows)==info['sweeps']
        end=self.first_safe or len(rows);prefix=rows[:end];safe_sequence=[r['safe'] for r in rows]
        total_core=sum(r['solver_core_seconds'] for r in rows);prefix_core=sum(r['solver_core_seconds'] for r in prefix)
        self.rounds.append(dict(outer=self.outer,original_solver_status=info['status'],original_sweeps=info['sweeps'],
            first_safe_sweep=self.first_safe,first_safe_counts=self.first_safe_counts,first_safe_candidate_sha256=self.first_safe_sha,
            original_candidate_sha256=self.record['candidate_sha256'],last_safe=rows[-1]['safe'],safe_sweeps=sum(safe_sequence),
            safe_to_unsafe_transitions=sum(a and not b for a,b in zip(safe_sequence,safe_sequence[1:])),
            plain_solver_seconds=plain_seconds,observed_core_seconds=total_core,
            prefix_solver_seconds_estimate=plain_seconds*prefix_core/total_core,
            all_checks_seconds=sum(r['check_seconds'] for r in rows),prefix_checks_seconds=sum(r['check_seconds'] for r in prefix),
            original_terminal_check_seconds=rows[-1]['check_seconds'],
            reconstruction_seconds=sum(r['reconstruction_seconds'] for r in rows),
            materialization_seconds=sum(r['materialization_seconds'] for r in rows),
            endpoint_counts_seconds=sum(r['endpoint_counts_seconds'] for r in rows)))
        return result

def run(budget,X,y,control):
    folder=ROOT/f'budget{budget}';folder.mkdir(exist_ok=False);rows=records(AUTH/f'stage1/budget{budget}/STEPS.jsonl.gz')
    summary=read(AUTH/f'stage1/budget{budget}/CHECKS.json');state=torch.load(OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt',weights_only=True)
    head=model();head.load_state_dict(state['model']);shadow=copy.deepcopy(head);opt=optimizer(shadow);opt.load_state_dict(state['optimizer'])
    labels=y.cpu().numpy();ids=np.flatnonzero((labels<7)&control);current_ids=np.flatnonzero((labels//7==1)&~control)
    bank=Bank(X[ids],y[ids],ids,14,Meter());previous=bank.exact(params(head));K=bank.counts(previous);assert K.tolist()==summary['K']
    predictor=Predictor(bank,read(OLD/'NUMERICS.json')['predictor_dense_flops_per256'])
    original_geometry=cutter.Geometry;original_solver=cutter.ball_hildreth;updates=[];wall=time.perf_counter()
    with gzip.open(folder/'SWEEPS.jsonl.gz','xt',encoding='utf8') as out:
        observer=Observer(bank,K,out)
        class WatchingGeometry(original_geometry):
            def __init__(self,b,p,ordered,z):
                super().__init__(b,p,ordered,z);observer.geometry(self,p);observer.local_z=z.copy()
        cutter.Geometry=WatchingGeometry;cutter.ball_hildreth=observer.solve
        try:
            for step,batch in enumerate(schedule(current_ids,2,budget),1):
                ref=rows[step-1];assert batch.tolist()==ref['batch_ids'] and not control[batch].any()
                before=params(head);assert phash(before)==ref['before_sha256'];native_step(head,shadow,opt,X[batch],y[batch],14);native=params(shadow)
                assert phash(native)==ref['native_sha256'] and opt_hash(opt)==ref['moments_sha256'] and opt_steps(opt)==ref['optimizer_steps']
                predicted=predictor.predict(before,native);assert ahash(predicted)==ref['predictor']['predicted_sha256'] and predictor.hashes()==ref['predictor']['components_sha256']
                actual=bank.exact(native);assert ahash(actual)==ref['predictor']['actual_sha256'];observer.step(step,ref,native)
                executed,z,decision=cutter.project(before,native,bank,K,prediction=predicted,native_endpoint=actual,before_endpoint=previous)
                assert decision==ref['decision'];commit(head,executed)
                assert phash(params(head))==ref['executed_sha256'] and opt_hash(opt)==ref['moments_sha256']
                checked=independent.endpoint(params(head),bank.x,bank.n);counts=independent.count(checked,bank.labels)
                assert ahash(checked)==ahash(z)==ref['endpoint_sha256'] and counts.tolist()==ref['counts'] and np.all(counts>=K)
                local=observer.rounds;earliest=next((i for i,r in enumerate(local) if r['first_safe_sweep'] is not None),None)
                prefix=[] if earliest is None else local[:earliest+1]
                prefix_sweeps=sum(r['original_sweeps'] for r in prefix[:-1])+(prefix[-1]['first_safe_sweep'] if prefix else 0)
                original_seconds=sum(r['plain_solver_seconds'] for r in local)
                prefix_seconds=sum(r['plain_solver_seconds'] for r in prefix[:-1])+(prefix[-1]['prefix_solver_seconds_estimate'] if prefix else 0.)
                prefix_checks=sum(r['all_checks_seconds'] for r in prefix[:-1])+(prefix[-1]['prefix_checks_seconds'] if prefix else 0.)
                original_checks=sum(r['original_terminal_check_seconds'] for r in local)
                same=None;distance=None
                if observer.first_candidate is not None:
                    actual_params=tuple(p.detach().cpu().numpy() for p in executed);same=nhash(actual_params)==nhash(observer.first_candidate)
                    distance=float(np.sqrt(sum(np.sum((a.astype(float)-b.astype(float))**2) for a,b in zip(actual_params,observer.first_candidate))))
                if decision['triggered']:assert earliest is not None, 'Original executed safe endpoint must be observed'
                updates.append(dict(step=step,unsafe_native=decision['triggered'],original_sweeps=decision['inner_sweeps'],
                    first_safe_location=observer.first_location,first_safe_cumulative_sweeps=prefix_sweeps,
                    first_safe_same_as_executed=same,first_safe_vs_executed_l2=distance,rounds=local,
                    original_solver_seconds=original_seconds,prefix_solver_seconds_estimate=prefix_seconds,
                    prefix_check_seconds=prefix_checks,original_check_seconds=original_checks,
                    estimated_net_seconds_saved=original_seconds-prefix_seconds-(prefix_checks-original_checks),
                    executed_sha256=ref['executed_sha256'],moments_sha256=ref['moments_sha256'],floor_violations=0))
                previous=z.copy()
                if step%10==0:print('diagnostic',budget,step,'probes',observer.total_probes,'first-safe rounds',sum(r['first_safe_sweep'] is not None for u in updates for r in u['rounds']),flush=True)
        finally:cutter.Geometry=original_geometry;cutter.ball_hildreth=original_solver
    expected=torch.load(AUTH/f'stage1/budget{budget}/CHECKPOINT.pt',weights_only=True)
    assert state_equal(dict(model=head.state_dict(),optimizer=opt.state_dict(),task=2),expected)
    with gzip.open(folder/'UPDATES.jsonl.gz','xt',encoding='utf8') as out:
        for row in updates:out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    report=dict(status='PASS',budget=budget,steps=budget,observed_sweeps=observer.total_probes,
        independent_candidate_checks=observer.independent_checks,independent_committed_checks=budget,
        native_heads_moments_predictor_solver_decisions_executed_heads_identical=True,final_checkpoint_state_identical=True,
        floor_violations=0,old_ce_backward_examples=0,test_files_opened=0,diagnostic_wall=time.perf_counter()-wall,
        observer_wall=observer.total_wall,independent_candidate_audit_wall=observer.audit_wall,
        diagnostic_only=True,early_candidates_never_committed=True)
    write(folder/'CHECKS.json',report);print('Completed diagnostic',budget,observer.total_probes,'sweeps, original trajectory identical',flush=True)
    return report

def main():
    access=install();setup();verify();assert read(ROOT/'NUMERICS.json')['status']=='PASS'
    X,y,c=load();reports=[run(b,X,y,c) for b in (210,630)];verify()
    write(ROOT/'CHECKS.json',dict(status='PASS',reports=reports,access=access.report(),no_controller_change=True))

if __name__=='__main__':main()
