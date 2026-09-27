"""Only if D diverges: replay the common prefix and compare directional maps at one state."""
from .common import *
from . import reference,optimized
from .predictor import Predictor
from .preflight import install,verify_inputs

def main():
    access=install();setup();verify_inputs();X,y,control=load();labels=y.cpu().numpy();reports=[]
    for budget in (210,630):
        identity=read(ROOT/f'stage2/budget{budget}/IDENTITY.json')
        if identity['status']=='PASS':continue
        target=identity['first_divergence']['step'];state=torch.load(OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt',weights_only=True)
        head=model();head.load_state_dict(state['model']);shadow=copy.deepcopy(head);opt=optimizer(shadow);opt.load_state_dict(state['optimizer'])
        ids=np.flatnonzero((labels<7)&control);current=np.flatnonzero((labels//7==1)&~control)
        bank=Bank(X[ids],y[ids],ids,14,Meter());previous=bank.exact(params(head));K=bank.counts(previous)
        predictor=Predictor(bank,read(AUTH/'NUMERICS.json')['predictor_dense_flops_per256']);checks=[]
        for step,batch in enumerate(schedule(current,2,budget),1):
            before=params(head);native_step(head,shadow,opt,X[batch],y[batch],14);native=params(shadow)
            predicted=predictor.predict(before,native);actual=bank.exact(native)
            if step==target:
                method=optimized.UniqueGeometry.apply
                def observe(self,d):
                    a=method(self,d);b=reference.Geometry.apply(self,d)
                    checks.append(dict(pairs=len(self.rows),unique=len(self.H_unique),byte_identical=a.tobytes()==b.tobytes(),
                        max_abs_error=float(np.max(np.abs(a-b))),candidate_sha256=ahash(a),reference_sha256=ahash(b)))
                    return a
                optimized.UniqueGeometry.apply=observe
                try:optimized.project(before,native,bank,K,prediction=predicted,native_endpoint=actual,before_endpoint=previous,unique_apply=True)
                finally:optimized.UniqueGeometry.apply=method
                break
            executed,_=reference.project(before,native,bank,K,prediction=predicted,native_endpoint=actual)
            commit(head,executed);previous=bank.exact(params(head))
        reports.append(dict(budget=budget,first_step=target,same_state_apply_comparisons=checks,
            shape_change_explanation='Same inputs and algebra; unique row GEMM changes row count, allowing BLAS reduction/kernel differences. Attribution is supported only if same-state apply differs.',
            first_divergence=identity['first_divergence'],final_checkpoint_state_identical=identity['final_checkpoint_state_byte_identical']))
    write(ROOT/'D_DIAGNOSTIC.json',dict(reports=reports,access=access.report(),no_training_changes=True));print('D diagnostic stored',flush=True)

if __name__=='__main__':main()
