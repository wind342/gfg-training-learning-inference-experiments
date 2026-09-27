"""Actual AdamW displacement, historical JVP/nested-JVP definition, fixed256."""
from .common import *

class Predictor:
    def __init__(self,bank,dense_flops):
        self.bank=bank;self.dense_flops=dense_flops;self.calls=0
        # Persistent result storage; forward AD owns its temporary intermediates.
        self.buffers=tuple(bank.x.new_empty((bank.n,77)) for _ in range(3))
    def predict(self,before,native):
        b=self.bank
        with b.meter.region('predictor',True),torch.no_grad():
            delta=tuple(n-p for n,p in zip(native,before))
            for k in range(0,b.padded,256):
                x=b.x[k:k+256];fn=lambda p:functional(p,x)
                f0,f1=torch.func.jvp(fn,(before,),(delta,))
                _,f2=torch.func.jvp(lambda p:torch.func.jvp(fn,(p,),(delta,))[1],(before,),(delta,))
                size=min(256,b.n-k)
                for dst,src in zip(self.buffers,(f0,f1,f2)):dst[k:k+size].copy_(src[:size])
            z=self.buffers[0]+self.buffers[1]+.5*self.buffers[2]
            result=z[:,:b.seen].cpu().numpy().copy()
        self.calls+=1;b.meter.calls['predictor']+=1
        b.meter.flops['predictor_dense']+=self.dense_flops*(b.padded//256)
        return result
    def hashes(self):return [phash((v,)) for v in self.buffers]

def measure(bank,previous,predicted,actual,K):
    a=bank.margins(actual);p=bank.margins(predicted);before=bank.margins(previous)>0
    ac=a>0;pc=p>0;ta=2*before.astype(int)+ac;tp=2*before.astype(int)+pc
    cc=np.zeros((2,2),dtype=int);tc=np.zeros((4,4),dtype=int)
    np.add.at(cc,(ac.astype(int),pc.astype(int)),1);np.add.at(tc,(ta,tp),1)
    C=bank.counts(actual);P=bank.counts(predicted);actualsafe=bool(np.all(C>=K));predsafe=bool(np.all(P>=K))
    return dict(rows=bank.n,margin_abs_sum=float(np.abs(a.astype(float)-p).sum()),margin_max_error=float(np.abs(a.astype(float)-p).max()),
        sign_matches=int((ac==pc).sum()),argmax_matches=int((actual.argmax(1)==predicted.argmax(1)).sum()),
        correctness_confusion=cc.tolist(),transition_confusion=tc.tolist(),predicted_counts=P.tolist(),actual_counts=C.tolist(),
        predicted_safe=predsafe,actual_safe=actualsafe,false_safe=predsafe and not actualsafe,false_unsafe=actualsafe and not predsafe,
        predicted_sha256=ahash(predicted),actual_sha256=ahash(actual))

def aggregate(rows):
    n=sum(r['rows'] for r in rows);cc=np.sum([r['correctness_confusion'] for r in rows],axis=0);tc=np.sum([r['transition_confusion'] for r in rows],axis=0)
    recall=[float(tc[i,i]/tc[i].sum()) if tc[i].sum() else None for i in range(4)]
    balanced=float(np.mean([cc[i,i]/cc[i].sum() for i in range(2) if cc[i].sum()]))
    fs=sum(r['false_safe'] for r in rows);fu=sum(r['false_unsafe'] for r in rows);ps=sum(r['predicted_safe'] for r in rows)
    r=dict(calls=len(rows),rows=n,margin_mae=sum(r['margin_abs_sum'] for r in rows)/n,margin_max_error=max(r['margin_max_error'] for r in rows),
        sign_agreement=sum(r['sign_matches'] for r in rows)/n,argmax_agreement=sum(r['argmax_matches'] for r in rows)/n,balanced_accuracy=balanced,
        correctness_confusion=cc.tolist(),transition_order=['wrong_wrong','wrong_correct','correct_wrong','correct_correct'],
        transition_confusion=tc.tolist(),transition_counts=tc.sum(1).tolist(),transition_recall=recall,
        floor_safe_agreement=sum(r['predicted_safe']==r['actual_safe'] for r in rows)/len(rows),
        predicted_safe_updates=ps,false_safe_updates=fs,false_unsafe_updates=fu,false_safe_per_update=fs/len(rows),
        false_safe_given_predicted_safe=fs/ps if ps else None)
    gates=dict(sign=r['sign_agreement']>=.99,argmax=r['argmax_agreement']>=.98,margin=r['margin_mae']<=.001,
        balanced=balanced>=.95,correct_wrong=recall[2] is None or recall[2]>=.9,wrong_correct=recall[1] is None or recall[1]>=.9,
        false_safe_per_update=r['false_safe_per_update']<=.01)
    r.update(accuracy_gate=gates,status='PASS' if all(gates.values()) else 'FAIL')
    return r
