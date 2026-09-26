"""Independent algebra replay from immutable head states, not another training."""
import gzip
from .core import *

@torch.no_grad()
def reconstruct(before,native,h,y,n,kind):
    def calc(s,features=h):return torch.nn.functional.linear(features,s['weight'],s['bias'])[:,:16]
    def mg(z):
        mask=torch.nn.functional.one_hot(y,16).bool()
        return z.gather(1,y[:,None]).flatten()-z.masked_fill(mask,-torch.inf).max(1).values
    b=mg(calc(before));z=calc(native);nm=mg(z);eligible=b[:n]>0
    bad=lambda m:torch.where(eligible&(m[:n]<=0))[0].cpu().tolist()
    initial=bad(nm);logs=[];candidate=native;cm=nm
    if initial:
        dw=native['weight']-before['weight'];db=native['bias']-before['bias']
        for j in range(2):
            pairs=[]
            for i in range(n):
                if bool(eligible[i]):
                    t=int(y[i]);pairs += [(i,t,c) for c in range(16) if c!=t and bool(z[i,c]>=z[i,t])]
            cert=[]
            for i,t,c in pairs:
                norm2=float(2*(torch.sum(h[i].to(torch.float64)**2)+1))
                if kind=='current':
                    value=float(torch.dot(h[i].double(),dw[t].double()-dw[c].double())+db[t].double()-db[c].double())
                    coeff=value/norm2 if value<0 else 0.
                    a=-coeff
                else:
                    zz=calc({'weight':before['weight']+dw,'bias':before['bias']+db},h[i:i+1])
                    value=float(zz[0,t]-zz[0,c]);coeff=(1e-6-value)/norm2 if value<=0 else 0.;a=coeff
                if a:
                    dw[t].add_(h[i],alpha=a);dw[c].add_(h[i],alpha=-a);db[t].add_(a);db[c].add_(-a)
                record=dict(control=i,target=t,competitor=c,norm2=norm2,coefficient=coeff)
                if kind=='current':
                    record.update(dot_before=value,dot_after=float(torch.dot(h[i].double(),dw[t].double()-dw[c].double())+db[t].double()-db[c].double()))
                else:record['margin_before']=value
                cert.append(record)
            candidate={'weight':before['weight']+dw,'bias':before['bias']+db};z=calc(candidate);cm=mg(z)
            logs.append(dict(pass_number=j+1,certificates=cert,margins=cm.cpu().tolist(),violating=bad(cm),safe=not bad(cm)))
            if not bad(cm):break
    skip=bool(bad(cm))
    return before if skip else candidate,candidate,dict(baseline=b.cpu().tolist(),native=nm.cpu().tolist(),cut=cm.cpu().tolist(),
        eligible=torch.where(eligible)[0].cpu().tolist(),native_violating=initial,passes=logs,skip=skip)

def independent_geometry(b,n,c):
    u=torch.concat([(n[k]-b[k]).to(torch.float64).reshape(-1) for k in b]);v=torch.concat([(c[k]-b[k]).to(torch.float64).reshape(-1) for k in b])
    un=float(torch.linalg.vector_norm(u));vn=float(torch.linalg.vector_norm(v))
    return dict(native_norm=un,cut_norm=vn,norm_ratio=vn/un if un else None,
        cosine=float(u@v)/(un*vn) if un and vn else None,removed_norm_fraction=float(torch.linalg.vector_norm(u-v))/un if un else None)

def main():
    pre=guard();setup();d=np.load(DATA/'data.npz');labels=torch.from_numpy(d['y']);ev=json.loads((ROOT/'EVALUATION.json').read_text())
    barrier=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text());assert ev['barrier_sha256']==sha(ROOT/'TRAINING_COMPLETE.json')
    assert sha(ROOT/'CACHE.pt')==barrier['cache_sha256']
    cache=torch.load(ROOT/'CACHE.pt',map_location='cpu',weights_only=False);byid=dict(zip(cache['indices'],cache['features']))
    parent=torch.load(PARENT_ROOT/'head_only_exact_cut'/'AUDIT.pt',map_location='cpu',weights_only=False)
    parent_rows=[json.loads(v) for v in (PARENT_ROOT/'head_only_exact_cut'/'STEPS.jsonl').read_text().splitlines()]
    parent_eval=json.loads((PARENT_ROOT/'EVALUATION.json').read_text())['result']['test']['primary']
    parent_result=json.loads((PARENT_ROOT/'head_only_exact_cut'/'TRAIN_RESULT.json').read_text())
    # Normalize only in memory; historical artifacts are never rewritten.
    for row in parent_rows:
        row['cut_geometry']=row['report']['cut_geometry'];row['executed_geometry']=row['report']['executed_geometry']
    table={'existing_10_current':dict(test={k:parent_eval['macro_'+k] for k in ('old','new','all')},
        **summary(parent_rows,80),training_wall=parent_result['training_wall'])}
    checks={};endpoint_values=0;certs=0
    for arm,(n,kind) in ARMS.items():
        out=ROOT/arm;r=json.loads((out/'TRAIN_RESULT.json').read_text());assert r['status']=='COMPLETED'
        assert sha(out/'TRAIN_RESULT.json')==barrier['train_result_hashes'][arm]
        assert ev['started_unix']>=r['finished_unix']
        for file,key in [('AUDIT.pt','audit_sha256'),('STEPS.jsonl','log_sha256'),('CONFIG.json','config_sha256')]:assert sha(out/file)==r[key]
        a=torch.load(out/'AUDIT.pt',map_location='cpu',weights_only=False)
        assert equal(a['initial_optimizer'],parent['initial_optimizer']) and equal(a['heads'][0],parent['heads'][0])
        assert a['control_ids']==bank_ids(d,n).tolist()
        assert torch.equal(a['control_features'],torch.stack([byid[i] for i in a['control_ids']]))
        h=a['control_features'].cuda();y=labels[a['control_ids']].cuda();rows=[json.loads(v) for v in (out/'STEPS.jsonl').read_text().splitlines()]
        assert len(rows)==210 and len(a['heads'])==211 and len(a['natives'])==len(a['cuts'])==210
        assert all(float(s['step'])==420 for s in a['final_optimizer']['state'].values())
        for j,row in enumerate(rows):
            assert row['step']==j+1 and row['current_ids']==order(d,2)[j].tolist() and row['seed']==2021*100000+211+j
            assert row['lr']==LR and row['old_replay']==0 and row['moments_unmodified_by_cut'] and row['bert_frozen_eval']
            b={k:v.cuda() for k,v in a['heads'][j].items()};native={k:v.cuda() for k,v in a['natives'][j].items()}
            outstate,candidate,report=reconstruct(b,native,h,y,n*8,kind)
            for key,value in report.items():assert value==row['report'][key],(arm,j,key)
            assert equal(cpu_state(outstate),a['heads'][j+1]) and equal(cpu_state(candidate),a['cuts'][j])
            actual=margins(affine(outstate,h),y).cpu().tolist();assert actual==row['report']['executed']
            assert not any(x>0 and z<=0 for x,z in zip(report['baseline'][:n*8],actual[:n*8]))
            for state,key in [(b,'head_before_sha256'),(outstate,'head_after_sha256'),(native,'native_sha256'),(candidate,'cut_sha256')]:assert tensor_hash(state)==row[key]
            for name,state in [('cut_geometry',candidate),('executed_geometry',outstate)]:
                expected=independent_geometry(b,native,state)
                for k,v in expected.items():
                    got=row[name][k]
                    assert (v is None and got is None) or (v is not None and got is not None and abs(v-got)<=1e-12),(arm,j,name,k)
            endpoint_values+=4*len(y);certs+=sum(len(p['certificates']) for p in report['passes'])
        assert tensor_hash(a['heads'][-1])==r['head_sha256'] and r['summary']==summary(rows,n*8)
        assert r['backbone_sha256']==pre['backbone_initial_sha256']
        assert ev['indices']==scope(d,16)[1].tolist()
        metrics=summarize_predictions(ev['indices'],ev['arms'][arm]['predictions'],d,2)
        assert metrics==ev['arms'][arm]['test']
        table[arm]=dict(test={k:metrics['macro_'+k] for k in ('old','new','all')},**r['summary'],training_wall=r['training_wall'])
        checks[arm]=dict(status='PASS',steps=210,independent_cut_reconstruction=True,actual_no_old_correct_to_wrong=True)
        print('CHECKED',arm,flush=True);del a,rows,h,y;torch.cuda.empty_cache()
    guard()
    write(ROOT/'CHECKS.json',dict(status='PASS',arms=checks,endpoint_values=endpoint_values,pair_certificates=certs,source_hashes=sources(),
        final_old_files_unchanged=True,test_after_all_training=True))
    write(ROOT/'RESULTS.json',dict(status='COMPLETED',checks='PASS',arms=table))
    print(json.dumps(table,indent=2),flush=True)

if __name__=='__main__':main()
