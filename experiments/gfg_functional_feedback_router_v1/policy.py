"""Bounded deterministic feedback access; no model or parameter-space optimizer."""
import hashlib
import json
from fractions import Fraction
import numpy as np

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Memory:
    def __init__(self):self.groups={};self.cursors={};self.stamp=identity(self.groups)
    def check(self):
        if identity(self.groups)!=self.stamp:raise ValueError('memory identity changed')
    def append(self,group,ids,*,completed,split,labels,forbidden):
        self.check();ids=tuple(int(i) for i in ids)
        if group not in completed or len(ids)!=10 or len(set(ids))!=10:raise ValueError('invalid completed memory group')
        if any(split[i]!=0 or labels[i]!=group or i in forbidden for i in ids):raise ValueError('nontraining/forbidden memory')
        if group in self.groups:
            if self.groups[group]!=ids:raise ValueError('memory replacement')
            return
        self.groups[group]=ids;self.cursors[group]=0;self.stamp=identity(self.groups)
    def take(self,group):
        self.check();c=self.cursors[group];ids=self.groups[group]
        self.cursors[group]=(c+2)%len(ids)
        return list(ids[c:c+2]) if c+2<=len(ids) else [ids[c],ids[0]]

class Router:
    def __init__(self):self.mode='FORMATION';self.turns={'degraded':0,'risk':0,'anchor':0};self.last_step=None
    def choose(self,step,labels,states,levels,old_groups,memory):
        if self.last_step is not None and step!=self.last_step+1:raise ValueError('nonconsecutive router step')
        if not(len(labels)==len(states)==len(levels)) or len(set(labels))!=len(labels):raise ValueError('group state mismatch')
        if not set(old_groups)<=set(labels):raise ValueError('unknown old group')
        degraded=sorted(g for g,s in zip(labels,states) if g in old_groups and s=='DEGRADED')
        risk=sorted(g for g,s,l in zip(labels,states,levels) if g in old_groups and s=='PROTECTED' and l==1)
        anchors=sorted(g for g,s,l in zip(labels,states,levels) if g in old_groups and s=='PROTECTED' and l==0)
        mode='RECOVERY' if degraded else 'BALANCE' if risk else 'FORMATION'
        release=self.mode!='FORMATION' and mode=='FORMATION'
        activated=self.mode=='FORMATION' and mode!='FORMATION'
        state_id=identity(dict(labels=list(labels),states=list(states),levels=[int(v) for v in levels],old=sorted(old_groups)))
        selected=[];selected_anchors=[];before={};examples=[]
        def rotate(pool,n,key):
            if not pool or n<=0:return []
            start=self.turns[key]%len(pool);take=min(n,len(pool))
            result=[pool[(start+j)%len(pool)] for j in range(take)]
            self.turns[key]=(start+take)%len(pool);return result
        if mode!='FORMATION':
            # Reserve an anchor slot when available, without exceeding 8 groups.
            room=7 if anchors else 8
            selected=rotate(degraded,room,'degraded')
            selected+=rotate(risk,room-len(selected),'risk')
            selected_anchors=rotate(anchors,8-len(selected),'anchor')
            selected+=selected_anchors
            if selected:
                memory.check()
                if not set(selected)<=set(memory.groups):raise ValueError('missing old training memory')
                for g in selected:
                    before[str(g)]=memory.cursors[g];examples.extend(memory.take(g))
        self.last_step=step;self.mode=mode
        return dict(router_mode=mode,active_degraded_groups=degraded,active_at_risk_groups=risk,
            selected_feedback_groups=selected,selected_anchor_groups=selected_anchors,
            feedback_example_ids=examples,feedback_batch_size=len(examples),
            per_group_cursor_before=before,per_group_cursor_after={str(g):memory.cursors[g] for g in selected},
            router_activation=activated,release_event=release,pre_step_state_sha256=state_id,
            rotation_cursors=dict(self.turns))

def emergency_gate(measurement,mapping,threshold=Fraction(9,10)):
    allowed=(.125,.25,.5,1.)
    if tuple(measurement.alphas)!=allowed:raise ValueError('gate requires exactly four shrink/native candidates')
    baseline=mapping.values(measurement.baseline_minima.detach().cpu().numpy())
    protected=[k for k,c in enumerate(baseline['correct_counts']) if Fraction(c,mapping.counts[k])>=threshold]
    scores={};keys={}
    for j,a in enumerate(allowed):
        v=mapping.values(measurement.predicted_minima[j].detach().cpu().numpy())
        deficits=[max(Fraction(0),threshold-Fraction(v['correct_counts'][k],mapping.counts[k])) for k in protected]
        key=(sum(d>0 for d in deficits),max(deficits,default=Fraction(0)),sum(deficits,Fraction(0)))
        keys[a]=key;scores[str(a)]=dict(**v,violation_count=key[0],worst_deficit=float(key[1]),total_deficit=float(key[2]))
    if keys[1.][0]==0:chosen=1.;reason='NATIVE_SAFE'
    else:
        chosen=min(allowed,key=lambda a:(keys[a],abs(a-1.)))
        reason='NO_SAFE_ALPHA' if keys[chosen][0] else 'EMERGENCY_SHRINK'
    return dict(alpha=chosen,reason=reason,protected_groups=[mapping.labels[k] for k in protected],scores=scores)
