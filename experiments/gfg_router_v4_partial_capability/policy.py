"""Completed partial capabilities: lower-priority bounded maintenance."""
from experiments.gfg_functional_feedback_router_v1.policy import Router as BaseRouter,identity
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel

def complete_old(sentinel,completed):
    converted=[]
    for k,g in enumerate(sentinel.mapping.labels):
        if g in completed and sentinel.state[k]=='FORMING':
            sentinel.state[k]='PARTIAL';converted.append(g)
    return converted

class PartialSentinel(GroupSentinel):
    def _observe_group(self,k,margins,source):
        if self.state[k]!='PARTIAL':return super()._observe_group(k,margins,source)
        # Reuse exactly the FORMING threshold/observation calculation. Being
        # subthreshold alone is not a new degradation trigger.
        self.state[k]='FORMING'
        try:return super()._observe_group(k,margins,source)
        finally:
            if self.state[k]=='FORMING':self.state[k]='PARTIAL'

class Router(BaseRouter):
    def __init__(self):
        super().__init__();self.turns['partial']=0

    def choose(self,step,labels,states,levels,old_groups,memory):
        if self.last_step is not None and step!=self.last_step+1:raise ValueError('nonconsecutive router step')
        if not(len(labels)==len(states)==len(levels)) or len(set(labels))!=len(labels):raise ValueError('group state mismatch')
        if not set(old_groups)<=set(labels):raise ValueError('unknown old group')
        degraded=sorted(g for g,s in zip(labels,states) if g in old_groups and s=='DEGRADED')
        risk=sorted(g for g,s,l in zip(labels,states,levels) if g in old_groups and s=='PROTECTED' and l==1)
        partial=sorted(g for g,s in zip(labels,states) if g in old_groups and s=='PARTIAL')
        anchors=sorted(g for g,s,l in zip(labels,states,levels) if g in old_groups and s=='PROTECTED' and l==0)
        mode='RECOVERY' if degraded else 'BALANCE' if risk or partial else 'FORMATION'
        release=self.mode!='FORMATION' and mode=='FORMATION'
        activated=self.mode=='FORMATION' and mode!='FORMATION'
        state_id=identity(dict(labels=list(labels),states=list(states),levels=[int(v) for v in levels],old=sorted(old_groups)))
        def rotate(pool,n,key):
            if not pool or n<=0:return []
            start=self.turns[key]%len(pool);take=min(n,len(pool))
            result=[pool[(start+j)%len(pool)] for j in range(take)]
            self.turns[key]=(start+take)%len(pool);return result
        selected=[];selected_anchors=[];selected_partial=[];before={};examples=[]
        if mode!='FORMATION':
            # Keep V3's high-priority reservation rule exactly: one auxiliary
            # slot when stable anchors exist; otherwise up to eight risk groups.
            room=7 if anchors else 8
            selected=rotate(degraded,room,'degraded')
            selected+=rotate(risk,room-len(selected),'risk')
            if partial and step%2==0 and len(selected)<8:
                selected_partial=rotate(partial,1,'partial')
            elif anchors:
                selected_anchors=rotate(anchors,1,'anchor')
            selected+=selected_partial+selected_anchors
            if selected:
                memory.check()
                if not set(selected)<=set(memory.groups):raise ValueError('missing old training memory')
                for g in selected:
                    before[str(g)]=memory.cursors[g];examples.extend(memory.take(g))
        if len(selected_partial)+len(selected_anchors)>1 or len(examples)>16:raise AssertionError('auxiliary/cap violation')
        if selected_partial and step%2:raise AssertionError('partial cadence violation')
        self.last_step=step;self.mode=mode
        return dict(router_mode=mode,active_degraded_groups=degraded,active_at_risk_groups=risk,
            active_partial_groups=partial,selected_partial_groups=selected_partial,
            partial_due=bool(partial and step%2==0),
            selected_feedback_groups=selected,selected_anchor_groups=selected_anchors,
            feedback_example_ids=examples,feedback_batch_size=len(examples),
            per_group_cursor_before=before,per_group_cursor_after={str(g):memory.cursors[g] for g in selected},
            router_activation=activated,release_event=release,pre_step_state_sha256=state_id,
            rotation_cursors=dict(self.turns))
