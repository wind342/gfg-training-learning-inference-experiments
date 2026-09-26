"""Fix V1 anchor overfill; keep real capability history and existing gate."""
from experiments.gfg_functional_feedback_router_v1.policy import Router as BaseRouter,identity

class Router(BaseRouter):
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
        def rotate(pool,n,key):
            if not pool or n<=0:return []
            start=self.turns[key]%len(pool);take=min(n,len(pool))
            result=[pool[(start+j)%len(pool)] for j in range(take)]
            self.turns[key]=(start+take)%len(pool);return result
        selected=[];selected_anchors=[];before={};examples=[]
        if mode!='FORMATION':
            room=7 if anchors else 8
            selected=rotate(degraded,room,'degraded')
            selected+=rotate(risk,room-len(selected),'risk')
            # The only policy change: ONE rotating stable anchor, no filler.
            selected_anchors=rotate(anchors,1,'anchor')
            selected+=selected_anchors
            if selected:
                memory.check()
                if not set(selected)<=set(memory.groups):raise ValueError('missing old training memory')
                for g in selected:
                    before[str(g)]=memory.cursors[g];examples.extend(memory.take(g))
        if len(selected_anchors)>1 or len(examples)>16:raise AssertionError('anchor/cap bound violated')
        if mode!='FORMATION' and len(degraded)+len(risk)==1 and len(examples)>4:raise AssertionError('single risk caused blanket replay')
        self.last_step=step;self.mode=mode
        return dict(router_mode=mode,active_degraded_groups=degraded,active_at_risk_groups=risk,
            selected_feedback_groups=selected,selected_anchor_groups=selected_anchors,
            feedback_example_ids=examples,feedback_batch_size=len(examples),
            per_group_cursor_before=before,per_group_cursor_after={str(g):memory.cursors[g] for g in selected},
            router_activation=activated,release_event=release,pre_step_state_sha256=state_id,
            rotation_cursors=dict(self.turns))
