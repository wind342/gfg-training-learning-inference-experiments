"""Cheap event scheduling and optional explicitly open-loop release."""
from collections import deque
import math
import statistics


class CheapSentinel:
    def __init__(self,max_interval=20,window=20,warmup=5,k=3.,relative_floor=.001,absolute_floor=1e-12):
        if not 1 <= warmup <= window or max_interval < 1:
            raise ValueError('invalid scheduler intervals')
        if any(not math.isfinite(v) or v <= 0 for v in (k,relative_floor,absolute_floor)):
            raise ValueError('sentinel thresholds must be positive finite')
        self.max_interval=max_interval;self.warmup=warmup;self.k=k
        self.relative_floor=relative_floor;self.absolute_floor=absolute_floor
        self.losses=deque(maxlen=window);self.norms=deque(maxlen=window)
        self.last_instrument=None;self.last_step=0

    def _anomaly(self,value,history):
        if len(history)<self.warmup:return False
        median=statistics.median(history)
        mad=statistics.median(abs(x-median) for x in history)
        scale=max(mad,abs(median)*self.relative_floor,self.absolute_floor)
        return value>median+self.k*scale

    def observe(self,step,loss,norm):
        if step != self.last_step+1:raise ValueError('steps must start at 1 and be consecutive')
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError('nonfinite training signal; optimizer must not execute')
        reasons=[]
        if self.last_instrument is None:reasons.append('FIRST_STEP')
        elif step-self.last_instrument>=self.max_interval:reasons.append('MAX_INTERVAL')
        if self._anomaly(loss,self.losses):reasons.append('LOSS_ANOMALY')
        if self._anomaly(norm,self.norms):reasons.append('GRAD_NORM_ANOMALY')
        # Compare against PAST observations, then append current signal.
        self.losses.append(loss);self.norms.append(norm);self.last_step=step
        return reasons or ['NONE']

    def measured(self,step):self.last_instrument=step


class ReleaseToNative:
    def __init__(self,release_steps=0):
        if not isinstance(release_steps,int) or release_steps<0:raise ValueError('invalid release_steps')
        self.release_steps=release_steps;self.step=None;self.alpha=1.

    def decided(self,step,alpha):
        if not math.isfinite(alpha) or alpha<=0:raise ValueError('invalid alpha')
        self.step=step;self.alpha=float(alpha)

    def at(self,step):
        if self.step is None:return 1.
        j=step-self.step
        if j<0:raise ValueError('release before decision')
        if j==0:return self.alpha
        if self.release_steps==0:return 1.
        return 1+(self.alpha-1)*max(0.,1-j/self.release_steps)
