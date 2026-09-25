"""Scientific plane: frozen endpoint arithmetic and selector."""
import math
import torch

ALPHAS=(.25,.5,.75,1.)
REFERENCE_CADENCE=5


def endpoints(g0,first,joint,curvature,alpha):
    return ((g0+alpha*first)+alpha*alpha*joint)+alpha*alpha*curvature


def select(minima,valid):
    n=minima.shape[1]
    if n<1:raise ValueError('empty control bank')
    ordered=minima.sort(1).values
    median=ordered[:,n//2] if n%2 else (ordered[:,n//2-1]+ordered[:,n//2])/2
    payload=torch.cat(((minima>0).sum(1).double(),median,valid.reshape(1).double())).cpu().tolist()
    if payload[-1]!=1 or not all(math.isfinite(x) for x in payload):raise FloatingPointError('nonfinite boundary')
    scores={str(a):{'predicted_correct_count':int(payload[j]),'median_predicted_minimum_margin':payload[j+4]} for j,a in enumerate(ALPHAS)}
    alpha=max(ALPHAS,key=lambda a:(scores[str(a)]['predicted_correct_count'],scores[str(a)]['median_predicted_minimum_margin'],-abs(a-1.)))
    return alpha,scores
