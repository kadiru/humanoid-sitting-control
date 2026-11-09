"""Pure algebra for a new direct nonlinear, command-hard complete-path problem.

Use the unchanged410 positive-load/monotone coordinate domain at its ORIGINAL
406 seed, never412's result. Fix the five already-declared event budgets rather
than proposing new allocations. All894 continuous shape/load/phase/interior-time
coordinates vary jointly. Make static command limits HARD alongside every
previous hard row and all original coordinate bounds. Only rate/headroom/window
rows receive one nonnegative epigraph. A single direct nonlinear SLSQP problem
replaces the MILP predictor/fraction-to-boundary/trust-floor algorithm entirely.
No previous optimizer is resumed, extended, polished or repeated.

The fixed one-natural-log-unit domain for structural variables and .1 for time
logratios reuses the maximum local coordinate box previously declared412, now
as a single explicit research domain centered at406. No physical threshold or
inward reserve changes. Internal nonlinear iterates need not be feasible; only
the final result is screened. No callback selection or feasibility fallback.
"""
import numpy as np

def partition():
    hard=np.ones(27708,dtype=bool)
    hard[12212:19508]=False
    hard[19759]=False
    hard[20284:]=False
    return hard

def coordinate_bounds(base,shape):
    lo=np.c_[base['lower'][:,2:4],base['lower'][:,21:23],base['lower'][:,16:21]].ravel()[shape]
    hi=np.c_[base['upper'][:,2:4],base['upper'][:,21:23],base['upper'][:,16:21]].ravel()[shape]
    assert len(lo)==len(hi)==786 and np.isfinite(lo).all() and np.isfinite(hi).all()
    return lo,hi

def assemble(iq,v,slack,scale,keep,lo,hi,I=None,G=None):
    hard=partition()
    out=np.r_[iq[keep]/scale[keep]+(~hard[keep])*slack,v[:786]-lo,hi-v[:786]]
    if I is None:return out
    jac=np.zeros((len(out),895));m=int(keep.sum())
    jac[:m,:894]=I[keep,:894]/scale[keep,None]
    jac[:m,-1]=(~hard[keep]).astype(float)
    jac[m:m+786,:894]=G[:786,:894]
    jac[m+786:,:894]=-G[:786,:894]
    return out,jac

def box(seed):
    radius=np.r_[np.ones(785),np.full(109,.1)]
    return seed[:894]-radius,seed[:894]+radius
