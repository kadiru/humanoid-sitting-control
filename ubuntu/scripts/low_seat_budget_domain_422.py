"""One mixed-integer whole-path domain with four independent event budgets.

Seed is original406 through unchanged410, never the result of418. All894
continuous shape/load/phase/interior-time coordinates keep the EXACT416 box.
The first four event budgets vary in the existing +/-5-tick research box;
fifth=650-sum(first four). Normal descent >=101 ticks, load >=11 ticks.
All physical thresholds and existing inward construction reserves are unchanged.
A bounded integer branching algorithm will solve direct nonlinear relaxations;
no failed relaxation is an infeasibility certificate or a command schedule.
"""
import numpy as np
from low_seat_direct_constraints_416 import partition,coordinate_bounds,box
BUDGET_LO=np.array([88.,189.,241.,101.,11.])
BUDGET_HI=np.array([98.,199.,251.,106.,21.])
BUDGET_MAP=np.vstack([np.eye(4),-np.ones(4)])
def expand(y):
    return np.r_[y[:898],650.-np.sum(y[894:898])]
def domain(seed):
    lo,hi=box(seed)
    return np.r_[lo,BUDGET_LO[:4]],np.r_[hi,BUDGET_HI[:4]]
def assemble(iq,v,z,slack,scale,keep,lo,hi,I=None,G=None):
    hard=partition();out=np.r_[iq[keep]/scale[keep]+(~hard[keep])*slack,v[:786]-lo,hi-v[:786],z[-1]-BUDGET_LO[-1],BUDGET_HI[-1]-z[-1]]
    if I is None:return out
    m=int(keep.sum());J=np.zeros((len(out),899));J[:m,:894]=I[keep,:894]/scale[keep,None];J[:m,894:898]=(I[keep,894:]@BUDGET_MAP)/scale[keep,None];J[:m,-1]=~hard[keep]
    Q=np.c_[G[:786,:894],G[:786,894:]@BUDGET_MAP]
    J[m:m+786,:898]=Q;J[m+786:m+1572,:898]=-Q
    J[-2,894:898]=-1.;J[-1,894:898]=1.
    return out,J

def full_clock(C,dt):
    """Diagnostic ideal shared clock, even for a fractional event relaxation.
    Fractional mandatory events are retained and NEVER called aligned/executable.
    """
    times=np.r_[0.,np.cumsum(np.r_[dt[:99],.5,dt[99:]])]
    vertices=C[np.r_[np.arange(100),99,np.arange(100,115)]]
    ticks=np.arange(676)*.02
    states=np.column_stack([np.interp(ticks,times,vertices[:,j]) for j in range(16)])
    return states,times
