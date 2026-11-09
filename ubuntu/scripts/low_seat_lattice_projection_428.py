"""Integer-exact research construction derived from the exported424 root only.

First solve a finite arithmetic resource allocation: minimize the maximum ratio
of saved necessary phase duration to allocated integer phase duration. This is
not a physical/allocation trial sweep. No other424 node or unvisited queue is used.
Then a NEW all-hard nonlinear closest-path problem changes ALL894 continuous
load/geometry/phase/duration coordinates jointly at the derived EXACT budgets.
There is NO feasibility slack, epigraph or continuation of424's branching tree.
Every retained original/support/window row and old coordinate bound is HARD.
A uniform float32-epsilon DIMENSIONLESS inward numerical buffer is added to the
already tightened rows, solely to avoid constructing on signed roundoff. It is
not an acceptance tolerance, physical allowance, or float32 execution guarantee.
The unchanged406-centered416 research box is retained; callbacks never selected.
"""
import numpy as np
from fractions import Fraction
GROUPS=((0,16),(16,64),(64,99),(99,104),(104,114))
BLO=(88,189,241,101,11);BHI=(98,199,251,106,21)
BUFFER=float(np.finfo(np.float32).eps)
def necessary(C,gaps,rates,lo,hi,ulp):
    dc=np.diff(C,axis=0);speed=rates-4*ulp/.02;up=hi-C[1:]-32*ulp;down=C[1:]-lo-32*ulp
    assert min(up.min(),down.min(),speed.min())>0
    normal=np.zeros(114);normal[99:104]=(gaps[98:103]-gaps[99:104])/.006
    parts=np.c_[np.full(114,.02),np.max(abs(dc)/speed,1),np.max(.18*np.maximum(dc,0)/up,1),np.max(.18*np.maximum(-dc,0)/down,1),normal]
    req=parts.max(1)
    return parts,req,np.array([sum(req[a:b]) for a,b in GROUPS])
def allocate(requirements):
    # Exact rational comparisons of the saved binary64 requirements. Dynamic
    # programming stores the optimal minimax value and lexicographic tie winner
    # for every reachable prefix tick sum; it evaluates NO robot candidate.
    layer={0:(Fraction(0),( ))};history=[]
    for g,r in enumerate(requirements):
        new={};rf=Fraction(float(r));dt=Fraction(1,50)
        for total,(value,path) in layer.items():
            for budget in range(BLO[g],BHI[g]+1):
                nt=total+budget;candidate=(max(value,rf/(dt*budget)),path+(budget,))
                if nt not in new or candidate<new[nt]:new[nt]=candidate
        history.append({str(k):dict(value_numerator=v[0].numerator,value_denominator=v[0].denominator,budgets=list(v[1])) for k,v in new.items()});layer=new
    value,budgets=layer[650]
    # Certificate: strictly smaller objective permits at most budget-1 at one
    # limiting group, or requires too many ticks when all phase ratios are below.
    lower_at=[];lower_strict=[]
    for g,r in enumerate(requirements):
        ok=[b for b in range(BLO[g],BHI[g]+1) if Fraction(float(r))/(Fraction(1,50)*b)<=value]
        strict=[b for b in range(BLO[g],BHI[g]+1) if Fraction(float(r))/(Fraction(1,50)*b)<value]
        lower_at.append(min(ok) if ok else None);lower_strict.append(min(strict) if strict else None)
    assert None not in lower_at and sum(lower_at)<=650
    assert None in lower_strict or sum(lower_strict)>650
    return np.array(budgets,dtype=np.int64),dict(objective=float(value),objective_numerator=value.numerator,objective_denominator=value.denominator,minimum_ticks_at_optimum=lower_at,minimum_ticks_strictly_below=lower_strict,strictly_better_allocation_impossible_for_saved_ratio_objective=True,dynamic_program_layers=history)
def rows(iq,v,scale,keep,lo,hi,I=None,G=None):
    out=np.r_[iq[keep]/scale[keep]-BUFFER,v[:786]-lo-BUFFER,hi-v[:786]-BUFFER]
    if I is None:return out
    return out,np.vstack([I[keep,:894]/scale[keep,None],G[:786,:894],-G[:786,:894]])
def objective(x,reference,radius):
    d=(x-reference)/radius
    return .5*float(d@d),d/radius
