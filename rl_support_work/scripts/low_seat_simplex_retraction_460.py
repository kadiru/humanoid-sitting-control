"""New bounded, total-normal-conserving chart for the entire clock domain.

This is arithmetic preparation, not a path solver or correction of any old
trial. The immutable440 input and unchanged444/446v2 rows are retained.
At each free state one persistent load is dependent: its tangent is the exact
negative sum of the other load tangents. A SINGLE rational scale is shared by
all loads at that state, so the unrounded update preserves their sum exactly.
For d in this null space, rho=max_i(abs(d_i)/room_i), alpha=1/(1+rho), and
w'=w+alpha*d. Exact Fraction arithmetic constructs the whole update; each
output is rounded to binary64 once. Bounds hold after rounding. Sum error is
bounded explicitly, not removed by normalization. Phase coordinates use the
same scalar formula, independently; their monotonicity remains a hard row.
At zero the feasible derivative is the sparse null-space basis. Nonzero ties
in rho may be nonsmooth; a prospective solver must re-center at each trial
and use the zero-chart derivative, not pretend it is the nonzero Jacobian.
"""
from fractions import Fraction as F
import numpy as np
from scipy import sparse
from low_seat_collocation_solver_448 import load_saved, D, sha

CONTRACT=dict(seed='immutable440; no450/456 final or rejected trial',
 coordinate_domain_radius=.02, max_consecutive_map_roundings=120,
 binary64_unit_roundoff=2.**-53, max_round_error_per_load=2.**-54,
 original_total_normal_bound=1e-13, derivative_bound=2e-6,
 exact_rational_load_sum_before_rounding=True,
 normalization_or_clipping=False, candidate_exists=False,
 structural_phase_monotonicity=False, phase_monotonicity_is_hard_constraint=True,
 physical_admission=False)

def layout(d):
 free=np.asarray(d['free']); pos={int(v):i for i,v in enumerate(free)}
 groups=[];rr=[];cc=[];vv=[];col=0
 for state in range(len(d['initialX'])):
  for kind,slots in [('phase_f',[0]),('phase_p',[1]),('loads',list(range(2,10)))]:
   ids=[pos[state*10+s] for s in slots if state*10+s in pos]
   if not ids:continue
   if kind=='loads':
    assert len(ids)>=2
    margins=np.minimum(d['reference'][ids]-d['lower'][ids],d['upper'][ids]-d['reference'][ids])
    pivot=int(np.argmax(margins)) # Fixed arithmetic convention, not candidate search.
    independent=[i for i in range(len(ids)) if i!=pivot]
   else:pivot=-1;independent=[0]
   columns=list(range(col,col+len(independent)))
   for j,k in zip(independent,columns):
    rr.append(ids[j]);cc.append(k);vv.append(1.)
    if pivot>=0:rr.append(ids[pivot]);cc.append(k);vv.append(-1.)
   groups.append(dict(state=state,kind=kind,indices=ids,pivot=pivot,independent=independent,columns=columns))
   col+=len(independent)
 R=sparse.csr_matrix((vv,(rr,cc)),shape=(len(free),col))
 return groups,R

def exact_direction(t,pivot,size):
 if pivot<0:return [F.from_float(float(t[0]))]
 ds=[];j=0
 for i in range(size):
  if i==pivot:ds.append(F(0))
  else:ds.append(F.from_float(float(t[j])));j+=1
 ds[pivot]=-sum(ds,F(0))
 return ds

def decode_group(c,t,lo,hi,pivot):
 if not (np.isfinite(c).all() and np.isfinite(t).all() and np.all(lo<=c) and np.all(c<=hi)):
  raise ValueError('invalid bounded-conservation chart origin')
 cq=list(map(lambda x:F.from_float(float(x)),c));lq=list(map(lambda x:F.from_float(float(x)),lo));uq=list(map(lambda x:F.from_float(float(x)),hi))
 ds=exact_direction(t,pivot,len(c));ratios=[]
 for i,di in enumerate(ds):
  if di==0:ratios.append(F(0));continue
  room=(uq[i]-cq[i]) if di>0 else (cq[i]-lq[i])
  if room<=0:raise ValueError('outward tangent at an exact bound')
  ratios.append(abs(di)/room)
 rho=max(ratios);alpha=1/(1+rho)
 q=[ci+alpha*di for ci,di in zip(cq,ds)]
 assert all(l<=v<=u for l,v,u in zip(lq,q,uq))
 if pivot>=0:assert sum(q,F(0))==sum(cq,F(0))
 out=np.asarray([float(v) for v in q]);assert np.all(lo<=out) and np.all(out<=hi)
 return out,alpha,ds,q

def decode(origin,tangent,lower,upper,groups):
 out=origin.copy();alphas=[];sum_errors=[]
 for g in groups:
  ids=g['indices'];t=tangent[g['columns']]
  v,a,ds,q=decode_group(origin[ids],t,lower[ids],upper[ids],g['pivot'])
  out[ids]=v;alphas.append(float(a))
  if g['pivot']>=0:
   delta=sum((F.from_float(float(x)) for x in v),F(0))-sum((F.from_float(float(x)) for x in origin[ids]),F(0))
   sum_errors.append(float(delta))
 return out,np.asarray(alphas),np.asarray(sum_errors)
