"""New FREE-gap implicit coordinates, not a prescribed gap trajectory.

The local controls are target gap, orientation p and persistent corner loads.
Fifteen dependent coordinates include descent f, rootxy and12 legs. All original
f bounds and monotonicity remain explicit hard rows. Normal-interval gap drops
are a bounded simplex with fixed endpoint sum; their profile is FREE. This is
not440's fixed piecewise gap profile and not a continuation of462.
Preparation only: no complete trajectory/optimizer/filter/physics is implemented.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy import sparse
from low_seat_clock_transcription_444 import D,sha
DEP=np.array([0,1,2]+list(range(4,16)))
IND=np.array([3]+list(range(16,24)))
EQS=np.r_[np.arange(14),47]
ROOT_TOL=1e-13
GAP_ROUND=2.**-59 # Half ulp at every gap in[0,.012].
def frac(x):return F.from_float(float(x))
def pair(x):return [x.numerator,x.denominator]
def project_box_sum(ref,lower,upper,total):
 """Exact finite water-level KKT construction of one bounded-simplex projection."""
 assert sum(lower)<=total<=sum(upper)
 breaks=sorted(set([r-u for r,u in zip(ref,upper)]+[r-l for r,l in zip(ref,lower)]))
 def at(lam):return [min(u,max(l,r-lam)) for r,l,u in zip(ref,lower,upper)]
 for k,lam in enumerate(breaks):
  d=at(lam)
  if sum(d)==total:return d,lam,k
  if k+1==len(breaks):break
  mid=(lam+breaks[k+1])/2;free=[i for i,(r,l,u) in enumerate(zip(ref,lower,upper)) if l<r-mid<u]
  fixed=sum(at(mid)[i] for i in range(len(ref)) if i not in free)
  if free:
   root=(fixed+sum(ref[i] for i in free)-total)/len(free)
   if lam<=root<=breaks[k+1]:
    d=at(root)
    assert sum(d)==total
    return d,root,k
 raise AssertionError('No exact bounded-simplex root')
def implicit(J):
 A=J[:,EQS][:,:,DEP];B=J[:,EQS][:,:,IND]
 rhs=np.concatenate([np.broadcast_to(np.eye(15)[-1],(len(J),15))[:,:,None],-B],axis=2)
 T=np.zeros((len(J),24,10));T[:,DEP,:]=np.linalg.solve(A,rhs);T[:,IND,1:]=np.eye(9)
 raw=np.einsum('nij,njk->nik',J,T);G=np.concatenate([T,raw],axis=1)
 # These are analytic derivatives of the defining implicit identities,
 # not thresholding: feet/moments=0 and actual gap=the gap coordinate.
 G[:,24:38,:]=0.;G[:,71,:]=0.;G[:,71,0]=1.
 assert np.array_equal(G[:,38,:],np.tile(np.r_[0.,0.,np.ones(8)],(len(J),1)))
 return T,G,A,rhs,raw

def root(oracle,x,target_gap):
 x=x.copy();journal=[]
 for k in range(11):
  y=oracle.values(x);r=np.c_[y[:,:14],y[:,47]-target_gap];err=float(abs(r).max());journal.append(err)
  if err<=ROOT_TOL:return x,y,journal
  if k==10:raise RuntimeError('Declared15-equation root limit')
  J=oracle.jacobian(x);dx=np.linalg.solve(J[:,EQS][:,:,DEP],-r[:,:,None])[:,:,0]
  accepted=False
  for j in range(9):
   q=x.copy();q[:,DEP]+=2.**-j*dx;yy=oracle.values(q);rr=np.c_[yy[:,:14],yy[:,47]-target_gap]
   if np.max(abs(rr))<err:x=q;accepted=True;break
  if not accepted:raise RuntimeError('Declared root backtracking failed')
 raise AssertionError()
