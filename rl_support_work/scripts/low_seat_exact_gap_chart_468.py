"""NEW finite, exactly conserved normal-drop chart; no trajectory solver.

Independent drops are binary64 numbers within inward representable endpoints.
The fixed pivot is the exact rational residual of the fixed endpoint sum.
One common rational contraction updates all drops. Directed joint rounding
selects floors, raising deterministic coordinates only if needed to retain the
pivot upper bound. This is part of a NEW coordinate representation, not clipping
or normalizing old failed trajectories. Every decoded gap is rounded ONCE from
an exact cumulative sum. Endpoint conservation has no accumulated-rounding term.
The analytic zero-chart tangent is for the continuous map BEFORE final rounding;
the stored finite map is quantized and is not asserted literally differentiable.
"""
from pathlib import Path
from fractions import Fraction as F
import json,math
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
P=D/'gap_coordinate_preparation_v3_466'
U=F(1,2**59) # Conservative ONE-ULP bound throughout0..0.012m.
def fq(v):return F.from_float(float(v))
def pair(v):return [v.numerator,v.denominator]
def unpack(v):return F(int(v[0]),int(v[1]))
def floor64(q):
 a=float(q)
 return math.nextafter(a,-math.inf) if fq(a)>q else a
def ceil64(q):
 a=float(q)
 return math.nextafter(a,math.inf) if fq(a)<q else a

def load_domain():
 L=json.loads((P/'layout.json').read_text());lo=list(map(unpack,L['lower']));hi=list(map(unpack,L['upper']));pivot=int(L['pivot']);ind=[i for i in range(len(lo)) if i!=pivot]
 low=lo.copy();upper=hi.copy()
 for i in ind:low[i]=fq(ceil64(lo[i]));upper[i]=fq(floor64(hi[i]))
 assert all(l<u for l,u in zip(low,upper));assert upper[pivot]-low[pivot]>=U
 total=unpack(L['total']);assert sum(low)<total<sum(upper)
 return dict(layout=L,lower=low,upper=upper,original_lower=lo,original_upper=hi,pivot=pivot,independent=ind,total=total,hover=unpack(L['target_gaps'][0]),terminal=unpack(L['target_gaps'][-1]))

def joint_round(q,d):
 """Bounded binary64 independents plus one EXACT residual coordinate.
 Floors sum <= exact independent sum <= allowable upper sum. If their sum
 is below the allowable lower sum, raise coordinates to adjacent ceilings in
 fixed index order. A raise is<=U, and pivot interval width>=U; first crossing
 therefore cannot overshoot. All ceilings together reach the required sum.
 """
 p=d['pivot'];ids=d['independent'];lo=d['lower'];hi=d['upper'];total=d['total']
 assert sum(q)==total and all(l<=v<=u for l,v,u in zip(lo,q,hi))
 z=[F(0)]*len(q)
 for i in ids:z[i]=fq(floor64(q[i]));assert lo[i]<=z[i]<=hi[i]
 S=sum(z);low=total-hi[p];upper=total-lo[p];raises=[]
 assert S<=upper
 for i in ids:
  if S>=low:break
  c=fq(ceil64(q[i]));step=c-z[i];assert 0<=step<=U and c<=hi[i]
  z[i]=c;S+=step;raises.append(i)
 assert low<=S<=upper
 z[p]=total-S
 assert sum(z)==total and all(l<=v<=u for l,v,u in zip(lo,z,hi))
 assert all(abs(z[i]-q[i])<=U for i in ids)
 assert abs(z[p]-q[p])<=len(ids)*U
 return z,raises

def initial(d):
 # One representation initialization of SAVED466 projected gap targets.
 # Each independent value is nearest among representable in-bound neighbors;
 # the dependent pivot is the exact residual. No complete path is evaluated.
 q=list(map(unpack,d['layout']['drops']));z=q.copy();p=d['pivot']
 for i in d['independent']:
  options=[fq(v) for v in [floor64(q[i]),ceil64(q[i])]]
  options=[v for v in options if d['lower'][i]<=v<=d['upper'][i]]
  assert options,(i,q[i]);z[i]=min(options,key=lambda v:(abs(v-q[i]),v));assert abs(z[i]-q[i])<=U
 z[p]=d['total']-sum(z[i] for i in d['independent'])
 assert all(l<=v<=u for l,v,u in zip(d['lower'],z,d['upper']));assert sum(z)==d['total']
 return z

def map_drops(origin,tangent,d):
 assert len(tangent)==len(d['independent']);assert sum(origin)==d['total']
 assert all(l<=v<=u for l,v,u in zip(d['lower'],origin,d['upper']))
 ds=[F(0)]*len(origin)
 for i,t in zip(d['independent'],tangent):ds[i]=fq(t)
 ds[d['pivot']]=-sum(ds);assert sum(ds)==0
 if not any(ds):return origin.copy(),dict(alpha=F(1),rho=F(0),unrounded=origin.copy(),direction=ds,raised=[])
 ratios=[]
 for i,v in enumerate(ds):
  if v==0:continue
  room=d['upper'][i]-origin[i] if v>0 else origin[i]-d['lower'][i]
  if room<=0:raise ValueError('Outward gap tangent at an exact endpoint')
  ratios.append(abs(v)/room)
 rho=max(ratios);alpha=1/(1+rho);q=[c+alpha*v for c,v in zip(origin,ds)]
 z,raised=joint_round(q,d)
 return z,dict(alpha=alpha,rho=rho,unrounded=q,direction=ds,raised=raised)

def gaps(drops,d):
 q=[d['hover']]
 for v in drops:q.append(q[-1]-v)
 assert q[-1]==d['terminal'];out=np.asarray([float(v) for v in q])
 assert all(abs(fq(v)-r)<=U for v,r in zip(out,q))
 return out,q

def fixture_center(d):
 # A SINGLE arithmetic interior fixture, never a path seed/candidate.
 a=(d['total']-sum(d['lower']))/(sum(d['upper'])-sum(d['lower']))
 q=[l+a*(u-l) for l,u in zip(d['lower'],d['upper'])]
 return joint_round(q,d)[0]
