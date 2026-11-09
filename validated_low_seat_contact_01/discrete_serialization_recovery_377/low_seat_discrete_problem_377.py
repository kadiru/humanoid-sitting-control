"""ONE new complete path formulation: fixed20ms segment counts and inward numerical envelopes.
Changes the whole nominal geometry/COM/force path, never a physical criterion or controller.
The analytic ramp envelope is a necessary construction screen; an exact legal-float32
forward-filter construction over ALL675 ticks is mandatory before schedule admission.
"""
from pathlib import Path
import numpy as np
from low_seat_support_problem_371 import make_problem
from low_seat_support_rows_371 import rows
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def allocate(duration):
 # One deterministic allocation, no integer/pose/weight sweep. Reserve101 normal
 # ticks: the necessary2s for12mm at6mm/s plus ONE20ms construction interval.
 # Other movement uses549ticks, plus25 separate synthetic quiet ticks=675 total.
 duration=np.asarray(duration,dtype=np.float64);ticks=np.zeros(114,np.int64)
 normal=np.arange(99,104);other=np.array([i for i in range(114) if i not in normal])
 for indices,budget in [(normal,101),(other,549)]:
  raw=duration[indices]/duration[indices].sum()*budget
  base=np.maximum(1,np.floor(raw).astype(np.int64));remaining=budget-int(base.sum())
  assert 0<=remaining<len(indices)
  order=np.lexsort((indices,-(raw-base)));base[order[:remaining]]+=1;ticks[indices]=base
 assert ticks.sum()==650 and ticks[99:104].sum()==101 and np.all(ticks>=1)
 return ticks

def make_discrete():
 p=make_problem();source=np.load(D/'support_path_373/final.npz');x0=source['x'].copy()
 ticks=allocate(x0[1026:]);x0[1026:]=ticks*.02
 scale,keep,hard,groups,proof=rows(p)
 # Fixed time row is an exact integer identity checked independently, not softened.
 keep[19759]=False
 # IEEE-derived CONSTRUCTION margins. These tighten, never relax physical criteria.
 eps=np.finfo(np.float32).eps
 contact_interior=8*eps;phase_interior=8*eps;support_interior=8*float(np.spacing(np.float32(1.)))
 magnitude=np.maximum(abs(p['desired_lo']),abs(p['desired_hi']))
 ulp=np.spacing(magnitude).astype(float)
 goal_round_envelope=ulp # endpoint rounding + interpolation rounding <=one ULP
 rate_per_tick_envelope=4*ulp # >two goal errors, leaving a further two ULP reserve
 target_envelope=32*ulp # >19*goal error in10*next-9*previous, plus expression reserve
 tighten=np.zeros(len(scale))
 for i in range(114):
  for j in range(8):
   for off in [64,72]:
    at=i*80+off+j
    if keep[at]:tighten[at]=contact_interior
 tighten[12004:12212]=phase_interior
 for start in [12212,14036]:tighten[start:start+1824]=(ticks[:,None]*rate_per_tick_envelope).ravel()
 for start in [15860,17684]:tighten[start:start+1824]=(ticks[:,None]*.02*target_envelope).ravel()
 tighten[19760:]=support_interior
 free=np.array([j for j in p['free'] if j<1026])
 def evaluate(x,derivative=False):
  if derivative:return p['evaluate'](x,True)
  eq,raw=p['evaluate'](x);return eq,raw-tighten
 def expand(z):
  x=x0.copy();x[free]=z;return x
 return locals()
