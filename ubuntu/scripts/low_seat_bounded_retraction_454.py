"""NEW representably bounded coordinate retraction for complete clock paths.

Research input is the ORIGINAL saved440 seed and unchanged444/446v2 domain.
450/453 motivates the change but no450 trial/final point initializes this run.
For a valid control c, tangent d moves toward its fixed interval endpoint b:
 a=-expm1(-abs(d)/abs(b-c)); C=(1-a)c+a*b.
The last blend is evaluated as an exact rational of the BINARY64 operands,
then rounded once to binary64. Thus C is representably inside [c,b]. This
constructs NEW coordinates; it never clips a prior trial, force or target.
Outward directions at an exact endpoint are rejected, not replaced by zero.
At zero the feasible one-sided derivative is identity; at nonzero d it is
exp(-abs(d)/room). Normal balance remains an explicit nonlinear equality.
All original hard/soft rows, reserves, fixed states and absolute boxes remain.
"""
import math
from fractions import Fraction
import numpy as np
from low_seat_collocation_solver_448 import load_saved,values,linearization,summary,assemble,solve_lp,Retraction,D,read,sha
from low_seat_collocation_solver_448 import SPEC as ORIGINAL_SPEC

SPEC=dict(ORIGINAL_SPEC)
SPEC.update(max_predictors=24,max_corrections_per_predictor=4,max_linear_programs=120,
 coordinate_retraction='endpoint exponential with exact-rational binary64 convex blend',
 seed='immutable440 original; never450 final or rejected trials',
 budget_justification='18 half-reductions cover source merit .1979113 down to the unchanged 2^-20 step guard; six additional predictors bound rejected steps. Four corrections bound new nonlinear normal-balance/geometry defects. No extension is allowed.',
 correction_target='unchanged half predicted soft improvement; hard rows have no slack',
 physical_admission=False)

def bounded_controls(origin,tangent,lower,upper):
 origin=np.asarray(origin);tangent=np.asarray(tangent);lower=np.asarray(lower);upper=np.asarray(upper)
 assert origin.shape==tangent.shape==lower.shape==upper.shape
 if not (np.isfinite(origin).all() and np.isfinite(tangent).all() and np.all(lower<=origin) and np.all(origin<=upper)):
  raise ValueError('invalid coordinate-map origin or tangent')
 out=origin.copy();alpha=np.zeros(len(origin));endpoint=origin.copy()
 for i in np.flatnonzero(tangent):
  c=float(origin[i]);d=float(tangent[i]);b=float(upper[i] if d>0 else lower[i]);room=abs(b-c)
  if room==0:raise ValueError('outward tangent at an exact coordinate endpoint: '+str(int(i)))
  a=-math.expm1(-abs(d)/room)
  if not (0<=a<=1):raise ValueError('nonfinite or invalid exponential blend')
  # Exact convex combination of representable operands, NOT clipping.
  q=(1-Fraction.from_float(a))*Fraction.from_float(c)+Fraction.from_float(a)*Fraction.from_float(b)
  out[i]=float(q);alpha[i]=a;endpoint[i]=b
 if not (np.all(lower<=out) and np.all(out<=upper)):raise ArithmeticError('exact convex-blend containment failed')
 return out,alpha,endpoint

class BoundedRetraction(Retraction):
 def run_bounded(self,a,tangent,lower,upper):
  origin=a['z'];controls,alpha,endpoint=bounded_controls(origin,tangent,lower,upper)
  # Existing14-equation root receives the NEW mapped controls. It does not
  #choose f or alter loads. Explicit overwrite avoids cancellation in c+(C-c).
  d=self.d;self.counts['trial_paths']+=1;dc=np.zeros(d['c']['reference'].size)
  dc[d['free']]=controls-origin;dc=dc.reshape(-1,10)
  x=a['X']+np.einsum('nic,nc->ni',a['T'],dc)
  allcontrols=a['X'][:,d['c']['control']].copy().ravel();allcontrols[d['free']]=controls
  x[:,d['c']['control']]=allcontrols.reshape(-1,10);x[d['g']['fixed']]=d['initialX'][d['g']['fixed']]
  dep=d['c']['dependent'];history=[];mapped=dict(mapped_controls=controls,alpha=alpha,endpoint=endpoint,map_lower=lower,map_upper=upper)
  for it in range(SPEC['root_max_updates']+1):
   y=self.val(x);res=y[:,:14];bad=(np.max(abs(res),axis=1)>SPEC['root_absolute_tolerance'])&~d['g']['fixed'];history.append(float(abs(res).max()))
   if not np.any(bad):return dict(passed=True,X=x,Y=y,root_history=history,**mapped)
   if it==SPEC['root_max_updates']:break
   j=self.jac(x)
   try:step=np.linalg.solve(j[:,:14][:,:,dep],-res[...,None])[...,0]
   except np.linalg.LinAlgError:return dict(passed=False,reason='singular root Jacobian',X=x,Y=y,root_history=history,**mapped)
   norm=np.linalg.norm(res,axis=1);accepted=~bad
   for h in range(SPEC['root_max_halvings']):
    active=np.flatnonzero(~accepted);candidate=x.copy();candidate[active[:,None],dep]+=step[active]*2.**(-h);v=self.val(candidate);take=(~accepted)&(np.linalg.norm(v[:,:14],axis=1)<norm);x[take]=candidate[take];accepted|=take
    if accepted.all():break
   self.counts['root_updates']+=1
   if not accepted.all():return dict(passed=False,reason='root line search cap',X=x,Y=self.val(x),root_history=history,**mapped)
  return dict(passed=False,reason='root update cap',X=x,Y=y,root_history=history,**mapped)

def map_interval(d,a,trust,origin):
 radius=SPEC['coordinate_domain_radius']*trust
 lo=np.maximum(d['lower'],origin-radius);hi=np.minimum(d['upper'],origin+radius)
 if a['mode']=='correction':
  lo=np.maximum(lo,a['z']-SPEC['correction_radius_fraction']*radius)
  hi=np.minimum(hi,a['z']+SPEC['correction_radius_fraction']*radius)
 # Intersecting declared domains constructs the map, not an output clipping.
 return lo,hi
