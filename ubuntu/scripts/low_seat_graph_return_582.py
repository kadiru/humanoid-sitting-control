"""New returned-point exact rational checks and once-rounded auxiliary graph lift."""
from pathlib import Path
from fractions import Fraction as F
import json,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def rational_check(u,source,out,analytic,check):
 values=list(map(lambda x:F(float(x)),u));maximum=F(0);count=0;products=0
 with (source/'rational_inequalities.jsonl').open() as f,(out/'exact_reduced_residuals.jsonl').open('x') as log:
  for line in f:
   i,terms,b,_=json.loads(line)
   if i%64==0:check()
   assert i==count
   residual=sum((F(n,d)*values[j] for j,n,d in terms),F(0))-F(*b)
   maximum=max(maximum,residual);count+=1;products+=len(terms)
   log.write(json.dumps([i,residual.numerator,residual.denominator],separators=(',',':'))+'\n')
 assert count==265967;check()
 a=dict(np.load(source/'arrays.npz'));E=sparse.load_npz(source/'retained_equality.npz').tocsr();keep=np.load(source/'maps.npz')['retained_columns'];lower=analytic['lower'][keep];upper=analytic['upper'][keep]
 eq=E@u-a['equality_rhs'];st=a['cost']+upper-lower;lo=np.isfinite(a['bounds'][:,0]);hi=np.isfinite(a['bounds'][:,1])
 bound=max(0.,float((a['bounds'][:,0]-u).max()),float((u-a['bounds'][:,1]).max()));equality=float(abs(eq).max(initial=0));station=float(abs(st).max(initial=0));gap=abs(float(a['cost']@u+a['bounds'][hi,1]@upper[hi]-a['bounds'][lo,0]@lower[lo]))
 finite=bool(np.isfinite(eq).all() and np.isfinite(st).all() and np.isfinite([bound,equality,station,gap]).all());signs=bool(np.all(lower>=0) and np.all(upper>=0) and np.all(lower[~lo]==0) and np.all(upper[~hi]==0))
 np.savez_compressed(out/'exact_reduced_other_residuals.npz',equality=eq,stationarity=st,lower=lower,upper=upper)
 return dict(passed=bool(maximum<=F(1,10**9) and finite and signs and max(bound,equality,station,gap)<=1e-9),primal=max(float(maximum),bound,equality),maximum_exact_violation=[maximum.numerator,maximum.denominator],rows=count,point_coefficient_products=products,equality=equality,bound=bound,stationarity=station,absolute_gap=gap,finite=finite,nonnegative_duals=signs,analytic_dual_inherited=True,scope='Exact rational inequality residuals at returned point plus explicit original reduced E/bounds/stationarity/gap checks. All row/E multipliers zero,lower/upper copied from inherited573. Numerical tolerances do not assert exact feasibility.')
def lift(u,source,out,check):
 maps=dict(np.load(source/'maps.npz'));keep=maps['retained_columns'];elim=maps['eliminated_columns'];w=np.empty(78791,dtype=np.float64);w[keep]=u
 assert w[keep].tobytes()==u.tobytes() and u.shape==(31016,) and u[-1]==0 and np.isfinite(u).all()
 values={int(j):F(float(v)) for j,v in zip(keep,u)};count=0;products=0;worst=F(0)
 with (source/'affine_graph.jsonl').open() as f,(out/'graph_point_lift.jsonl').open('x') as log:
  for line in f:
   col,row,d,h,terms=json.loads(line)
   if count%64==0:check()
   assert col==int(elim[count])
   exact=(F(*h)-sum((F(n,den)*values[j] for j,n,den in terms),F(0)))/F(*d)
   rounded=float(exact)
   if not math.isfinite(rounded):raise ValueError('nonfinite once-rounded auxiliary graph value')
   error=F(rounded)-exact;w[col]=rounded;worst=max(worst,abs(error));products+=len(terms);count+=1
   log.write(json.dumps([col,row,[exact.numerator,exact.denominator],[error.numerator,error.denominator]],separators=(',',':'))+'\n')
 assert count==47775 and w[keep].tobytes()==u.tobytes() and np.isfinite(w).all();check()
 np.save(out/'returned_native.npy',w)
 return w,dict(outputs=count,point_coefficient_products=products,exact_graph_divisions=count,once_rounded_outputs=count,maximum_exact_rounding_error=[worst.numerator,worst.denominator],retained_values_bitwise_backend=True,scalar_already_zero=True,clip_repair_normalize=False)
