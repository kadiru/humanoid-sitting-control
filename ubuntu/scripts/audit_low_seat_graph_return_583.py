"""Independent returned-point rational residuals and graph rounding audit; no producer."""
from pathlib import Path
from fractions import Fraction as F
import json,struct,itertools,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def adjacent(v,up):
 bits=struct.unpack('>Q',struct.pack('>d',v))[0]
 if v==0:bits=1 if up else (1<<63)|1
 else:bits+=1 if (v>0)==up else -1
 value=struct.unpack('>d',struct.pack('>Q',bits))[0];assert math.isfinite(value);return value
def audit_return(T,u,w):
 source=D/'affine_output_preparation_578';maps=dict(np.load(source/'maps.npz'));keep=maps['retained_columns'];elim=maps['eliminated_columns'];assert u.shape==(31016,) and w.shape==(78791,) and u.tobytes()==w[keep].tobytes() and u[-1]==0 and np.isfinite(w).all()
 ratios=[float(v).as_integer_ratio() for v in u];maximum=F(0);products=0;count=0
 with (source/'rational_inequalities.jsonl').open() as f,(T/'exact_reduced_residuals.jsonl').open() as g:
  for line,saved in itertools.zip_longest(f,g):
   assert line is not None and saved is not None
   i,terms,rhs,_=json.loads(line);si,n,d=json.loads(saved);assert i==si==count
   value=F(-rhs[0],rhs[1])
   for j,an,ad in terms:
    un,ud=ratios[j];value+=F(an*un,ad*ud);products+=1
   assert value==F(n,d);maximum=max(maximum,value);count+=1
 assert count==265967
 a=dict(np.load(source/'arrays.npz'));E=sparse.load_npz(source/'retained_equality.npz').tocsr();analytic=dict(np.load(D/'zero_face_preparation_572/analytic_lower_bound_dual.npz'));lower=analytic['lower'][keep];upper=analytic['upper'][keep]
 eq=E@u-a['equality_rhs'];st=a['cost']+upper-lower;lo=np.isfinite(a['bounds'][:,0]);hi=np.isfinite(a['bounds'][:,1]);bound=max(0.,float((a['bounds'][:,0]-u).max()),float((u-a['bounds'][:,1]).max()));equality=float(abs(eq).max(initial=0));station=float(abs(st).max(initial=0));gap=abs(float(a['cost']@u+a['bounds'][hi,1]@upper[hi]-a['bounds'][lo,0]@lower[lo]))
 finite=bool(np.isfinite(eq).all() and np.isfinite(st).all() and np.isfinite([bound,equality,station,gap]).all());signs=bool(np.all(lower>=0) and np.all(upper>=0) and np.all(lower[~lo]==0) and np.all(upper[~hi]==0))
 other=dict(np.load(T/'exact_reduced_other_residuals.npz'))
 for key,v in [('equality',eq),('stationarity',st),('lower',lower),('upper',upper)]:assert v.tobytes()==other[key].tobytes()
 result=dict(passed=bool(maximum<=F(1,10**9) and finite and signs and max(bound,equality,station,gap)<=1e-9),primal=max(float(maximum),bound,equality),maximum_exact_violation=[maximum.numerator,maximum.denominator],rows=count,point_coefficient_products=products,equality=equality,bound=bound,stationarity=station,absolute_gap=gap,finite=finite,nonnegative_duals=signs,analytic_dual_inherited=True)
 saved=read(T/'exact_reduced_checks.json')
 for key,value in result.items():assert saved[key]==value,key
 original_ratios={int(col):r for col,r in zip(keep,ratios)};count=0;products=0;worst=F(0)
 with (source/'affine_graph.jsonl').open() as f,(T/'graph_point_lift.jsonl').open() as g:
  for line,saved in itertools.zip_longest(f,g):
   assert line is not None and saved is not None
   col,row,d,h,terms=json.loads(line);sc,sr,sv,se=json.loads(saved);assert col==sc==int(elim[count]) and row==sr
   value=F(*h)
   for j,n,den in terms:
    un,ud=original_ratios[j];value-=F(n*un,den*ud);products+=1
   value=F(value.numerator*d[1],value.denominator*d[0]);assert value==F(*sv)
   actual=F(float(w[col]));error=actual-value;assert error==F(*se)
   if error:
    left=(F(adjacent(float(w[col]),False))+actual)/2;right=(actual+F(adjacent(float(w[col]),True)))/2;assert left<=value<=right
    if value==left or value==right:assert struct.unpack('>Q',struct.pack('>d',float(w[col])))[0]&1==0
   worst=max(worst,abs(error));count+=1
 assert count==47775
 expected=dict(outputs=count,point_coefficient_products=products,exact_graph_divisions=count,once_rounded_outputs=count,maximum_exact_rounding_error=[worst.numerator,worst.denominator],retained_values_bitwise_backend=True,scalar_already_zero=True,clip_repair_normalize=False)
 assert read(T/'graph_lift_checks.json')==expected
 return dict(exact_reduced=result,graph_lift=expected,independent_original_point_lift_verified=True,old_graph_or_encoding_proofs_repeated=False)
