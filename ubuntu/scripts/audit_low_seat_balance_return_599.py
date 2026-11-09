"""Independent NEW returned-point audit, including any saved incomplete prefix. No producer imports."""
from pathlib import Path
from fractions import Fraction as F
import json,struct,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def pair(x):return [x.numerator,x.denominator]
def adjacent(v,up):
 bits=struct.unpack('>Q',struct.pack('>d',v))[0]
 if v==0:bits=1 if up else (1<<63)|1
 else:bits+=1 if (v>0)==up else -1
 value=struct.unpack('>d',struct.pack('>Q',bits))[0];assert math.isfinite(value);return value
def nearest(value,hexvalue,error):
 x=float.fromhex(hexvalue);assert math.isfinite(x);actual=F(x);assert actual-value==F(*error)
 if actual!=value:
  l=(F(adjacent(x,False))+actual)/2;r=(actual+F(adjacent(x,True)))/2;assert l<=value<=r
  if value==l or value==r:assert struct.unpack('>Q',struct.pack('>d',x))[0]&1==0
 return x,abs(actual-value)
def audit_point(u,source,out,kind):
 ratios=[float(x).as_integer_ratio() for x in u];maxa=F(0);maxe=F(0);counts={};products={};full=True
 for group,total in [('inequalities',265967)]+([('equalities',1367 if kind=='594' else 5462)] if kind in ['594','586'] else []):
  count=0;prod=0;log=out/(group+'_residuals.jsonl')
  if log.exists():
   with (source/('rational_'+group+'.jsonl')).open() as f,log.open() as g:
    for saved in g:
     source_row=next(f,None);assert source_row is not None
     sid,terms,rhs=json.loads(source_row)[:3];si,ss,n,d=json.loads(saved);assert si==count and ss==sid
     value=F(-rhs[0],rhs[1])
     for j,an,ad in terms:
      un,ud=ratios[j];value+=F(an*un,ad*ud);prod+=1
     assert value==F(n,d)
     if group=='inequalities':maxa=max(maxa,value)
     else:maxe=max(maxe,abs(value))
     count+=1
    if count==total:assert next(f,None) is None
  assert count<=total;full &= count==total;counts[group]=count;products[group]=prod
 result=dict(complete=bool(full),saved_rows_verified=counts,saved_point_products_verified=products,passed_certificate=False)
 if (out/'checks.json').exists():
  assert full;a=dict(np.load(source/'arrays.npz'));m=dict(np.load(source/'maps.npz'));keep=m['original_columns'] if kind in ['594','586'] else m['retained_columns'];analytic=dict(np.load(D/'zero_face_preparation_572/analytic_lower_bound_dual.npz'));lower=analytic['lower'][keep];upper=analytic['upper'][keep]
  st=a['cost']+upper-lower;eq=np.empty(0) if kind in ['594','586'] else sparse.load_npz(source/'retained_equality.npz')@u-a['equality_rhs'];equality=float(maxe) if kind in ['594','586'] else float(abs(eq).max(initial=0));lo=np.isfinite(a['bounds'][:,0]);hi=np.isfinite(a['bounds'][:,1])
  bound=max(0.,float((a['bounds'][:,0]-u).max()),float((u-a['bounds'][:,1]).max()));station=float(abs(st).max(initial=0));gap=abs(float(a['cost']@u+a['bounds'][hi,1]@upper[hi]-a['bounds'][lo,0]@lower[lo]))
  finite=bool(np.isfinite(eq).all() and np.isfinite(st).all() and np.isfinite([bound,equality,station,gap]).all());signs=bool(np.all(lower>=0) and np.all(upper>=0) and np.all(lower[~lo]==0) and np.all(upper[~hi]==0));other=dict(np.load(out/'other_residuals.npz'))
  for key,v in [('equality',eq),('stationarity',st),('lower',lower),('upper',upper)]:assert v.tobytes()==other[key].tobytes()
  expected=dict(passed=bool(maxa<=F(1,10**9) and maxe<=F(1,10**9) and finite and signs and max(bound,equality,station,gap)<=1e-9),maximum_exact_inequality_violation=pair(maxa),maximum_exact_equality_violation=pair(maxe) if kind in ['594','586'] else None,rows=counts,point_coefficient_products=products,equality=equality,bound=bound,stationarity=station,absolute_gap=gap,finite=finite,nonnegative_duals=signs,analytic_dual_inherited=True,point='unchanged backend10541' if kind=='594' else 'final once-rounded original vector projected to'+kind)
  saved=read(out/'checks.json')
  for key,v in expected.items():assert saved[key]==v,key
  result.update(checks=expected,passed_certificate=expected['passed'])
 return result

def foot_rows(source):
 with (source/'block_graphs.jsonl').open() as f:
  for line in f:
   bid,cols,rows,inputs,K,G,k,trace=json.loads(line)
   for i,col in enumerate(cols):yield col,inputs,G[i],k[i]
def balance_rows(source):
 with (source/'block_graphs.jsonl').open() as f:
  for line in f:
   bid,cols,rows,labels,inputs,K,G,k,trace=json.loads(line)
   for i,col in enumerate(cols):yield col,inputs,G[i],k[i]
def audit_lift(T,v,w):
 balance=D/'balance_block_preparation_594';foot=D/'foot_block_preparation_586';aux=D/'affine_output_preparation_578';bm=dict(np.load(balance/'maps.npz'));fm=dict(np.load(foot/'maps.npz'));am=dict(np.load(aux/'maps.npz'))
 exact586={int(j):F(*float(x).as_integer_ratio()) for j,x in zip(bm['retained_columns'],v)};worst=F(0);roots=0;bp=0;legs=0;fp=0;count=0;ap=0;exact={}
 if (T/'balance_point_lift.jsonl').exists():
  sources=balance_rows(balance)
  with (T/'balance_point_lift.jsonl').open() as log:
   for line in log:
    col,sv,hx,se=json.loads(line);sc,inputs,G,k=next(sources);assert col==sc;value=F(*k)
    for j,(n,d) in zip(inputs,G):
     x=exact586[j];value+=F(n*x.numerator,d*x.denominator);bp+=1
    assert value==F(*sv);rounded,err=nearest(value,hx,se);exact586[col]=value;worst=max(worst,err);roots+=1
    if w is not None:assert struct.pack('>d',rounded)==struct.pack('>d',float(w[fm['original_columns'][col]]))
  if roots==4095:assert next(sources,None) is None
 assert roots<=4095
 if roots==4095:
  assert len(exact586)==14636;exact={int(j):exact586[i] for i,j in enumerate(fm['retained_columns'])}
 if (T/'foot_point_lift.jsonl').exists():
  assert roots==4095
  sources=foot_rows(foot)
  with (T/'foot_point_lift.jsonl').open() as log:
   for line in log:
    col,sv,hx,se=json.loads(line);sc,inputs,G,k=next(sources);assert col==sc;value=F(*k)
    for j,(n,d) in zip(inputs,G):
     x=exact[j];value+=F(n*x.numerator,d*x.denominator);fp+=1
    assert value==F(*sv);rounded,err=nearest(value,hx,se);exact[col]=value;worst=max(worst,err);legs+=1
    if w is not None:assert struct.pack('>d',rounded)==struct.pack('>d',float(w[am['retained_columns'][col]]))
  if legs==16380:assert next(sources,None) is None
 assert legs<=16380
 if (T/'auxiliary_point_lift.jsonl').exists():
  assert legs==16380 and len(exact)==31016;original={int(j):exact[i] for i,j in enumerate(am['retained_columns'])}
  with (aux/'affine_graph.jsonl').open() as f,(T/'auxiliary_point_lift.jsonl').open() as log:
   for line in log:
    col,row,sv,hx,se=json.loads(line);sc,sr,d,h,terms=json.loads(next(f));assert col==sc==int(am['eliminated_columns'][count]) and row==sr;value=F(*h)
    for j,n,den in terms:
     x=original[j];value-=F(n*x.numerator,den*x.denominator);ap+=1
    value=F(value.numerator*d[1],value.denominator*d[0]);assert value==F(*sv);rounded,err=nearest(value,hx,se);worst=max(worst,err);count+=1
    if w is not None:assert struct.pack('>d',rounded)==struct.pack('>d',float(w[col]))
   if count==47775:assert next(f,None) is None
 assert count<=47775
 info=dict(rootxyf_outputs=roots,balance_point_products=bp,leg_outputs=legs,auxiliary_outputs=count,outputs=roots+legs+count,foot_point_products=fp,auxiliary_point_products=ap,exact_auxiliary_divisions=count,once_rounded_outputs=roots+legs+count,maximum_exact_rounding_error=pair(worst),old_auxiliary_graph_uses_unrounded_exact_legs=True,foot_and_auxiliary_graphs_use_unrounded_exact_rootxyf=True,retained_values_bitwise_backend=True,scalar_already_zero=True,clip_repair_normalize=False)
 complete=roots==4095 and legs==16380 and count==47775
 if (T/'graph_lift_checks.json').exists():assert complete and read(T/'graph_lift_checks.json')==info
 return dict(complete=complete,saved_point_arithmetic=info,completed_certificate_present=(T/'graph_lift_checks.json').exists())

def audit_return(T,u,w=None):
 bm=dict(np.load(D/'balance_block_preparation_594/maps.npz'));fm=dict(np.load(D/'foot_block_preparation_586/maps.npz'));assert u.shape==(10541,) and u.dtype==np.dtype('float64') and np.isfinite(u).all() and u[-1]==0
 if w is not None:assert w.shape==(78791,) and np.isfinite(w).all() and u.tobytes()==w[bm['original_columns']].tobytes()
 bcheck=audit_point(u,D/'balance_block_preparation_594',T/'rational594','594') if (T/'rational594').exists() else None
 lift=audit_lift(T,u,w);fcheck=None;acheck=None
 if (T/'rational586').exists():
  assert w is not None and lift['complete'];fcheck=audit_point(w[fm['original_columns']],D/'foot_block_preparation_586',T/'rational586','586')
 if (T/'rational578').exists():
  assert w is not None and lift['complete'];keep=np.load(D/'affine_output_preparation_578/maps.npz')['retained_columns'];acheck=audit_point(w[keep],D/'affine_output_preparation_578',T/'rational578','578')
 complete=all(x is not None and x['complete'] for x in [bcheck,fcheck,acheck]) and lift['complete']
 passed=bool(complete and all(x['passed_certificate'] for x in [bcheck,fcheck,acheck]))
 if (T/'exact_reduced_checks.json').exists():
  saved=read(T/'exact_reduced_checks.json');assert complete and saved['passed']==passed and saved['balance594']==read(T/'rational594/checks.json') and saved['foot586']==read(T/'rational586/checks.json') and saved['auxiliary578']==read(T/'rational578/checks.json')
 progress=read(T/'result.json')['new_point_arithmetic_progress']
 for kind,point in [('594',bcheck),('586',fcheck),('578',acheck)]:
  if point:
   for group,count in point['saved_rows_verified'].items():assert count==progress.get(kind+'_'+group+'_rows',0)
   for group,count in point['saved_point_products_verified'].items():assert count==progress.get(kind+'_'+group+'_products',0)
 for key,sourcekey in [('completed_rootxyf_outputs','rootxyf_outputs'),('balance_point_products','balance_point_products'),('completed_leg_outputs','leg_outputs'),('completed_auxiliary_outputs','auxiliary_outputs'),('foot_point_products','foot_point_products'),('auxiliary_point_products','auxiliary_point_products'),('auxiliary_point_divisions','exact_auxiliary_divisions')]:assert progress.get(key,0)==lift['saved_point_arithmetic'][sourcekey]
 return dict(exact_reduced=dict(passed=passed,complete=bool(complete),balance594=bcheck,foot586=fcheck,auxiliary578=acheck),graph_lift=lift,independent_original_point_lift_verified=bool(complete),old_graph_or_encoding_proofs_repeated=False,incomplete_prefix_is_not_certificate=not complete)
