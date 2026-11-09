"""NEW point checks and composite exact graph lift. No old coefficient proof replay."""
from pathlib import Path
from fractions import Fraction as F
import json,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def pair(x):return [x.numerator,x.denominator]
def emit(log,x):log.write(json.dumps(x,separators=(',',':'))+'\n')
def exact_point(u,source,out,analytic,check,progress,kind):
 out.mkdir();values=[F(float(v)) for v in u];maximum=F(0);eqmax=F(0);counts={};products={}
 groups=['inequalities','equalities'] if kind in ['594','586'] else ['inequalities']
 for group in groups:
  count=0;prod=0
  with (source/('rational_'+group+'.jsonl')).open() as f,(out/(group+'_residuals.jsonl')).open('x') as log:
   for line in f:
    if count%64==0:check()
    data=json.loads(line);sid,terms,rhs=data[:3]
    residual=sum((F(n,d)*values[j] for j,n,d in terms),F(0))-F(*rhs)
    if group=='inequalities':maximum=max(maximum,residual)
    else:eqmax=max(eqmax,abs(residual))
    emit(log,[count,sid,*pair(residual)]);count+=1;prod+=len(terms)
    progress[kind+'_'+group+'_rows']=count;progress[kind+'_'+group+'_products']=prod
  counts[group]=count;products[group]=prod;check()
 assert counts['inequalities']==265967
 if kind in ['594','586']:assert counts['equalities']==(1367 if kind=='594' else 5462)
 a=dict(np.load(source/'arrays.npz'));maps=dict(np.load(source/'maps.npz'));keep=maps['original_columns'] if kind in ['594','586'] else maps['retained_columns']
 lower=analytic['lower'][keep];upper=analytic['upper'][keep];st=a['cost']+upper-lower
 eq=np.empty(0) if kind in ['594','586'] else sparse.load_npz(source/'retained_equality.npz')@u-a['equality_rhs']
 equality=float(eqmax) if kind in ['594','586'] else float(abs(eq).max(initial=0));lo=np.isfinite(a['bounds'][:,0]);hi=np.isfinite(a['bounds'][:,1])
 bound=max(0.,float((a['bounds'][:,0]-u).max()),float((u-a['bounds'][:,1]).max()));station=float(abs(st).max(initial=0));gap=abs(float(a['cost']@u+a['bounds'][hi,1]@upper[hi]-a['bounds'][lo,0]@lower[lo]))
 finite=bool(np.isfinite(eq).all() and np.isfinite(st).all() and np.isfinite([bound,equality,station,gap]).all());signs=bool(np.all(lower>=0) and np.all(upper>=0) and np.all(lower[~lo]==0) and np.all(upper[~hi]==0))
 np.savez_compressed(out/'other_residuals.npz',equality=eq,stationarity=st,lower=lower,upper=upper)
 result=dict(passed=bool(maximum<=F(1,10**9) and eqmax<=F(1,10**9) and finite and signs and max(bound,equality,station,gap)<=1e-9),maximum_exact_inequality_violation=pair(maximum),maximum_exact_equality_violation=pair(eqmax) if kind in ['594','586'] else None,rows=counts,point_coefficient_products=products,equality=equality,bound=bound,stationarity=station,absolute_gap=gap,finite=finite,nonnegative_duals=signs,analytic_dual_inherited=True,point='unchanged backend10541' if kind=='594' else 'final once-rounded original vector projected to'+kind,scope='New returned-point checks only; numerical gates do not assert exact rational feasibility or optimality.')
 (out/'checks.json').write_text(json.dumps(result,indent=2));check();return result

def lift(v,balance,foot,aux,out,check,progress):
 bm=dict(np.load(balance/'maps.npz'));fm=dict(np.load(foot/'maps.npz'));am=dict(np.load(aux/'maps.npz'));keep=bm['original_columns'];keep578=am['retained_columns'];w=np.empty(78791,dtype=np.float64);w[keep]=v
 assert v.shape==(10541,) and v[-1]==0 and np.isfinite(v).all() and w[keep].tobytes()==v.tobytes()
 exact586={int(j):F(float(value)) for j,value in zip(bm['retained_columns'],v)};rootcount=0;rootprod=0;worst=F(0)
 with (balance/'block_graphs.jsonl').open() as f,(out/'balance_point_lift.jsonl').open('x') as log:
  for line in f:
   check();bid,cols,rows,labels,inputs,K,G,k,trace=json.loads(line)
   for i,col in enumerate(cols):
    value=F(*k[i])+sum((F(*c)*exact586[j] for j,c in zip(inputs,G[i])),F(0));rounded=float(value)
    if not math.isfinite(rounded):raise ValueError('nonfinite once-rounded rootxy/f output')
    error=F(rounded)-value;exact586[col]=value;w[fm['original_columns'][col]]=rounded;rootcount+=1;rootprod+=len(inputs);worst=max(worst,abs(error))
    emit(log,[col,pair(value),rounded.hex(),pair(error)]);progress.update(completed_rootxyf_outputs=rootcount,balance_point_products=rootprod)
 assert rootcount==4095 and len(exact586)==14636;check()
 # Neither rootxy/f nor leg rounding feeds the next graph: all graphinputs stayexact.
 exact={int(j):exact586[i] for i,j in enumerate(fm['retained_columns'])};footcount=0;footprod=0
 with (foot/'block_graphs.jsonl').open() as f,(out/'foot_point_lift.jsonl').open('x') as log:
  for line in f:
   check();bid,cols,rows,inputs,K,G,k,trace=json.loads(line)
   for i,col in enumerate(cols):
    value=F(*k[i])+sum((F(*c)*exact[j] for j,c in zip(inputs,G[i])),F(0));rounded=float(value)
    if not math.isfinite(rounded):raise ValueError('nonfinite once-rounded leg output')
    error=F(rounded)-value;exact[col]=value;w[keep578[col]]=rounded;footcount+=1;footprod+=len(inputs);worst=max(worst,abs(error))
    emit(log,[col,pair(value),rounded.hex(),pair(error)]);progress.update(completed_leg_outputs=footcount,foot_point_products=footprod)
 assert footcount==16380 and len(exact)==31016;check()
 originalexact={int(col):exact[i] for i,col in enumerate(keep578)};count=0;products=0
 with (aux/'affine_graph.jsonl').open() as f,(out/'auxiliary_point_lift.jsonl').open('x') as log:
  for line in f:
   if count%64==0:check()
   col,row,d,h,terms=json.loads(line);assert col==int(am['eliminated_columns'][count])
   value=(F(*h)-sum((F(n,den)*originalexact[j] for j,n,den in terms),F(0)))/F(*d);rounded=float(value)
   if not math.isfinite(rounded):raise ValueError('nonfinite once-rounded auxiliary output')
   error=F(rounded)-value;w[col]=rounded;worst=max(worst,abs(error));products+=len(terms);count+=1
   emit(log,[col,row,pair(value),rounded.hex(),pair(error)]);progress.update(completed_auxiliary_outputs=count,auxiliary_point_products=products,auxiliary_point_divisions=count)
 assert count==47775 and w[keep].tobytes()==v.tobytes() and np.isfinite(w).all();check();np.save(out/'returned_native.npy',w)
 info=dict(rootxyf_outputs=rootcount,balance_point_products=rootprod,leg_outputs=footcount,auxiliary_outputs=count,outputs=rootcount+footcount+count,foot_point_products=footprod,auxiliary_point_products=products,exact_auxiliary_divisions=count,once_rounded_outputs=rootcount+footcount+count,maximum_exact_rounding_error=pair(worst),old_auxiliary_graph_uses_unrounded_exact_legs=True,foot_and_auxiliary_graphs_use_unrounded_exact_rootxyf=True,retained_values_bitwise_backend=True,scalar_already_zero=True,clip_repair_normalize=False)
 (out/'graph_lift_checks.json').write_text(json.dumps(info,indent=2));check();return w,info
