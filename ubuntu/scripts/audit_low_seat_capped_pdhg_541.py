"""Independent exact diagonal norm, saved PDHG/proximal-gap and known-optimum audit. No solver import."""
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import json,hashlib,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ff(x):return F(float(x))
def metric_audit(H,S,m):
 # Independent rational sums, not the producer's integer-unit accumulation.
 K=sparse.vstack([H,S],format='csr');cols=[F(0) for _ in range(K.shape[1])];row_verified=np.zeros(K.shape[0],dtype=np.uint8);rowmax=F(0);colmax=F(0)
 assert np.array_equal(np.ldexp(np.ones(len(m['row_power'])),m['row_power']),m['sigma']) and np.array_equal(np.ldexp(np.ones(len(m['column_power'])),m['column_power']),m['tau'])
 for i in range(K.shape[0]):
  r=F(0)
  for j,a in zip(K.indices[K.indptr[i]:K.indptr[i+1]],K.data[K.indptr[i]:K.indptr[i+1]]):
   v=abs(ff(a));r+=v;cols[int(j)]+=v
  step=ff(m['sigma'][i]);assert step>0
  if r:assert F(1,4)<step*r<=F(1,2);rowmax=max(rowmax,step*r)
  else:assert step==1
  row_verified[i]=1
 cv=np.zeros(K.shape[1],dtype=np.uint8)
 for j,c in enumerate(cols):
  step=ff(m['tau'][j]);assert step>0
  if c:assert F(1,4)<step*c<=F(1,2);colmax=max(colmax,step*c)
  else:assert step==1
  cv[j]=1
 return row_verified,cv,dict(rows=len(row_verified),columns=len(cv),nonzeros=K.nnz,exact_row_product_max=str(rowmax),exact_column_product_max=str(colmax),exact_operator_norm_squared_bound='1/4',all_positive_finite=True,all_minimal_dyadic_step_bounds=True)
def main():
 assert read(D/'capped_pdhg_preparation_exit_540.json')['actual_exit']==0
 for p,h in read(D/'capped_pdhg_launch_540.json')['source_input_sha256'].items():assert sha(p)==h,p
 prep=D/'capped_pdhg_preparation_540'
 for p,h in read(prep/'SHA256.json').items():assert sha(prep/p)==h,p
 out=D/'capped_pdhg_audit_541';out.mkdir(exist_ok=False);systems={};tests={}
 for role in ['primary','witness']:
  src=D/'capped_max_preparation_538'/role;H=sparse.load_npz(src/'hard.npz');S=sparse.load_npz(src/'coupled.npz');m=dict(np.load(prep/role/'metric.npz'));rv,cv,res=metric_audit(H,S,m);np.savez(out/(role+'_step_bound_verified.npz'),rows=rv,columns=cv);systems[role]=res;print(json.dumps(dict(role=role,**res)),flush=True)
 for name in ['asymmetric_sum_primary','translated_clearance']:
  p=prep/name;H=sparse.load_npz(p/'hard.npz').toarray();S=sparse.load_npz(p/'coupled.npz').toarray();a=dict(np.load(p/'arrays.npz'));bH=a['bH'];bS=a['bS'];L,U,e=map(float,[a['L'],a['U'],a['e']]);m=dict(np.load(p/'solve/metric.npz'));rv,cv,met=metric_audit(sparse.csr_matrix(H),sparse.csr_matrix(S),m)
  known=read(p/'known_optimum.json');kx=list(map(ff,known['known_x']));ky=list(map(ff,known['known_hard_dual']));kl=list(map(ff,known['known_coupled_dual']));tu,tl=ff(known['known_upper']),ff(known['known_lower']);lo,hi=sorted([ff(e)*ff(L),ff(e)*ff(U)])
  for row,b in zip(H,bH):assert sum((ff(q)*x for q,x in zip(row,kx[:-1])),F(0))<=ff(b)
  for row,b in zip(S,bS):assert sum((ff(q)*x for q,x in zip(row,kx[:-1])),F(0))-ff(e)*kx[-1]<=ff(b)
  for j in range(H.shape[1]):assert sum((ff(H[i,j])*ky[i] for i in range(len(ky))),F(0))+sum((ff(S[i,j])*kl[i] for i in range(len(kl))),F(0))==0
  assert lo<=kx[-1]<=hi and all(q>=0 for q in ky+kl+[tu,tl]) and ff(e)-ff(e)*sum(kl)+tu-tl==0
  dobj=-sum((ff(b)*q for b,q in zip(bH,ky)),F(0))-sum((ff(b)*q for b,q in zip(bS,kl)),F(0))-hi*tu+lo*tl
  assert ff(e)*kx[-1]==dobj==ff(known['known_objective'])
  trace=dict(np.load(p/'solve/small_trace.npz'));records=read(p/'solve/iteration_metrics.json');count=len(records);assert count==len(trace['k'])<=4096
  sigH=m['sigma'][:len(bH)];sigS=m['sigma'][len(bH):];tau=m['tau'];maxgap=F(0);maxtransition=0.
  for k in range(count):
   z={key:val[k] for key,val in trace.items()};assert int(z['k'])==k
   if k==0:assert all(np.all(z[key]==0) for key in ['old_x','old_xbar','old_y','old_lambda'])
   else:
    for old,new in [('old_x','x'),('old_xbar','xbar'),('old_y','y'),('old_lambda','lam')]:assert np.array_equal(z[old],trace[new][k-1])
   checks=[z['wH']-(z['old_y']+sigH*(H@z['old_xbar'])),z['wS']-(z['old_lambda']+sigS*(S@z['old_xbar'])),z['y']-np.maximum(z['wH']-sigH*bH,0),z['x']-(z['old_x']-tau*(H.T@z['y']+S.T@z['lam'])),z['xbar']-(2*z['x']-z['old_x'])]
   err=max(float(np.max(abs(v),initial=0)) for v in checks);assert err<=1e-13;maxtransition=max(maxtransition,err)
   lam=list(map(ff,z['lam']));w=list(map(ff,z['wS']));step=list(map(ff,sigS));threshold=ff(z['threshold']);width=ff(U)-ff(L);mass=sum(lam);assert all(x>=0 for x in lam) and 0<=threshold<=width
   gap=width*max(F(0),mass-1)-threshold*(mass-1)
   for li,wi,si,bi in zip(lam,w,step,bS):
    q=wi-si*(ff(bi)+ff(L)+threshold);target=max(q,F(0));gap+=((li-target)**2+2*max(-q,F(0))*li)/(2*si)
   assert gap>=0 and gap<=F(1,10**10);maxgap=max(maxgap,gap)
   assert records[k]['proximal']['orthant_relative']<=1e-12 and records[k]['proximal']['hinge_relative']<=1e-12
  result=read(p/'solve/result.json');assert result['status']==0 and result['candidate'] and result['iterations']==count and result['elapsed_s']<=45
  can=dict(np.load(p/'solve/candidate.npz'));x=can['x'][:-1];t=can['x'][-1];y=can['hard_dual'];lam=can['coupled_dual'];upper=float(can['upper_multiplier']);lower=float(can['lower_multiplier']);assert np.array_equal(x,trace['x'][-1]) and np.array_equal(y,trace['y'][-1]) and np.array_equal(lam,trace['lam'][-1])
  primal=max(0.,float(np.max(H@x-bH,initial=0)),float(np.max(S@x-e*t-bS,initial=0)),float(lo)-t,t-float(hi));dual=max(float(np.max(abs(H.T@y+S.T@lam),initial=0)),abs(e-e*float(np.sum(lam,dtype=np.longdouble))+upper-lower));gap=abs(e*t+bH@y+bS@lam+float(hi)*upper-float(lo)*lower)
  assert all(np.all(np.isfinite(v)) for v in can.values()) and np.all(y>=0) and np.all(lam>=0) and upper>=0 and lower>=0 and max(primal,dual,gap)<=1e-9
  exactg=max([ff(L)]+[sum((ff(v)*ff(xx) for v,xx in zip(row,x)),F(0))-ff(b) for row,b in zip(S,bS)]);sr=read(p/'solve/scalar_reconstruction.json');assert F(sr['exact_canonical_scalar'])==exactg and float(exactg)==e*t and F(sr['rounding_error'])==ff(e*t)-exactg
  tests[name]=dict(exact_known_primal_dual_optimum=True,known_objective=known['known_objective'],saved_PDHG_transitions_verified=count,maximum_transition_association_error=maxtransition,maximum_exact_saved_proximal_gap=float(maxgap),exact_proximal_gap_scope='Primal-dual objective gap bounds saved local proximal error; finite fixed screens are not a summable-error convergence theorem.',primal=primal,dual=dual,gap=float(gap),objective_error=float(e*t-known['known_objective']),scalar_rounding_record=sr,metrics=met,independent_LP_calls=0)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,systems=systems,tests=tests,new_small_LP_calls_in_producer=2,independent_LP_calls=0,full_size_LP_calls=0,all_saved_small_transitions_verified=True,production_imported=False,prior538539_proof_inherited_without_repeat=True,model_calls=0,root_calls=0,finite_maps=0,new_whole_path_wrapper=False,physical_admission=False)
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
