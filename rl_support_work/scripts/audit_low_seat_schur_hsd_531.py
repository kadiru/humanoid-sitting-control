"""Independent scaling/certificates/iteration audit; no production import or solver replay."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'schur_hsd_preparation_530';O=D/'schur_hsd_audit_531'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def exact_power_check(a,b,p):
 # frexp is an exact binary decomposition, including subnormal input values.
 p=np.broadcast_to(p,a.shape);nz=a!=0;assert np.all(b[~nz]==0)
 m,e=np.frexp(a[nz]);n,f=np.frexp(b[nz]);assert np.array_equal(m,n) and np.array_equal(f-e,p[nz]);assert np.all(np.isfinite(b))
def same(a,b):
 assert a.shape==b.shape
 for key in ['data','indices','indptr']:assert np.array_equal(getattr(a,key),getattr(b,key))
def check_scaling(A,b,c,bounds,S,z,U):
 rows=[];vals=[];cols=[];rhs=[];owners=[]
 for j in range(len(c)):
  if np.isfinite(bounds[j,1]):owners.append((j,1));rows.append(len(rhs));cols.append(j);vals.append(1.);rhs.append(bounds[j,1])
  if np.isfinite(bounds[j,0]):owners.append((j,-1));rows.append(len(rhs));cols.append(j);vals.append(-1.);rhs.append(-bounds[j,0])
 appended=sparse.vstack([A,sparse.csr_matrix((vals,(rows,cols)),shape=(len(rhs),len(c)))],format='csr');same(appended,U);assert np.array_equal(z['original_rhs'],np.r_[b,rhs]);assert np.array_equal(z['original_cost'],c);assert np.array_equal(z['bound_owners'],np.asarray(owners).reshape(-1,2))
 assert np.array_equal(S.indptr,U.indptr) and np.array_equal(S.indices,U.indices)
 ids=np.repeat(np.arange(U.shape[0]),np.diff(U.indptr));rp=z['row_power'];cp=z['column_power'];exact_power_check(U.data,S.data,rp[ids]+cp[U.indices]);exact_power_check(z['original_rhs'],z['rhs'],rp);exact_power_check(c,z['cost'],cp)
 expected=-np.frexp(np.maximum.reduce([np.ones(U.shape[0]),np.asarray(abs(U).sum(axis=1)).ravel(),abs(z['original_rhs'])]))[1];assert np.array_equal(expected,rp)
 ar=U.copy();ar.data=np.ldexp(ar.data,rp[ids]);cn=np.sqrt(np.asarray(ar.power(2).sum(axis=0)).ravel());assert np.array_equal(cp,np.where(cn>0,-np.frexp(cn)[1],0))
 return dict(source_rows=A.shape[0],cone_rows=U.shape[0],nonzeros=S.nnz,added_bounds=len(owners),all_coefficients_exact_dyadic=True,source_copy_error=0)
def exact_certificate(A,b,c,x,y):
 a=[[F(float(v)) for v in row] for row in A];bb=list(map(lambda x:F(float(x)),b));cc=list(map(lambda x:F(float(x)),c));xx=list(map(lambda x:F(float(x)),x));yy=list(map(lambda x:F(float(x)),y))
 assert all(q>=0 for q in yy);assert all(sum(v*t for v,t in zip(row,xx))<=bi for row,bi in zip(a,bb))
 assert all(sum(a[i][j]*yy[i] for i in range(len(a)))+cc[j]==0 for j in range(len(c)))
 obj=sum(v*t for v,t in zip(cc,xx));assert obj+sum(v*t for v,t in zip(bb,yy))==0;return str(obj)
def main():
 assert read(D/'schur_hsd_preparation_exit_530.json')['actual_exit']==0
 for p,h in read(D/'schur_hsd_launch_530.json')['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 O.mkdir(exist_ok=False);systems={};tests={}
 for role in ['primary','witness']:
  src=D/'torque_projection_preparation_522'/role;out=P/role;A=sparse.load_npz(src/'matrix.npz');a=dict(np.load(src/'arrays.npz'));E=sparse.load_npz(src/'equality.npz');assert np.all(E.data==0) and np.all(a['equality_rhs']==0)
  z=dict(np.load(out/'arrays.npz'));S=sparse.load_npz(out/'matrix.npz');U=sparse.load_npz(out/'cone_original.npz');systems[role]=check_scaling(A,a['rhs'],a['cost'],a['bounds'],S,z,U)
  old=np.load(D/'arrow_operator_preparation_528'/role/'partition.npz')
  for key in ['local','separator','block_ptr']:assert np.array_equal(old[key],z[key])
  assert all(j in set(z['separator']) for j in z['bound_owners'][:,0]);systems[role]['529_support_proof_inherited_not_repeated']=True
 for name in ['infeasible_start_primary','clearance_maximization']:
  out=P/name;t=dict(np.load(out/'input.npz'));A=sparse.csr_matrix(t['A']);S=sparse.load_npz(out/'scaled_matrix.npz');U=sparse.load_npz(out/'cone_original.npz');z=dict(np.load(out/'scaled_arrays.npz'));check_scaling(A,t['b'],t['c'],t['bounds'],S,z,U)
  obj=exact_certificate(U.toarray(),z['original_rhs'],t['c'],t['exact_x'],t['exact_y']);n=S.shape[1];m=S.shape[0];Q=np.zeros((n+m+1,n+m+1));Q[:n,n:n+m]=S.T.toarray();Q[n:n+m,:n]=-S.toarray();Q[:n,-1]=z['cost'];Q[-1,:n]=-z['cost'];Q[n:n+m,-1]=z['rhs'];Q[-1,n:n+m]=-z['rhs'];assert np.array_equal(Q,-Q.T)
  records=dict(np.load(out/'iterations.npz')) if (out/'iterations.npz').exists() else {};metrics=read(out/'iteration_metrics.json') if (out/'iteration_metrics.json').exists() else [];errs=[]
  for k in range(len(metrics)):
   ou=records['old_u'][k];ov=records['old_v'][k];tt=records['tilde'][k];un=records['u'][k];vn=records['v'][k]
   if k==0:
    want=np.zeros(n+m+1);want[-1]=1.;assert np.array_equal(ou,want) and np.array_equal(ov,want)
   else:assert np.array_equal(ou,records['u'][k-1]) and np.array_equal(ov,records['v'][k-1])
   err=float(np.linalg.norm(tt+Q@tt-ou-ov));assert err<=metrics[k]['epsilon']+1e-14
   want=tt-ov;want[n:]=np.maximum(want[n:],0.);assert np.array_equal(want,un) and np.array_equal(ov-tt+un,vn)
   assert np.all(un[n:]>=0) and np.all(vn[n:]>=0) and np.all(vn[:n]==0);errs.append(err)
  result=read(out/'result.json');certified=False;finalmetrics=None
  if result['candidate_returned']:
   q=dict(np.load(out/'candidate.npz'));last=records['u'][-1];v=records['v'][-1]
   for key,want in [('x',np.ldexp(last[:n]/last[-1],z['column_power'])),('y',np.ldexp(last[n:n+m]/last[-1],z['row_power'])),('slack',np.ldexp(v[n:n+m]/last[-1],-z['row_power']))]:assert np.array_equal(q[key],want)
   pr=np.max(abs(U@q['x']+q['slack']-z['original_rhs']));du=np.max(abs(U.T@q['y']+t['c']));gap=abs(t['c']@q['x']+z['original_rhs']@q['y']);assert max(pr,du,gap)<=1e-9 and np.all(q['y']>=0) and np.all(q['slack']>=0)
   objerr=float(t['c']@q['x'])-float(F(obj));lower_allowance=float(np.sum(t['exact_y'])*max(0,np.max(U@q['x']-z['original_rhs'])));upper_allowance=float(gap+du*np.sum(abs(t['exact_x'])))
   assert -lower_allowance-1e-13<=objerr<=upper_allowance+1e-13
   certified=True;finalmetrics=dict(primal=float(pr),dual=float(du),gap=float(gap),objective_error=objerr,objective_lower_error_allowance=lower_allowance,objective_upper_error_allowance=upper_allowance)
  np.savez_compressed(O/(name+'_verification.npz'),resolvent_residuals=errs)
  tests[name]=dict(exact_optimum=obj,independent_exact_primal_dual_certificate=True,iterations_checked=len(metrics),maximum_resolvent_residual=max(errs,default=None),returned_numeric_witness_certified=certified,final_metrics=finalmetrics,solver_repeated=False,status=result['status'],stop_reason=result['reason'])
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,small_tests_returned_certified_witnesses=all(t['returned_numeric_witness_certified'] for t in tests.values()),systems=systems,small_tests=tests,new_full_size_LP_calls=0,independent_LP_solver_calls=0,old528_partition_factor_fixture_repeated=False,new_actual_path=False,physical_admission=False,scope='Full source dyadic scaling/support provenance and two NEW small LPs audited without replaying their solver. No implication for full-size convergence,45s performance,actual feasibility or original torque backsubstitution.')
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
