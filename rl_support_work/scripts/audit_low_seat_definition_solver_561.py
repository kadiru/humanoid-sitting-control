"""Independent560 input identity, full native primal/dual KKT and terminal resource audit; no solver import/call."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def audit_solver(T):
 r=read(T/'result.json');actual=(T/'scipy_started.json').exists();assert r['actual_lp_call']==actual and r['full_size_robot_LP']==actual and not r['small_test'] and r['input_loading_inside_clock']
 assert r['method']=='exact-definition-extended-IPM' and r['configured_address_space_bytes']==40*1024**3 and not r['peak_memory_measured'];Q=T.parent
 A=sparse.load_npz(Q/'matrix.npz');a=dict(np.load(Q/'arrays.npz'));E=sparse.load_npz(Q/'equality.npz');assert E.nnz==0 and np.all(a['equality_rhs']==0)
 out=dict(passed=True,actual_lp_call=actual,numerical_candidate=bool(r['candidate']),physical_admission=False,backend_internal_iterates_available=False,backend_iterations_replayed=False,coordinate_reconstruction_attempts=0)
 if actual:
  cfg=read(T/'scipy_started.json');opt=cfg['options'];assert cfg['cold_start'] and not cfg['old_internal_state_used'] and cfg['method']=='highs-ipm'
  assert 0<opt['time_limit']<=45 and opt['maxiter']==4096 and opt['presolve'] and all(opt[k]==1e-9 for k in ['primal_feasibility_tolerance','dual_feasibility_tolerance','ipm_optimality_tolerance'])
  if (T/'backend_result.json').exists():
   b=read(T/'backend_result.json');assert b['status']==r['native_status'] and b['iterations']==r['iterations'] and b['crossover_iterations']==r['crossover_iterations'] and b['candidate_returned']==r['native_candidate_returned'] and b['elapsed_s']==r['backend_elapsed_s'] and b['configured_time_limit_s']==opt['time_limit']
   out['backend_result']=b
 if not (T/'input_proof.json').exists():
  assert not actual and not r['candidate'] and not (T/'witness.npy').exists();out['input_preparation_completed']=False;return out
 proof=read(T/'input_proof.json');assert proof['passed'];frozen=read(D/'definition_path_launch_560.json')['source_input_sha256']
 for p,h in {**proof['input_sha256'],**proof['full_input_sha256']}.items():assert sha(p)==h
 if proof['inherited']:
  for p,h in proof['input_sha256'].items():assert h==frozen[p]
  for n,h in proof['proof_sha256'].items():assert sha(D/n/'SHA256.json')==h and read(D/n/'independent_audit.json')['passed']
  base=D/'lifted_feasibility_preparation_490'/proof['role'];same(A,sparse.load_npz(base/'matrix.npz'));same(E,sparse.load_npz(base/'equality.npz'));old=dict(np.load(base/'arrays.npz'));assert all(np.array_equal(a[k],old[k]) for k in a)
 else:
  assert np.all(np.load(T/'new_quotient_verified.npy')) and np.all(np.load(T/'new_box_verified.npy'))
  for name in ['new_definition_decisions_verified.npy','new_definition_rows_verified.npy','new_definition_duals_verified.npy']:assert np.all(np.load(T/name))
 src=Path(proof['source']);red=Path(proof['reduced']);SA=sparse.load_npz(src/'matrix.npz');sa=dict(np.load(src/'arrays.npz'));RA=sparse.load_npz(red/'matrix.npz');ra=dict(np.load(red/'arrays.npz'));ids=np.asarray(proof['source_row_ids']);keep=np.asarray(proof['reduced_source_positions'])
 same(SA,A[ids]);same(RA,SA[keep]);same(sparse.load_npz(red/'equality.npz'),E);assert np.array_equal(sa['rhs'],a['rhs'][ids]) and np.array_equal(ra['rhs'],sa['rhs'][keep])
 assert np.array_equal(np.load(red/'original_row_ids.npy'),ids[keep])
 for k in ['cost','bounds','equality_rhs']:assert np.array_equal(sa[k],a[k]) and np.array_equal(ra[k],a[k])
 ext=Path(proof['extended']);XA=sparse.load_npz(ext/'matrix.npz');XE=sparse.load_npz(ext/'equality.npz');xa=dict(np.load(ext/'arrays.npz'));pairs=np.load(ext/'pairs.npy');n=78791;ne=1366;k=len(pairs)
 assert XA.shape==(RA.shape[0],n+k) and XE.shape==(ne+k,n+k) and XE.nnz>0 and proof['original_scalar_column']==78790
 assert list(XA.shape)==proof['extended_shape'] and XA.nnz==proof['extended_nnz'] and XE.nnz==proof['extended_equality_nnz']
 assert np.array_equal(xa['rhs'],ra['rhs']) and np.array_equal(xa['cost'][:n],ra['cost']) and np.all(xa['cost'][n:]==0)
 assert np.array_equal(xa['bounds'][:n],ra['bounds']) and np.isneginf(xa['bounds'][n:,0]).all() and np.isposinf(xa['bounds'][n:,1]).all()
 assert np.array_equal(xa['equality_rhs'][:ne],a['equality_rhs']) and np.all(xa['equality_rhs'][ne:]==0)
 same(XE[:ne,:n],E);assert XE[:ne,n:].nnz==0
 assert np.array_equal(np.load(ext/'original_row_ids.npy'),ids[keep]) and np.array_equal(np.load(ext/'source516_positions.npy'),keep)
 # Reference558 exact signed-row/dual-column identities are inherited559 byhash above, never replayed.
 clock=read(T/'clock_origin.json');assert clock['all_new_preparation_charged'] and clock['parent_preparation_elapsed_s']>=0 and r['parent_preparation_charged_s']>=clock['parent_preparation_elapsed_s']
 out.update(input_preparation_completed=True,original_quotient_and_box_proof_inherited=proof['inherited'],definition_proof_inherited=proof['inherited'],old_proofs_repeated=False,full_rows=A.shape[0],source_rows=SA.shape[0],retained_rows=RA.shape[0],retained_nonzeros=RA.nnz,extended_shape=list(XA.shape),extended_inequality_nonzeros=XA.nnz,extended_equality_shape=list(XE.shape),extended_equality_nonzeros=XE.nnz,original_scalar_column=78790,new_free_definition_variables=k,all_new_preparation_charged=True)
 if (T/'original_checks.json').exists():
  w=np.load(T/'returned_native.npy');x=np.load(T/'returned_extended.npy');assert w.shape==(n,) and x.shape==(n+k,) and np.array_equal(w.view(np.uint64),x[:n].view(np.uint64))
  sd=dict(np.load(T/'original_duals.npz'));sy=np.zeros(SA.shape[0]);sy[keep]=sd['retained'];fy=np.zeros(A.shape[0]);fy[ids]=sy
  assert np.array_equal(sy,sd['source']) and np.array_equal(fy,sd['full'])
  for key,cut in [('upper',n),('lower',n),('equality',ne)]:assert np.array_equal(sd[key].view(np.uint64),sd['extended_'+key][:cut].view(np.uint64))
  res=np.load(T/'original_residuals.npz');saved=read(T/'original_checks.json');checks={}
  for name,M,z,EQ,ww,dual,upper,lower,ey in [('extended',XA,xa,XE,x,sd['retained'],sd['extended_upper'],sd['extended_lower'],sd['extended_equality']),('retained',RA,ra,E,w,sd['retained'],sd['upper'],sd['lower'],sd['equality']),('source',SA,sa,E,w,sy,sd['upper'],sd['lower'],sd['equality']),('full',A,a,E,w,fy,sd['upper'],sd['lower'],sd['equality'])]:
   values=M@ww-z['rhs'];st=z['cost']+M.T@dual+EQ.T@ey+upper-lower;low=np.isfinite(z['bounds'][:,0]);high=np.isfinite(z['bounds'][:,1]);pr=max(0.,float(values.max()),float(abs(EQ@ww-z['equality_rhs']).max()),float((z['bounds'][:,0]-ww).max()),float((ww-z['bounds'][:,1]).max()));du=float(abs(st).max());gap=abs(float(z['cost']@ww+z['rhs']@dual+z['equality_rhs']@ey+z['bounds'][high,1]@upper[high]-z['bounds'][low,0]@lower[low]))
   finite=all(np.all(np.isfinite(v)) for v in [ww,dual,upper,lower,ey,values,st,pr,du,gap]);sign=bool(np.all(dual>=0) and np.all(upper>=0) and np.all(lower>=0) and np.all(upper[~high]==0) and np.all(lower[~low]==0));passed=bool(finite and sign and max(pr,du,gap)<=1e-9)
   assert np.array_equal(values,res[name]) and np.array_equal(st,res[name+'_stationarity']);checks[name]=dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=sign,passed=passed);assert checks[name]==saved[name]
  assert saved['passed']==all(c['passed'] for c in checks.values());assert saved['definition_columns_discarded']==k and saved['original_scalar_column']==78790
  out['original_KKT']=checks;out['all_coordinates_bitwise_unchanged']=True
 if r['candidate']:
  assert r['status']==0 and r['native_status']==0 and r['elapsed_s']<=45 and read(T/'original_checks.json')['passed']
  assert np.array_equal(np.load(T/'witness.npy').view(np.uint64),np.load(T/'returned_native.npy').view(np.uint64))
 else:assert not (T/'witness.npy').exists()
 return out
