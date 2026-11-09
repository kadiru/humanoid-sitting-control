"""Independent554 input identity, full native primal/dual KKT and terminal resource audit; no solver import/call."""
from pathlib import Path
import json,hashlib
from fractions import Fraction as F
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def audit_solver(T):
 r=read(T/'result.json');actual=(T/'scipy_started.json').exists();assert r['actual_lp_call']==actual and r['full_size_robot_LP']==actual and not r['small_test'] and r['input_loading_inside_clock']
 assert r['method']=='finite-bound-native-IPM' and r['configured_address_space_bytes']==40*1024**3 and not r['peak_memory_measured'];Q=T.parent
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
 proof=read(T/'input_proof.json');assert proof['passed'];frozen=read(D/'float_path_launch_554.json')['source_input_sha256']
 for p,h in {**proof['input_sha256'],**proof['full_input_sha256']}.items():assert sha(p)==h
 if proof['inherited']:
  for p,h in proof['input_sha256'].items():assert h==frozen[p]
  for n,h in proof['proof_sha256'].items():assert sha(D/n/'SHA256.json')==h and read(D/n/'independent_audit.json')['passed']
  base=D/'lifted_feasibility_preparation_490'/proof['role'];same(A,sparse.load_npz(base/'matrix.npz'));same(E,sparse.load_npz(base/'equality.npz'));old=dict(np.load(base/'arrays.npz'));assert all(np.array_equal(a[k],old[k]) for k in a)
 else:
  assert all(np.all(np.load(T/n)) for n in ['new_quotient_verified.npy','new_box_verified.npy','new_float_endpoints_verified.npy','new_float_unaries_verified.npy'])
 src=Path(proof['source']);box=Path(proof['box_reduced']);red=Path(proof['reduced']);SA=sparse.load_npz(src/'matrix.npz').tocsr();sa=dict(np.load(src/'arrays.npz'));BA=sparse.load_npz(box/'matrix.npz').tocsr();ba=dict(np.load(box/'arrays.npz'));RA=sparse.load_npz(red/'matrix.npz').tocsr();ra=dict(np.load(red/'arrays.npz'));ids=np.asarray(proof['source_row_ids']);keep=np.asarray(proof['reduced_source_positions']);bk=np.asarray(proof['box_source_positions']);fk=np.asarray(proof['reduced_box_positions'])
 same(SA,A[ids]);same(BA,SA[bk]);same(RA,BA[fk]);same(sparse.load_npz(red/'equality.npz'),E);same(sparse.load_npz(box/'equality.npz'),E)
 assert np.array_equal(keep,bk[fk]) and np.array_equal(sa['rhs'],a['rhs'][ids]) and np.array_equal(ba['rhs'],sa['rhs'][bk]) and np.array_equal(ra['rhs'],ba['rhs'][fk])
 assert np.array_equal(np.load(red/'original490_row_ids.npy'),ids[keep]) and np.array_equal(np.load(red/'original_bounds.npy'),a['bounds']) and np.all(np.isfinite(ra['bounds'])) and np.all(ra['bounds'][:,0]<=ra['bounds'][:,1])
 for k in ['cost','equality_rhs']:assert np.array_equal(sa[k],a[k]) and np.array_equal(ba[k],a[k]) and np.array_equal(ra[k],a[k])
 assert np.array_equal(sa['bounds'],a['bounds']) and np.array_equal(ba['bounds'],a['bounds']) and not proof['real_equivalence_claimed'] and proof['finite_binary64_equivalence']
 out.update(input_preparation_completed=True,original_quotient_box_float_proof_inherited=proof['inherited'],old_proofs_repeated=False,full_rows=A.shape[0],source_rows=SA.shape[0],box_rows=BA.shape[0],retained_rows=RA.shape[0],retained_nonzeros=RA.nnz,real_equivalence_claimed=False)
 if (T/'original_checks.json').exists():
  w=np.load(T/'returned_native.npy');sd=dict(np.load(T/'original_duals.npz'));sy=np.zeros(SA.shape[0]);sy[keep]=sd['retained'];lower=np.zeros(SA.shape[1]);upper=lower.copy();owners=np.load(red/'bound_owners516.npy');ep=read(box/'endpoints.json');delta=F(0);sterr=F(0);rhsround=F(0);rows=[]
  # Independent exact coefficient equations for the saved rounded duals. No production lift import.
  for j in range(SA.shape[1]):
   for side in [0,1]:
    mu=F(float(sd['new_upper' if side else 'new_lower'][j]));assert mu>=0;owner=int(owners[j,side]);q=F(ep[j][side]);new=F(float(ra['bounds'][j,side]));shift=q-new if side else new-q;assert shift>=0;delta+=mu*shift
    if owner==-1:
     assert q==F(float(sa['bounds'][j,side]));(upper if side else lower)[j]=float(mu)
    elif mu:
     p=SA.indptr[owner];assert SA.indptr[owner+1]-p==1 and SA.indices[p]==j;coeff=F(float(SA.data[p]));rhs=F(float(sa['rhs'][owner]));assert (coeff>0)==bool(side) and coeff*q==rhs and sy[owner]==0
     saved=F(float(sd['source'][owner]));assert saved>=0;exact=mu/abs(coeff);assert float(exact)==float(saved);err=saved-exact;sy[owner]=float(saved);sterr=max(sterr,abs(coeff*err));rhsround+=rhs*err
     # Exact unrounded signed stationarity coefficient is +/-mu.
     assert coeff*exact==(mu if side else -mu);rows.append([j,side,owner,str(exact),str(err),str(coeff*err),str(rhs*err)])
    else:assert owner>=0
  ledger=[json.loads(l) for l in (T/'dual_lift.jsonl').read_text().splitlines()];assert ledger==rows;lift=read(T/'dual_lift.json');assert lift['nonzero_row_transfers']==len(rows) and F(lift['exact_endpoint_gap_penalty'])==delta and F(lift['maximum_exact_stationarity_rounding_error'])==sterr and F(lift['exact_rhs_rounding_contribution'])==rhsround
  fy=np.zeros(A.shape[0]);fy[ids]=sy;by=sy[bk];assert np.array_equal(sy,sd['source']) and np.array_equal(by,sd['box']) and np.array_equal(fy,sd['full']) and np.array_equal(upper,sd['upper']) and np.array_equal(lower,sd['lower']);ey=sd['equality'];res=np.load(T/'original_residuals.npz');saved=read(T/'original_checks.json');checks={}
  for name,M,z,dual,up,lo in [('retained',RA,ra,sd['retained'],sd['new_upper'],sd['new_lower']),('box',BA,ba,by,upper,lower),('source',SA,sa,sy,upper,lower),('full',A,a,fy,upper,lower)]:
   values=M@w-z['rhs'];st=z['cost']+M.T@dual+E.T@ey+up-lo;low=np.isfinite(z['bounds'][:,0]);high=np.isfinite(z['bounds'][:,1]);pr=max(0.,float(values.max()),float(abs(E@w-z['equality_rhs']).max()),float((z['bounds'][:,0]-w).max()),float((w-z['bounds'][:,1]).max()));du=float(abs(st).max());gap=abs(float(z['cost']@w+z['rhs']@dual+z['equality_rhs']@ey+z['bounds'][high,1]@up[high]-z['bounds'][low,0]@lo[low]))
   finite=all(np.all(np.isfinite(v)) for v in [w,dual,up,lo,ey,values,st,pr,du,gap]);sign=bool(np.all(dual>=0) and np.all(up>=0) and np.all(lo>=0) and np.all(up[~high]==0) and np.all(lo[~low]==0));passed=bool(finite and sign and max(pr,du,gap)<=1e-9)
   assert np.array_equal(values,res[name]) and np.array_equal(st,res[name+'_stationarity']);checks[name]=dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=sign,passed=passed);assert checks[name]==saved[name]
  assert saved['passed']==all(c['passed'] for c in checks.values());out['original_KKT']=checks;out['dual_lift']=lift
 if r['candidate']:
  assert r['status']==0 and r['native_status']==0 and r['elapsed_s']<=45 and read(T/'original_checks.json')['passed']
  assert np.array_equal(np.load(T/'witness.npy').view(np.uint64),np.load(T/'returned_native.npy').view(np.uint64));out['all_coordinates_bitwise_unchanged']=True
 else:assert not (T/'witness.npy').exists()
 return out
