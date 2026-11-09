"""Independent574 numerical return/input/resource audit. No production solver/producer import."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return sparse.load_npz(p/'matrix.npz').tocsr(),dict(np.load(p/'arrays.npz')),sparse.load_npz(p/'equality.npz').tocsr()
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and A.data.tobytes()==B.data.tobytes()
def audit_solver(T):
 r=read(T/'result.json');actual=(T/'scipy_started.json').exists();assert r['actual_lp_call']==actual and r['full_size_robot_LP']==actual and not r['small_test'] and r['input_loading_inside_clock']
 assert r['method']=='primary-zero-face-native-IPM' and r['configured_address_space_bytes']==40*1024**3 and not r['peak_memory_measured'];Q=T.parent
 A,a,E=load(Q);primary=read(Q/'stage.json')['primary'];assert r['primary']==primary
 out=dict(passed=True,actual_lp_call=actual,numerical_candidate=bool(r['candidate']),physical_admission=False,backend_internal_iterates_available=False,backend_iterations_replayed=False,coordinate_reconstruction_attempts=0)
 clock=read(T/'clock_origin.json');assert clock['all_new_preparation_charged'] and clock['parent_preparation_elapsed_s']>=0 and r['parent_preparation_charged_s']>=clock['parent_preparation_elapsed_s']
 if actual:
  cfg=read(T/'scipy_started.json');opt=cfg['options'];assert cfg['cold_start'] and not cfg['old_internal_state_used'] and cfg['method']=='highs-ipm' and cfg['primary']==primary
  assert 0<opt['time_limit']<=45 and opt['maxiter']==4096 and opt['presolve'] and all(opt[k]==1e-9 for k in ['primal_feasibility_tolerance','dual_feasibility_tolerance','ipm_optimality_tolerance'])
  if (T/'backend_result.json').exists():
   b=read(T/'backend_result.json');assert b['status']==r['native_status'] and b['iterations']==r['iterations'] and b['crossover_iterations']==r['crossover_iterations'] and b['candidate_returned']==r['native_candidate_returned'] and b['elapsed_s']==r['backend_elapsed_s'] and b['configured_time_limit_s']==opt['time_limit'];out['backend_result']=b
 if not (T/'input_proof.json').exists():
  assert not actual and not r['candidate'] and not (T/'witness.npy').exists();out['input_preparation_completed']=False;return out
 proof=read(T/'input_proof.json');assert proof['passed'];frozen=read(D/'zero_face_path_launch_574.json')['source_input_sha256']
 for p,h in {**proof['input_sha256'],**proof['full_input_sha256']}.items():assert sha(p)==h
 systems={};paths=proof['system_paths']
 for name,path in paths.items():
  if name=='source':systems[name]=(sparse.load_npz(Path(path)/'matrix.npz').tocsr(),dict(np.load(Path(path)/'arrays.npz')),E)
  else:systems[name]=load(Path(path))
 if proof['inherited']:
  assert primary and Q==D/'zero_face_path_574/major_00/primary'
  for p,h in proof['input_sha256'].items():assert frozen[p]==h
  for n,h in proof['proof_sha256'].items():assert frozen[str(D/n/'SHA256.json')]==h==sha(D/n/'SHA256.json') and read(D/n/'independent_audit.json')['passed']
  old=D/'lifted_feasibility_preparation_490/primary';oa,oz,oe=load(old);same(A,oa);same(E,oe);assert all(a[k].tobytes()==oz[k].tobytes() for k in a)
  ids=np.load(Path(paths['source']).parent/'row_ids.npy');keep=np.load(Path(paths['retained'])/'retained_source_positions.npy');ck=np.load(Path(paths['coupled'])/'retained_546_positions.npy')
  SA,sa,SE=systems['source'];RA,ra,RE=systems['retained'];CA,ca,CE=systems['coupled'];same(SA,A[ids]);same(RA,SA[keep]);same(CA,RA[ck]);same(RE,E);same(CE,E)
  for z,rr in [(sa,ids),(ra,ids[keep]),(ca,ids[keep][ck])]:
   for key in a:assert np.array_equal(z[key],a[key][rr] if key=='rhs' else a[key])
  # The prepared572 AFFINE equality copies/scalar/RHS identities and analytic
  # lower-bound proof are inherited573, not repeated on this same input.
  assert set(systems)=={'face','coupled','retained','source','full'}
 else:
  assert proof['new_full_input']['passed'] and proof['all_original_rows_retained_without_reduction']
  if primary:assert proof['new_face_input']['passed'] and np.all(np.load(T/'new_face_pairs_verified.npy')==1)
  assert set(systems)==({'face','full'} if primary else {'full'})
 B,b,H=systems[proof['solver_system']];assert list(B.shape)==proof['solver_inequality_shape'] and B.nnz==proof['solver_inequality_nnz'] and list(H.shape)==proof['solver_equality_shape'] and H.nnz==proof['solver_equality_nnz']
 assert proof['original_columns']==78791 and proof['original_scalar_column']==78790
 out.update(input_preparation_completed=True,initial_proofs_inherited=proof['inherited'],old_proofs_repeated=False,solver_inequality_shape=list(B.shape),solver_inequality_nonzeros=B.nnz,solver_equality_shape=list(H.shape),solver_equality_nonzeros=H.nnz,all_new_preparation_charged=True,original_coordinates_retained=78791,primary_zero_face_restriction=primary,whole_original_equivalence=False if primary else True)
 if (T/'original_checks.json').exists():
  w=np.load(T/'returned_native.npy');raw=np.load(T/'backend_primal.npy');assert w.shape==raw.shape==(78791,) and w.dtype==raw.dtype==np.dtype('float64') and raw.tobytes()==w.tobytes() and np.isfinite(w).all()
  sd=dict(np.load(T/'original_duals.npz'));res=np.load(T/'original_residuals.npz');saved=read(T/'original_checks.json');checks={}
  if primary:
   assert w[78790]==0;analytic=dict(np.load(D/'zero_face_preparation_572/analytic_lower_bound_dual.npz'))
  else:bm=dict(np.load(T/'backend_marginals.npz'))
  for name,(M,z,EQ) in systems.items():
   y,upper,lower,ed=[sd[name+'_'+tag] for tag in ['row','upper','lower','equality']]
   assert y.shape==(M.shape[0],) and upper.shape==lower.shape==(78791,) and ed.shape==(EQ.shape[0],)
   if primary:assert np.all(y==0) and np.all(ed==0) and upper.tobytes()==analytic['upper'].tobytes() and lower.tobytes()==analytic['lower'].tobytes()
   else:
    for val,key,sgn in [(y,'inequality',-1),(ed,'equality',-1),(upper,'upper',-1),(lower,'lower',1)]:assert val.tobytes()==(sgn*bm[key]).tobytes()
   values=M@w-z['rhs'];eq=EQ@w-z['equality_rhs'];st=z['cost']+M.T@y+EQ.T@ed+upper-lower;lo=np.isfinite(z['bounds'][:,0]);hi=np.isfinite(z['bounds'][:,1])
   pr=max(0.,float(values.max(initial=0)),float(abs(eq).max(initial=0)),float((z['bounds'][:,0]-w).max()),float((w-z['bounds'][:,1]).max()));du=float(abs(st).max());gap=abs(float(z['cost']@w+z['rhs']@y+z['equality_rhs']@ed+z['bounds'][hi,1]@upper[hi]-z['bounds'][lo,0]@lower[lo]))
   finite=all(np.all(np.isfinite(v)) for v in [w,y,upper,lower,ed,values,eq,st,pr,du,gap]);sign=bool(np.all(y>=0) and np.all(upper>=0) and np.all(lower>=0) and np.all(upper[~hi]==0) and np.all(lower[~lo]==0));passed=bool(finite and sign and max(pr,du,gap)<=1e-9)
   assert np.array_equal(values,res[name]) and np.array_equal(eq,res[name+'_equality']) and np.array_equal(st,res[name+'_stationarity']);checks[name]=dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=sign,passed=passed);assert checks[name]==saved[name]
  assert saved['passed']==all(c['passed'] for c in checks.values()) and saved['analytic_dual_used']==primary;out['original_KKT']=checks;out['all_original_coordinates_bitwise_backend_primal']=True
 if r['candidate']:
  assert r['status']==0 and r['native_status']==0 and r['elapsed_s']<=45 and read(T/'original_checks.json')['passed'];assert np.load(T/'witness.npy').tobytes()==np.load(T/'returned_native.npy').tobytes()
 else:assert not (T/'witness.npy').exists()
 return out
