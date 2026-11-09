"""ONE cold IPM invocation on the exact546 native reduction; unchanged coordinates and full KKT gates."""
from pathlib import Path
import sys,time,resource,json,hashlib,shutil
START=time.monotonic();LIMIT=40*1024**3;resource.setrlimit(resource.RLIMIT_AS,(LIMIT,LIMIT))
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from low_seat_box_input_548 import quotient,derive
from certify_low_seat_box_input_549 import prove
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Terminal(Exception):pass
def check():
 if time.monotonic()-START>45:raise Terminal('45s total solver wall deadline')
def same(A,B):
 A=A.tocsr();B=B.tocsr();return A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def load(p):return sparse.load_npz(p/'matrix.npz').tocsr(),dict(np.load(p/'arrays.npz')),sparse.load_npz(p/'equality.npz').tocsr()
def kkt(A,a,E,w,y,upper,lower,eqdual):
 vals=A@w-a['rhs'];station=a['cost']+A.T@y+E.T@eqdual+upper-lower
 low=np.isfinite(a['bounds'][:,0]);high=np.isfinite(a['bounds'][:,1]);pr=max(0.,float(vals.max()),float(abs(E@w-a['equality_rhs']).max()),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 du=float(abs(station).max());gap=abs(float(a['cost']@w+a['rhs']@y+a['equality_rhs']@eqdual+a['bounds'][high,1]@upper[high]-a['bounds'][low,0]@lower[low]))
 finite=all(np.all(np.isfinite(v)) for v in [w,y,upper,lower,eqdual,vals,station,pr,du,gap]);signs=bool(np.all(y>=0) and np.all(upper>=0) and np.all(lower>=0) and np.all(upper[~high]==0) and np.all(lower[~low]==0))
 return dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=signs,passed=bool(finite and signs and max(pr,du,gap)<=1e-9)),vals,station
def main():
 out=Path(sys.argv[1]);full=out.parent;actual=False;nit=0;cross=0;native_status=None;backend_elapsed=0.;native_candidate=False;configured_remaining=None
 try:
  check();A,a,E=load(full);check();role='primary' if read(full/'stage.json')['primary'] else 'witness'
  assert A.shape[1]==78791 and E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
  assert np.all(a['cost'][:-1]==0) and a['cost'][-1] in (-1,1) and np.all(np.isneginf(a['bounds'][:-1,0])) and np.all(np.isposinf(a['bounds'][:-1,1]))
  old=D/'lifted_feasibility_preparation_490'/role;oa,oz,oe=load(old);inherited=same(A,oa) and same(E,oe) and all(np.array_equal(a[k],oz[k]) for k in a);del oa,oz,oe;check()
  write(out/'input_plan.json',dict(inherited=inherited,role=role,full_shape=list(A.shape),full_nnz=A.nnz,all_native_directions=78790,scalar_retained=True,no_projection=True))
  if inherited:
   src=D/'exact_row_quotient_preparation_516'/role/'original';ids=np.load(src.parent/'row_ids.npy');SA=sparse.load_npz(src/'matrix.npz').tocsr();sa=dict(np.load(src/'arrays.npz'));red=D/'box_support_preparation_546'/role
   RA,ra,RE=load(red);keep=np.load(red/'retained_source_positions.npy');rid=ids[keep]
   assert same(SA,A[ids]) and same(RA,SA[keep]) and same(RE,E)
   assert np.array_equal(np.load(red/'original_row_ids.npy'),rid)
   for k in a:assert np.array_equal(sa[k],a[k][ids] if k=='rhs' else a[k]) and np.array_equal(ra[k],sa[k][keep] if k=='rhs' else sa[k])
   inputs=[src/'matrix.npz',src/'arrays.npz',src.parent/'row_ids.npy',old/'equality.npz']+list(red.iterdir())
   frozen=read(D/'box_path_launch_548.json')['source_input_sha256']
   for p in inputs:assert sha(p)==frozen[str(p)],str(p)
   proof=dict(passed=True,inherited=True,new_quotient_proof_rows=0,new_box_proof_rows=0,old_box_proof_repeated=False,proof_sha256={n:sha(D/n/'SHA256.json') for n in ['exact_row_quotient_audit_517','box_support_audit_547']})
   for n in proof['proof_sha256']:assert read(D/n/'independent_audit.json')['passed']
  else:
   src=out/'source';src.mkdir();led=quotient(A,a['rhs'],check);ids=led['representatives'];SA=A[ids].tocsr();sa=dict(a,rhs=a['rhs'][ids]);check()
   sparse.save_npz(src/'matrix.npz',SA);sparse.save_npz(src/'equality.npz',E);np.savez(src/'arrays.npz',**sa);np.save(src/'row_ids.npy',ids);np.savez(src/'ledger.npz',**led)
   red=out/'reduced';red.mkdir();stats=derive(SA,sa,ids,red,check);shutil.copy2(src/'equality.npz',red/'equality.npz');write(red/'scope.json',stats);check()
   proof=prove(A,a,SA,sa,ids,led,src,red,stats,out,check);proof.update(inherited=False)
   RA,ra,RE=load(red);keep=np.load(red/'retained_source_positions.npy');rid=ids[keep]
   inputs=list(src.iterdir())+list(red.iterdir())+[out/'new_quotient_verified.npy',out/'new_box_verified.npy']
  proof.update(source=str(src),reduced=str(red),role=role,source_row_ids=ids.tolist(),reduced_source_positions=keep.tolist(),input_sha256={str(p):sha(p) for p in inputs},full_input_sha256={str(full/n):sha(full/n) for n in ['matrix.npz','arrays.npz','equality.npz']},full_rows=A.shape[0],source_rows=SA.shape[0],reduced_rows=RA.shape[0],source_columns=SA.shape[1],full_nnz=A.nnz,source_nnz=SA.nnz,reduced_nnz=RA.nnz)
  write(out/'input_proof.json',proof);check()
  configured_remaining=45-(time.monotonic()-START)
  options=dict(time_limit=configured_remaining,maxiter=4096,presolve=True,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
  write(out/'scipy_started.json',dict(actual_lp_call=True,method='highs-ipm',construction='exact-box-native-IPM',options=options,cold_start=True,old_internal_state_used=False,input_loading_inside_clock=True))
  actual=True;bs=time.monotonic();r=linprog(ra['cost'],A_ub=RA,b_ub=ra['rhs'],A_eq=RE,b_eq=ra['equality_rhs'],bounds=ra['bounds'],method='highs-ipm',options=options);backend_elapsed=time.monotonic()-bs
  nit=int(r.nit);cross=int(getattr(r,'crossover_nit',0));native_status=int(r.status);native_candidate=r.x is not None
  write(out/'backend_result.json',dict(status=native_status,message=str(r.message),success=bool(r.success),candidate_returned=native_candidate,iterations=nit,crossover_iterations=cross,elapsed_s=backend_elapsed,configured_time_limit_s=configured_remaining,internal_vectors_saved=False))
  check()
  if native_status!=0 or not r.success or r.x is None:raise Terminal('native IPM returned no status0 candidate: '+str(r.message))
  w=np.asarray(r.x);ry=-np.asarray(r.ineqlin.marginals);ey=-np.asarray(r.eqlin.marginals);upper=-np.asarray(r.upper.marginals);lower=np.asarray(r.lower.marginals)
  np.save(out/'returned_native.npy',w);check()
  assert w.shape==(78791,) and ry.shape==(RA.shape[0],) and ey.shape==(1366,) and upper.shape==lower.shape==w.shape
  sy=np.zeros(SA.shape[0]);sy[keep]=ry;fy=np.zeros(A.shape[0]);fy[ids]=sy;checks={};res={}
  for name,M,z,dual in [('retained',RA,ra,ry),('source',SA,sa,sy),('full',A,a,fy)]:
   checks[name],res[name],res[name+'_stationarity']=kkt(M,z,E,w,dual,upper,lower,ey);check()
  np.savez_compressed(out/'original_residuals.npz',**res);np.savez_compressed(out/'original_duals.npz',retained=ry,source=sy,full=fy,upper=upper,lower=lower,equality=ey)
  write(out/'original_checks.json',dict(passed=all(c['passed'] for c in checks.values()),**checks,native_values_unchanged=True,scalar_values_unchanged=True,dual_mapping='zero on box-implied and original ray-implied rows',coordinate_reconstruction=False));check()
  if not all(c['passed'] for c in checks.values()):raise Terminal('full retained/source/original numerical KKT failed; no coordinate repair')
  # No scalar/torque reconstruction, clipping, perturbation or post-solve coordinate edit.
  assert np.array_equal(w.view(np.uint64),np.asarray(r.x).view(np.uint64));np.save(out/'witness.npy',w);check()
  result=dict(status=0,message='unchanged native+scalar vector passed all three full numerical KKT gates',candidate=True,success=True)
 except (Terminal,AssertionError,MemoryError,ValueError,OverflowError,FloatingPointError,OSError) as exc:
  # A vector written just before the deadline is not eligible for use.
  if (out/'witness.npy').exists(): (out/'witness.npy').rename(out/'rejected_after_deadline.npy')
  result=dict(status=1,message=str(exc),candidate=False,success=False)
 result.update(iterations=nit,crossover_iterations=cross,native_status=native_status,native_candidate_returned=native_candidate,elapsed_s=time.monotonic()-START,backend_elapsed_s=backend_elapsed,configured_backend_seconds=configured_remaining,actual_lp_call=actual,full_size_robot_LP=actual,small_test=False,method='exact-box-native-IPM',configured_address_space_bytes=LIMIT,peak_memory_measured=False,input_loading_inside_clock=True,coordinate_reconstruction_attempts=0,physical_admission=False)
 write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
