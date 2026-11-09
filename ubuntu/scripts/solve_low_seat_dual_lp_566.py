"""ONE cold IPM on exact564 native dual balance; originalw from equality marginals and all FOUR KKT gates."""
from pathlib import Path
import sys,time,resource,json,hashlib,shutil
CHILD_START=time.monotonic();START=float(sys.argv[2]) if len(sys.argv)>2 else CHILD_START;LIMIT=40*1024**3;resource.setrlimit(resource.RLIMIT_AS,(LIMIT,LIMIT))
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from low_seat_dual_input_566 import quotient,derive,derive_dual
from certify_low_seat_box_input_549 import prove
from certify_low_seat_dual_input_567 import prove_dual
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
 low=np.isfinite(a['bounds'][:,0]);high=np.isfinite(a['bounds'][:,1]);pr=max(0.,float(vals.max(initial=0)),float(abs(E@w-a['equality_rhs']).max(initial=0)),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 du=float(abs(station).max());gap=abs(float(a['cost']@w+a['rhs']@y+a['equality_rhs']@eqdual+a['bounds'][high,1]@upper[high]-a['bounds'][low,0]@lower[low]))
 finite=all(np.all(np.isfinite(v)) for v in [w,y,upper,lower,eqdual,vals,station,pr,du,gap]);signs=bool(np.all(y>=0) and np.all(upper>=0) and np.all(lower>=0) and np.all(upper[~high]==0) and np.all(lower[~low]==0))
 return dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=signs,passed=bool(finite and signs and max(pr,du,gap)<=1e-9)),vals,station
def main():
 out=Path(sys.argv[1]);full=out.parent;actual=False;nit=0;cross=0;native_status=None;backend_elapsed=0.;native_candidate=False;eqmarginal_candidate=False;configured_remaining=None
 try:
  check();A,a,E=load(full);check();role='primary' if read(full/'stage.json')['primary'] else 'witness'
  assert A.shape[1]==78791 and E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
  assert np.all(a['cost'][:-1]==0) and a['cost'][-1] in (-1,1) and np.all(np.isneginf(a['bounds'][:-1,0])) and np.all(np.isposinf(a['bounds'][:-1,1]))
  old=D/'lifted_feasibility_preparation_490'/role;oa,oz,oe=load(old);inherited=same(A,oa) and same(E,oe) and all(np.array_equal(a[k],oz[k]) for k in a);del oa,oz,oe;check()
  write(out/'input_plan.json',dict(inherited=inherited,role=role,full_shape=list(A.shape),full_nnz=A.nnz,all_native_directions=78790,scalar_retained=True,no_native_substitution=True,equality_marginal_return_required=True,optimizer_primal_is_multiplier_vector=True))
  if inherited:
   src=D/'exact_row_quotient_preparation_516'/role/'original';ids=np.load(src.parent/'row_ids.npy');SA=sparse.load_npz(src/'matrix.npz').tocsr();sa=dict(np.load(src/'arrays.npz'));red=D/'box_support_preparation_546'/role
   RA,ra,RE=load(red);keep=np.load(red/'retained_source_positions.npy');rid=ids[keep]
   assert same(SA,A[ids]) and same(RA,SA[keep]) and same(RE,E)
   assert np.array_equal(np.load(red/'original_row_ids.npy'),rid)
   for k in a:assert np.array_equal(sa[k],a[k][ids] if k=='rhs' else a[k]) and np.array_equal(ra[k],sa[k][keep] if k=='rhs' else sa[k])
   new=D/'dual_balance_preparation_564'/role;BE=sparse.load_npz(new/'equality.npz').tocsr();da=dict(np.load(new/'arrays.npz'));layout=read(new/'layout.json');m=RA.shape[0]
   assert BE.shape==(78791,m+2) and BE.nnz==RA.nnz+2 and da['cost'].shape==(m+2,) and da['bounds'].shape==(m+2,2)
   assert np.array_equal(da['cost'][:m],ra['rhs']) and da['cost'][m]==ra['bounds'][78790,1] and da['cost'][m+1]==-ra['bounds'][78790,0]
   assert np.array_equal(da['equality_rhs'],-ra['cost']) and np.all(da['bounds'][:,0]==0) and np.isposinf(da['bounds'][:,1]).all()
   assert np.array_equal(np.load(new/'original_row_ids.npy'),rid) and np.array_equal(np.load(new/'source516_positions.npy'),keep)
   assert np.array_equal(np.load(new/'original_coordinate_slots.npy'),np.arange(78791)) and layout['original_scalar_slot']==78790
   # Exact transpose/coefficient identities inherited565 byhash, not recomputed.
   inputs=[src/'matrix.npz',src/'arrays.npz',src.parent/'row_ids.npy',old/'equality.npz']+list(red.iterdir())+list(new.iterdir())
   frozen=read(D/'dual_path_launch_566.json')['source_input_sha256']
   for p in inputs:assert sha(p)==frozen[str(p)],str(p)
   proof=dict(passed=True,inherited=True,new_quotient_proof_rows=0,new_box_proof_rows=0,old_box_proof_repeated=False,old_dual_proof_repeated=False,new_dual_proof_coefficients=0,proof_sha256={n:sha(D/n/'SHA256.json') for n in ['exact_row_quotient_audit_517','box_support_audit_547','dual_balance_audit_565']})
   for n in proof['proof_sha256']:assert read(D/n/'independent_audit.json')['passed']
  else:
   src=out/'source';src.mkdir();led=quotient(A,a['rhs'],check);ids=led['representatives'];SA=A[ids].tocsr();sa=dict(a,rhs=a['rhs'][ids]);check()
   sparse.save_npz(src/'matrix.npz',SA);sparse.save_npz(src/'equality.npz',E);np.savez(src/'arrays.npz',**sa);np.save(src/'row_ids.npy',ids);np.savez(src/'ledger.npz',**led)
   red=out/'reduced';red.mkdir();stats=derive(SA,sa,ids,red,check);shutil.copy2(src/'equality.npz',red/'equality.npz');write(red/'scope.json',stats);check()
   proof=prove(A,a,SA,sa,ids,led,src,red,stats,out,check);proof.update(inherited=False)
   RA,ra,RE=load(red);keep=np.load(red/'retained_source_positions.npy');rid=ids[keep]
   new=out/'dual';new.mkdir();ds=derive_dual(red,new,check);check()
   dp=prove_dual(red,new,out,check);proof['new_dual_proof']=dp;proof['new_dual_proof_coefficients']=dp['original_stationarity_coefficient_identities'];proof['old_dual_proof_repeated']=False;check()
   BE=sparse.load_npz(new/'equality.npz').tocsr();da=dict(np.load(new/'arrays.npz'));m=RA.shape[0]
   inputs=list(new.iterdir())+[out/'new_dual_coordinate_slots_verified.npy',out/'new_dual_multiplier_slots_verified.npy']+list(src.iterdir())+list(red.iterdir())+[out/'new_quotient_verified.npy',out/'new_box_verified.npy']
  DA=sparse.csr_matrix((0,BE.shape[1]));da['rhs']=np.empty(0)
  proof.update(dual=str(new),dual_equality_shape=list(BE.shape),dual_equality_nnz=BE.nnz,dual_inequality_rows=0,optimizer_primal_variables=BE.shape[1],original_coordinate_marginal_slots=78791,original_scalar_slot=78790,source=str(src),reduced=str(red),role=role,source_row_ids=ids.tolist(),reduced_source_positions=keep.tolist(),input_sha256={str(p):sha(p) for p in inputs},full_input_sha256={str(full/n):sha(full/n) for n in ['matrix.npz','arrays.npz','equality.npz']},full_rows=A.shape[0],source_rows=SA.shape[0],reduced_rows=RA.shape[0],source_columns=SA.shape[1],full_nnz=A.nnz,source_nnz=SA.nnz,reduced_nnz=RA.nnz)
  write(out/'input_proof.json',proof);check()
  configured_remaining=45-(time.monotonic()-START)
  options=dict(time_limit=configured_remaining,maxiter=4096,presolve=True,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
  write(out/'scipy_started.json',dict(actual_lp_call=True,method='highs-ipm',construction='exact-native-dual-balance-IPM',options=options,cold_start=True,old_internal_state_used=False,input_loading_inside_clock=True))
  actual=True;bs=time.monotonic();r=linprog(da['cost'],A_eq=BE,b_eq=da['equality_rhs'],bounds=da['bounds'],method='highs-ipm',options=options);backend_elapsed=time.monotonic()-bs
  nit=int(r.nit);cross=int(getattr(r,'crossover_nit',0));native_status=int(r.status);native_candidate=r.x is not None;eqmarginal_candidate=r.eqlin.marginals is not None
  write(out/'backend_result.json',dict(status=native_status,message=str(r.message),success=bool(r.success),candidate_returned=native_candidate,optimizer_primal_is_multiplier=True,equality_marginals_returned=eqmarginal_candidate,iterations=nit,crossover_iterations=cross,elapsed_s=backend_elapsed,configured_time_limit_s=configured_remaining,internal_vectors_saved=False))
  check()
  if native_status!=0 or not r.success or r.x is None or r.eqlin.marginals is None:raise Terminal('dual IPM returned no status0 multiplier/equality-marginal pair: '+str(r.message))
  p=np.asarray(r.x);eqm=np.asarray(r.eqlin.marginals);lm=np.asarray(r.lower.marginals);um=np.asarray(r.upper.marginals)
  assert p.shape==(m+2,) and eqm.shape==(78791,) and lm.shape==um.shape==p.shape
  assert all(v.dtype==np.dtype('float64') and np.isfinite(v).all() for v in [p,eqm,lm,um])
  np.savez_compressed(out/'backend_pair.npz',primal_multipliers=p,equality_marginals=eqm,lower_marginals=lm,upper_marginals=um);check()
  w=eqm.copy();assert w.tobytes()==eqm.tobytes();np.save(out/'returned_native.npy',w)
  ry=p[:m].copy();upper=np.zeros(78791);lower=np.zeros(78791);upper[78790]=p[m];lower[78790]=p[m+1];ey=np.zeros(1366)
  dy=np.empty(0);de=-eqm;du=-um;dl=lm.copy()
  sy=np.zeros(SA.shape[0]);sy[keep]=ry;fy=np.zeros(A.shape[0]);fy[ids]=sy;checks={};res={}
  for name,M,z,EQ,ww,dual,up,lo,ed in [('dual',DA,da,BE,p,dy,du,dl,de),('retained',RA,ra,RE,w,ry,upper,lower,ey),('source',SA,sa,E,w,sy,upper,lower,ey),('full',A,a,E,w,fy,upper,lower,ey)]:
   checks[name],res[name],res[name+'_stationarity']=kkt(M,z,EQ,ww,dual,up,lo,ed);check()
  np.savez_compressed(out/'original_residuals.npz',**res);np.savez_compressed(out/'original_duals.npz',retained=ry,source=sy,full=fy,upper=upper,lower=lower,equality=ey,dual_upper=du,dual_lower=dl,dual_equality=de)
  write(out/'original_checks.json',dict(passed=all(c['passed'] for c in checks.values()),**checks,native_values_unchanged=True,scalar_values_unchanged=True,original_coordinate_marginal_slots=78791,original_scalar_slot=78790,original_primal_source='BITWISE scipy.eqlin.marginals; no negation; NOT result.x',original_dual_source='result.x row multiplier prefix and scalar upper then lower; native bound/oldE duals zero; zero implied516/490 rows',coordinate_reconstruction=False));check()
  if not all(c['passed'] for c in checks.values()):raise Terminal('dual/546/516/FULL490 numerical KKT failed; no repair')
  assert w.tobytes()==np.asarray(r.eqlin.marginals).tobytes();np.save(out/'witness.npy',w);check()
  result=dict(status=0,message='unchanged original78791 equality marginals passed all FOUR numerical KKT gates',candidate=True,success=True)
 except (Terminal,AssertionError,MemoryError,ValueError,OverflowError,FloatingPointError,OSError) as exc:
  # A vector written just before the deadline is not eligible for use.
  if (out/'witness.npy').exists(): (out/'witness.npy').rename(out/'rejected_after_deadline.npy')
  result=dict(status=1,message=str(exc),candidate=False,success=False)
 result.update(iterations=nit,crossover_iterations=cross,native_status=native_status,native_candidate_returned=native_candidate,backend_candidate_is_multiplier_vector=True,equality_marginals_returned=eqmarginal_candidate,elapsed_s=time.monotonic()-START,child_startup_elapsed_s=time.monotonic()-CHILD_START,parent_preparation_charged_s=CHILD_START-START,backend_elapsed_s=backend_elapsed,configured_backend_seconds=configured_remaining,actual_lp_call=actual,full_size_robot_LP=actual,small_test=False,method='exact-native-dual-balance-IPM',configured_address_space_bytes=LIMIT,peak_memory_measured=False,input_loading_inside_clock=True,coordinate_reconstruction_attempts=0,physical_admission=False)
 write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
