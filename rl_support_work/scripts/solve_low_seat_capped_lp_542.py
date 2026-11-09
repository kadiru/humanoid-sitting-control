"""ONE full-size capped PDHG child. Timer includes all input loads and new-input preparation; no retries."""
from pathlib import Path
import sys,time,resource,json,hashlib,shutil
limit=40*1024**3
resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
import numpy as np
from scipy import sparse
from low_seat_capped_solver_542 import solve_lp,exact_metric,Terminal,sha
from low_seat_capped_max_538 import split
from low_seat_exact_row_quotient_v2_516 import quotient
from certify_low_seat_capped_input_543 import prove
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2))
def same(A,B):
 A=A.tocsr();B=B.tocsr();return A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def load(p):return sparse.load_npz(p/'matrix.npz').tocsr(),dict(np.load(p/'arrays.npz')),sparse.load_npz(p/'equality.npz').tocsr()
def kkt(A,a,E,w,y,tu,tl):
 vals=A@w-a['rhs'];st=a['cost']+A.T@y;st[-1]+=tu-tl
 lo,hi=a['bounds'][-1];pr=max(0.,float(vals.max()),float(abs(E@w-a['equality_rhs']).max()),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 du=float(abs(st).max());gap=abs(float(a['cost']@w+a['rhs']@y+hi*tu-lo*tl));finite=all(np.all(np.isfinite(v)) for v in [w,y,vals,st,tu,tl,pr,du,gap]);nonnegative=bool(np.all(y>=0) and tu>=0 and tl>=0)
 return dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=nonnegative,passed=bool(finite and nonnegative and max(pr,du,gap)<=1e-9)),vals,st

def main():
 out=Path(sys.argv[1]);start=time.monotonic();deadline=start+45;full=out.parent
 write(out/'pdhg_started.json',dict(actual_lp_call=True,method='capped-maximum-native-PDHG',time_limit_s=45.,address_space_bytes=limit,input_loading_inside_clock=True))
 def check():
  if time.monotonic()>deadline:raise Terminal('45s total solver wall deadline')
 try:
  A,a,E=load(full);check();role='primary' if read(full/'stage.json')['primary'] else 'witness'
  assert A.shape[1]==78791 and E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
  assert np.all(a['cost'][:-1]==0) and a['cost'][-1] in(-1,1) and np.all(np.isneginf(a['bounds'][:-1,0])) and np.all(np.isposinf(a['bounds'][:-1,1]))
  old=D/'lifted_feasibility_preparation_490'/role;oa,oz,oe=load(old)
  inherited=same(A,oa) and same(E,oe) and all(np.array_equal(a[k],oz[k]) for k in a);del oa,oz,oe;check()
  write(out/'input_plan.json',dict(inherited=inherited,role=role,full_shape=list(A.shape),full_nnz=A.nnz,all_native_directions=78790,no_torque_projection=True))
  if inherited:
   src=D/'exact_row_quotient_preparation_516'/role/'original';ids=np.load(src.parent/'row_ids.npy');SA=sparse.load_npz(src/'matrix.npz').tocsr();sa=dict(np.load(src/'arrays.npz'));SE=E
   form=D/'capped_max_preparation_538'/role;metricpath=D/'capped_pdhg_preparation_540'/role/'metric.npz'
   for n in ['exact_row_quotient_audit_517','capped_max_audit_539','capped_pdhg_audit_541']:assert read(D/n/'independent_audit.json')['passed']
   assert same(SA,A[ids]) and all(np.array_equal(sa[k],a[k] if k!='rhs' else a[k][ids]) for k in sa)
   H=sparse.load_npz(form/'hard.npz');S=sparse.load_npz(form/'coupled.npz');z=dict(np.load(form/'arrays.npz'));metric=dict(np.load(metricpath));assert np.array_equal(z['source_row_ids'],ids)
   # All current references and their already-proved copies are bound by the frozen launch.
   inputs=[src/'matrix.npz',src/'arrays.npz',src.parent/'row_ids.npy',old/'equality.npz',form/'hard.npz',form/'coupled.npz',form/'arrays.npz',form/'native_equality.npz',metricpath]
   frozen=read(D/'capped_path_launch_542.json')['source_input_sha256']
   for p in inputs:assert sha(p)==frozen[str(p)],str(p)
   proof=dict(passed=True,inherited=True,new_quotient_proof_rows=0,old_split_metric_proofs_repeated=False,proof_sha256={n:sha(D/n/'SHA256.json') for n in ['exact_row_quotient_audit_517','capped_max_audit_539','capped_pdhg_audit_541']})
  else:
   src=out/'source';src.mkdir();led=quotient(A,a['rhs']);check();ids=led['representatives'];SA=A[ids].tocsr();sa=dict(a,rhs=a['rhs'][ids]);SE=E
   sparse.save_npz(src/'matrix.npz',SA);sparse.save_npz(src/'equality.npz',E);np.savez(src/'arrays.npz',**sa);np.save(src/'row_ids.npy',ids);np.savez(src/'ledger.npz',**led)
   form=out/'formulation';form.mkdir();split(SA,sa,ids,form);sparse.save_npz(form/'native_equality.npz',E[:,:-1]);check()
   H=sparse.load_npz(form/'hard.npz');S=sparse.load_npz(form/'coupled.npz');z=dict(np.load(form/'arrays.npz'));metric=exact_metric(H,S);metricpath=out/'metric.npz';np.savez(metricpath,**metric);check()
   proof=prove(A,a,SA,sa,ids,led,H,S,z,metric,out,check);proof.update(inherited=False)
   inputs=[src/'matrix.npz',src/'arrays.npz',src/'equality.npz',src/'row_ids.npy',src/'ledger.npz',form/'hard.npz',form/'coupled.npz',form/'arrays.npz',form/'native_equality.npz',metricpath,out/'new_implication_verified.npy',out/'new_metric_verified.npz']
  assert not np.any(z['native_cost']) and np.all(np.isneginf(z['native_bounds'][:,0])) and np.all(np.isposinf(z['native_bounds'][:,1])) and np.all(z['equality_rhs']==0) and sparse.load_npz(form/'native_equality.npz').nnz==0
  assert len(metric['sigma'])==H.shape[0]+S.shape[0] and len(metric['tau'])==78790 and np.all(np.isfinite(metric['sigma'])) and np.all(np.isfinite(metric['tau'])) and np.all(metric['sigma']>0) and np.all(metric['tau']>0)
  proof.update(source=str(src),formulation=str(form),metric=str(metricpath),role=role,source_row_ids=ids.tolist(),input_sha256={str(p):sha(p) for p in inputs},full_input_sha256={str(full/n):sha(full/n) for n in ['matrix.npz','arrays.npz','equality.npz']},full_rows=A.shape[0],source_rows=SA.shape[0],source_columns=SA.shape[1],full_nnz=A.nnz,source_nnz=SA.nnz)
  write(out/'input_proof.json',proof);check()
  def accept(can):
   w=can['x'];sy=np.zeros(SA.shape[0]);sy[z['hard_source_positions']]=can['hard_dual'];sy[z['coupled_source_positions']]=can['coupled_dual'];fy=np.zeros(A.shape[0]);fy[ids]=sy;tu=float(can['upper_multiplier']);tl=float(can['lower_multiplier'])
   sc,sr,ss=kkt(SA,sa,SE,w,sy,tu,tl);fc,fr,fs=kkt(A,a,E,w,fy,tu,tl);check()
   np.savez_compressed(out/'original_residuals.npz',source=sr,full=fr,source_stationarity=ss,full_stationarity=fs)
   np.savez_compressed(out/'original_duals.npz',source=sy,full=fy,upper=tu,lower=tl)
   result=dict(passed=sc['passed'] and fc['passed'],source=sc,full_original=fc,native_values_unchanged=True,dual_mapping='original representative positions; zero omitted row multipliers',no_torque_backsubstitution=True)
   write(out/'original_checks.json',result)
   if not result['passed']:raise Terminal('full source/original KKT failed; unchanged vector rejected')
  r=solve_lp(H,S,z['hard_rhs'],z['coupled_rhs'],float(z['lower_v']),float(z['upper_v']),float(z['objective_sign']),out/'solve',metric=metric,small_trace=False,start_time=start,candidate_check=accept)
 except (Terminal,AssertionError,MemoryError,ValueError,OverflowError,FloatingPointError,OSError) as exc:
  r=dict(status=1,message='input/resource failure:'+repr(exc),candidate=False,success=False,iterations=0,weighted_proximal_evaluations=0,scalar_reconstruction_attempts=0,elapsed_s=time.monotonic()-start,actual_lp_call=True,full_size_robot_LP=True,small_test=False,physical_admission=False,method='capped-maximum-native-PDHG')
 r.update(child_elapsed_including_evidence_s=time.monotonic()-start,configured_address_space_bytes=limit,peak_memory_measured=False,input_loading_inside_clock=True);write(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
