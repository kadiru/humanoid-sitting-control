"""ONE cold primary-zero-face or actual full-original interior LP; immutable original return."""
from pathlib import Path
import sys,time,resource,json,hashlib
CHILD_START=time.monotonic();START=float(sys.argv[2]);LIMIT=40*1024**3;resource.setrlimit(resource.RLIMIT_AS,(LIMIT,LIMIT))
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from low_seat_balance_path_input_598 import load_problem,same,fresh_face
from certify_low_seat_balance_path_input_599 import prove_full,prove_face
from low_seat_balance_return_598 import exact_point as original_exact_point,lift as original_lift
from diagnostic_common import O, event
def exact_point(*args):
 event('exact_point_'+args[-1]+'_started',START)
 value=original_exact_point(*args)
 event('exact_point_'+args[-1]+'_finished',START)
 return value
def lift(*args):
 event('lift_started',START)
 value=original_lift(*args)
 event('lift_finished',START)
 return value
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Terminal(Exception):pass
def check():
 if time.monotonic()-START>=600:raise Terminal('600s total diagnostic wall deadline')
def kkt(A,a,E,w,y,upper,lower,eqdual):
 vals=A@w-a['rhs'];eq=E@w-a['equality_rhs'];station=a['cost']+A.T@y+E.T@eqdual+upper-lower
 lo=np.isfinite(a['bounds'][:,0]);hi=np.isfinite(a['bounds'][:,1])
 pr=max(0.,float(vals.max(initial=0)),float(abs(eq).max(initial=0)),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 du=float(abs(station).max());gap=abs(float(a['cost']@w+a['rhs']@y+a['equality_rhs']@eqdual+a['bounds'][hi,1]@upper[hi]-a['bounds'][lo,0]@lower[lo]))
 finite=all(np.all(np.isfinite(v)) for v in [w,y,upper,lower,eqdual,vals,eq,station,pr,du,gap]);signs=bool(np.all(y>=0) and np.all(upper>=0) and np.all(lower>=0) and np.all(upper[~hi]==0) and np.all(lower[~lo]==0))
 return dict(primal=pr,dual=du,gap=gap,finite=finite,nonnegative_duals=signs,passed=bool(finite and signs and max(pr,du,gap)<=1e-9)),vals,eq,station
def main():
 out=Path(sys.argv[1]);full=out.parent;actual=False;nit=0;cross=0;native_status=None;backend_elapsed=0.;native_candidate=False;configured_remaining=None;reconstruction_attempts=0;completed_outputs=0;rational_attempted=False;progress={};inherited=full==O/'primary';primary=read(full/'stage.json')['primary']
 try:
  assert inherited and primary, 'Only the frozen initial primary is authorized'
  event('input_checks_started',START)
  check();A,a,E=load_problem(full);assert A.shape[1]==78791 and E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0);check()
  inherited=full==O/'primary'
  systems={'full':(A,a,E)};paths={'full':str(full)};inputs=[];proof=dict(passed=True,inherited=inherited,role='primary' if primary else 'interior',new_quotient_proof_rows=0,old_proofs_repeated=False,initial_face_proof_repeated=False)
  if inherited:
   assert primary;old=D/'lifted_feasibility_preparation_490/primary';oa,oz,oe=load_problem(old)
   assert same(A,oa) and same(E,oe) and all(a[k].tobytes()==oz[k].tobytes() for k in a);del oa,oz,oe;check()
   src=D/'exact_row_quotient_preparation_516/primary/original';SA=sparse.load_npz(src/'matrix.npz').tocsr();sa=dict(np.load(src/'arrays.npz'));ids=np.load(src.parent/'row_ids.npy')
   red=D/'box_support_preparation_546/primary';RA,ra,RE=load_problem(red);keep=np.load(red/'retained_source_positions.npy')
   coupled=D/'coupled_support_preparation_570/primary';CA,ca,CE=load_problem(coupled);ck=np.load(coupled/'retained_546_positions.npy')
   new=D/'zero_face_preparation_572';B,b,H=load_problem(new)
   assert same(SA,A[ids]) and same(RA,SA[keep]) and same(CA,RA[ck]) and same(RE,E) and same(CE,E)
   for zz,rows in [(sa,ids),(ra,ids[keep]),(ca,ids[keep][ck])]:
    for key in a:assert np.array_equal(zz[key],a[key][rows] if key=='rhs' else a[key])
   assert B.shape==(265967,78791) and H.shape==(69617,78791) and H.nnz==749167
   assert np.array_equal(b['cost'],a['cost']) and np.array_equal(b['bounds'],a['bounds'])
   systems.update(face=(B,b,H),coupled=(CA,ca,CE),retained=(RA,ra,RE),source=(SA,sa,E));paths.update(face=str(new),coupled=str(coupled),retained=str(red),source=str(src))
   inputs=[src/'matrix.npz',src/'arrays.npz',src.parent/'row_ids.npy']+list(red.iterdir())+list(coupled.iterdir())+list(new.iterdir())
   frozen=read(D/'balance_path_launch_598.json')['source_input_sha256']
   for p in inputs:assert sha(p)==frozen[str(p)],str(p);check()
   proof['proof_sha256']={n:sha(D/n/'SHA256.json') for n in ['exact_row_quotient_audit_517','box_support_audit_547','definition_audit_559','coupled_support_audit_571','zero_face_audit_573']}
   for n,h in proof['proof_sha256'].items():assert frozen[str(D/n/'SHA256.json')]==h and read(D/n/'independent_audit.json')['passed']
   analytic=dict(np.load(new/'analytic_lower_bound_dual.npz'));check()
   graph=D/'affine_output_preparation_578';foot=D/'foot_block_preparation_586';balance=D/'balance_block_preparation_594';encoded=D/'balance_encoding_preparation_596'
   B=sparse.load_npz(encoded/'inequality_matrix.npz').tocsr();b=dict(np.load(encoded/'arrays.npz'));H=sparse.load_npz(encoded/'equality_matrix.npz').tocsr();maps=dict(np.load(balance/'maps.npz'));retained=maps['original_columns']
   assert B.shape==(265967,10541) and B.nnz==6102946 and H.shape==(1367,10541) and H.nnz==1
   assert np.isfinite(b['bounds']).all() and retained[-1]==78790
   systems['encoded']=(B,b,H);paths['encoded']=str(encoded)
   for folder in [graph,foot,balance,encoded,D/'balance_block_audit_595',D/'balance_encoding_audit_597',D/'foot_encoding_preparation_588',D/'affine_output_audit_579',D/'graph_encoding_preparation_580',D/'graph_encoding_audit_581',D/'foot_block_audit_587',D/'foot_encoding_audit_589']:
    for p in folder.iterdir():
     if p.is_file():assert sha(p)==frozen[str(p)],str(p);inputs.append(p);check()
   for n in ['affine_output_audit_579','graph_encoding_audit_581','foot_block_audit_587','foot_encoding_audit_589','balance_block_audit_595','balance_encoding_audit_597']:
    proof['proof_sha256'][n]=sha(D/n/'SHA256.json');assert read(D/n/'independent_audit.json')['passed']
   proof.update(exact_graph_path=str(graph),exact_foot_graph_path=str(foot),exact_balance_graph_path=str(balance),encoded_inner_input=True,inequality_block_inner_implication=True,approximate_equality_surrogate=False,explicit_outer_bounds_imposed=True,retained_columns_path=str(balance/'maps.npz'),old_graph_and_encoding_proofs_repeated=False)
   check()
  else:
   proof['new_full_input']=prove_full(full,check);proof['all_original_rows_retained_without_reduction']=True
   if primary:
    new=out/'face';new.mkdir();B,b,H=fresh_face(A,a,E,new,check);proof['new_face_input']=prove_face(A,a,E,B,b,H,out,check)
    systems['face']=(B,b,H);paths['face']=str(new);inputs+=list(new.iterdir())+[out/'new_face_pairs_verified.npy']
   else:B,b,H=A,a,E
   # No newly constructed analytic vector: reuse immutable573 lower/upper arrays.
   analytic=dict(np.load(D/'zero_face_preparation_572/analytic_lower_bound_dual.npz'))
   inputs+=[D/'zero_face_preparation_572/analytic_lower_bound_dual.npz']
  if primary:
   assert a['cost'][78790]==1 and np.count_nonzero(a['cost'])==1 and a['bounds'][78790,0]==0
   assert np.all(analytic['lower'][:78790]==0) and analytic['lower'][78790]==1 and np.all(analytic['upper']==0)
  proof.update(system_paths=paths,solver_system='encoded' if inherited else ('face' if primary else 'full'),solver_inequality_shape=list(B.shape),solver_inequality_nnz=int(B.nnz),solver_equality_shape=list(H.shape),solver_equality_nnz=int(H.nnz),original_columns=78791,original_scalar_column=78790,input_sha256={str(p):sha(p) for p in inputs},full_input_sha256={str(full/n):sha(full/n) for n in ['matrix.npz','arrays.npz','equality.npz']})
  write(out/'input_proof.json',proof);check();configured_remaining=300-(time.monotonic()-START)
  if configured_remaining<=0:raise Terminal('300s backend deadline exhausted during setup')
  event('input_checks_finished',START)
  options=dict(time_limit=configured_remaining,maxiter=4096,presolve=True,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
  write(out/'scipy_started.json',dict(actual_lp_call=True,method='highs-ipm',construction='condensed-balance-inner-zero-face-IPM',primary=primary,options=options,cold_start=True,old_internal_state_used=False,input_loading_inside_clock=True))
  event('backend_started',START)
  actual=True;bs=time.monotonic();r=linprog(b['cost'],A_ub=B,b_ub=b['rhs'],A_eq=H,b_eq=b['equality_rhs'],bounds=b['bounds'],method='highs-ipm',options=options);backend_elapsed=time.monotonic()-bs
  event('backend_returned',START,backend_elapsed_s=backend_elapsed)
  if time.monotonic()-START>=300:raise Terminal('backend returned at or after 300s deadline')
  nit=int(r.nit);cross=int(getattr(r,'crossover_nit',0));native_status=int(r.status);native_candidate=r.x is not None
  write(out/'backend_result.json',dict(status=native_status,message=str(r.message),success=bool(r.success),candidate_returned=native_candidate,iterations=nit,crossover_iterations=cross,elapsed_s=backend_elapsed,configured_time_limit_s=configured_remaining,internal_vectors_saved=False))
  check()
  if native_status!=0 or not r.success or r.x is None:raise Terminal('IPM returned no successful complete original vector: '+str(r.message))
  raw=np.asarray(r.x);assert raw.dtype==np.dtype('float64') and raw.shape==((10541,) if inherited else (78791,)) and np.isfinite(raw).all()
  np.save(out/'backend_primal.npy',raw)
  if primary and raw[-1]!=0:raise Terminal('returned primary scalar is not exactzero; no postsolve repair')
  exact_check=None
  if inherited:
   rational_attempted=True;balancecheck=exact_point(raw,balance,out/'rational594',analytic,check,progress,'594')
   if not balancecheck['passed']:raise Terminal('exact594 point residual gate failed; no repair')
   reconstruction_attempts=1;w,liftinfo=lift(raw,balance,foot,graph,out,check,progress);completed_outputs=liftinfo['auxiliary_outputs'];check()
   footcheck=exact_point(w[np.load(foot/'maps.npz')['original_columns']],foot,out/'rational586',analytic,check,progress,'586')
   if not footcheck['passed']:raise Terminal('final once-rounded586 point residual gate failed; no repair')
   auxcheck=exact_point(w[np.load(graph/'maps.npz')['retained_columns']],graph,out/'rational578',analytic,check,progress,'578')
   exact_check=dict(passed=balancecheck['passed'] and footcheck['passed'] and auxcheck['passed'],balance594=balancecheck,foot586=footcheck,auxiliary578=auxcheck)
   write(out/'exact_reduced_checks.json',exact_check);check()
   if not exact_check['passed']:raise Terminal('final rounded578 point gate failed; no repair')
  else:
   w=raw.copy();assert raw.tobytes()==w.tobytes();np.save(out/'returned_native.npy',w)
  if primary:
   upper=analytic['upper'].copy();lower=analytic['lower'].copy()
   duals={name:(np.zeros(M.shape[0]),upper[retained] if name=='encoded' else upper,lower[retained] if name=='encoded' else lower,np.zeros(EQ.shape[0])) for name,(M,z,EQ) in systems.items()}
  else:
   y=-np.asarray(r.ineqlin.marginals);ed=-np.asarray(r.eqlin.marginals);upper=-np.asarray(r.upper.marginals);lower=np.asarray(r.lower.marginals)
   np.savez_compressed(out/'backend_marginals.npz',inequality=r.ineqlin.marginals,equality=r.eqlin.marginals,upper=r.upper.marginals,lower=r.lower.marginals);duals={'full':(y,upper,lower,ed)}
  event('numeric_checks_started',START)
  checks={};res={};saved_duals={}
  for name,(M,z,EQ) in systems.items():
   event('numeric_'+name+'_started',START)
   dual,up,lo,ed=duals[name];checks[name],res[name],res[name+'_equality'],res[name+'_stationarity']=kkt(M,z,EQ,raw if name=='encoded' else w,dual,up,lo,ed)
   event('numeric_'+name+'_finished',START)
   for tag,val in [('row',dual),('upper',up),('lower',lo),('equality',ed)]:saved_duals[name+'_'+tag]=val
   check()
  np.savez_compressed(out/'original_residuals.npz',**res);np.savez_compressed(out/'original_duals.npz',**saved_duals)
  passed=all(c['passed'] for c in checks.values()) and (exact_check is None or exact_check['passed']);write(out/'original_checks.json',dict(passed=passed,**checks,native_values_unchanged=not inherited,retained_values_unchanged=True,scalar_values_unchanged=True,primary_scalar_exactzero=bool(w[78790]==0) if primary else None,analytic_dual_used=primary,coordinate_reconstruction=inherited,exact_reduced=exact_check));check()
  if not passed:raise Terminal('all required original numerical KKT gates failed; no repair')
  assert raw.tobytes()==(w[retained].tobytes() if inherited else w.tobytes());np.save(out/'witness.npy',w);check();result=dict(status=0,message='complete original vector with unchanged retained values passed all required numerical KKT gates',candidate=True,success=True)
 except (Terminal,AssertionError,MemoryError,ValueError,OverflowError,FloatingPointError,OSError) as exc:
  if (out/'witness.npy').exists():(out/'witness.npy').rename(out/'rejected_after_terminal.npy')
  result=dict(status=1,message=type(exc).__name__+': '+str(exc),candidate=False,success=False)
 result.update(primary=primary,iterations=nit,crossover_iterations=cross,native_status=native_status,native_candidate_returned=native_candidate,elapsed_s=time.monotonic()-START,child_startup_elapsed_s=time.monotonic()-CHILD_START,parent_preparation_charged_s=CHILD_START-START,backend_elapsed_s=backend_elapsed,configured_backend_seconds=configured_remaining,actual_lp_call=actual,full_size_robot_LP=actual,small_test=False,method='condensed-balance-inner-zero-face-IPM',configured_address_space_bytes=LIMIT,peak_memory_measured=False,input_loading_inside_clock=True,coordinate_reconstruction_attempts=reconstruction_attempts,completed_auxiliary_outputs=progress.get('completed_auxiliary_outputs',0),exact_rational_point_check_attempted=rational_attempted,physical_admission=False,new_point_arithmetic_progress=progress,completed_rootxyf_outputs=progress.get('completed_rootxyf_outputs',0),completed_leg_outputs=progress.get('completed_leg_outputs',0),complete_system_checks_required=9 if inherited else (2 if primary else 1))
 write(out/'result.json',result);event('numerical_result_written',START,candidate=result['candidate']);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
