"""ONE bounded mixed-integer tangent restoration, followed by nonlinear rejection screens.
All whole-path shape/COM/force and interior timings vary jointly with five integer
mandatory-event budgets. This is a different linearized formulation/objective from388,
not a resumed SLSQP run. No restart, second allocation solve or nonlinear polishing.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,time
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_array
from low_seat_event_clock_v2_386 import make_event,GROUPS
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'integer_restoration_path_394'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);prep=json.loads((D/'integer_restoration_preparation_392/preparation.json').read_text());aud=json.loads((D/'integer_restoration_preparation_audit_393/independent_audit.json').read_text());assert prep['passed'] and aud['passed']
 for p,h in {**prep['source_input_sha256'],**aud['source_sha256']}.items():assert sha(p)==h,p
 np.savez_compressed(O/'initial.npz',**dict(np.load(D/'event_path_388/initial.npz')));f=np.load(D/'integer_restoration_preparation_392/fixture.npz');keep=f['keep'];scale=f['scale'];rad=f['radius'];n=900;m=int(keep.sum())
 # Delta coordinates plus rho. rho is a step-size objective, NOT feasibility slack.
 A=np.zeros((m+2+2*n,n+1));lower=np.full(len(A),-np.inf);upper=np.full(len(A),np.inf)
 A[:m,:n]=f['I'][keep]/scale[keep,None];lower[:m]=-f['ineq'][keep]/scale[keep]
 A[m,:n]=f['E'][0];lower[m]=upper[m]=-float(f['eq'][0]);A[m+1,895:900]=1.;lower[m+1]=upper[m+1]=0.
 for k in range(n):
  A[m+2+k,k]=1/rad[k];A[m+2+k,n]=-1.;upper[m+2+k]=0.
  A[m+2+n+k,k]=-1/rad[k];A[m+2+n+k,n]=-1.;upper[m+2+n+k]=0.
 blo=np.r_[f['lower'],0.];bhi=np.r_[f['upper'],1.];cost=np.r_[np.zeros(n),1.];integrality=np.zeros(n+1,int);integrality[895:900]=1
 hashes={**prep['source_input_sha256'],**aud['source_sha256']};hashes.update({str(p.resolve()):sha(p) for p in [Path(__file__),Path('scripts/export_low_seat_support_373.py'),Path('scripts/audit_low_seat_integer_path_395.py')]})
 revision=dict(scope=__doc__,source_input_sha256=hashes,solver='scipy.optimize.milp / HiGHS',maximum_solver_calls=1,time_limit_s=90,node_limit=1000,relative_gap=0,objective='minimum maximum coordinate displacement normalized by the declared trust box',feasibility_slack_variables=0,all20284_original_support_rows_final_checked=True,omitted_exact_constant_rows=392,omitted_time_identity_rows=1,full_clock_rows_final_checked=True,clock_rows_not_in_linear_budget_subproblem=True,mandatory_event_budgets_are_integer=True,normal_budget_at_least101=True,total_movement_ticks=650,synthetic_quiet_ticks=25,physical_admission=False)
 (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'linear_problem.npz',A=A,lower=lower,upper=upper,bounds_lower=blo,bounds_upper=bhi,cost=cost,integrality=integrality)
 start=time.monotonic();fit=milp(cost,integrality=integrality,bounds=Bounds(blo,bhi),constraints=LinearConstraint(csc_array(A),lower,upper),options=dict(time_limit=90,node_limit=1000,mip_rel_gap=0));elapsed=time.monotonic()-start
 status=dict(success=bool(fit.success),status=int(fit.status),message=str(fit.message),solver_wall_s=elapsed,mip_node_count=getattr(fit,'mip_node_count',None),mip_gap=getattr(fit,'mip_gap',None),mip_dual_bound=getattr(fit,'mip_dual_bound',None),objective=None if fit.fun is None else float(fit.fun),candidate_available=fit.x is not None,solver_calls=1,nonlinear_optimizer_calls=0)
 (O/'solver_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if fit.x is None:
  (O/'results.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,solver=status,candidate_available=False,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0),indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));return
 step=np.asarray(fit.x);np.save(O/'linear_solution.npy',step);budget_float=f['budgets']+step[895:900];assert abs(budget_float-np.rint(budget_float)).max()<1e-7
 budgets=np.rint(budget_float).astype(int);assert budgets.sum()==650 and budgets[3]>=101
 # Integer conversion is explicit solver-integrality decoding, NOT rounding failed
 # physical rows. The original floating witness remains saved and independently audited.
 z=f['z']+step[:895];d=make_event();d['budgets'][:]=budgets;d['last'].clear();eq,iq=d['evaluate'](z);saved=dict(d['last']);p=d['p'];opt=dict(success=bool(fit.success),message=fit.message,iterations=int(getattr(fit,'mip_node_count',0) or 0),evaluations=1,milp_success_is_not_nonlinear_feasibility=True,minimum_augmented_inequality=float((iq[d['keep']]/d['scale'][d['keep']]).min()))
 ep=dict(p);ep['evaluate']=lambda x:(eq,iq);result=export_result(ep,saved['x'],opt,revision,hashes,O,0.,d['scale'],d['keep'],d['hard'])
 result['optimizer']['objective_is_dimensionless_slack']=False;result['optimizer']['objective_is_normalized_step_radius']=True;result['optimizer']['normalized_step_radius']=float(step[-1]);result['optimizer']['feasibility_slack_variables']=0
 result.update(solver=status,candidate_available=True,event_clock=dict(event_budgets=budgets.tolist(),previous_event_budgets=f['budgets'].tolist(),event_budget_change=(budgets-f['budgets']).tolist(),integer_decode_max_error=float(abs(budget_float-budgets).max()),total_ticks=675,integer_time_s=13.5,minimum_original_clock_rate_row=float(saved['clock_raw'][:,:32].min()),minimum_original_clock_target_row=float(saved['clock_raw'][:,32:].min()),all_clock_rate_rows_passed=bool(np.all(saved['clock_full'][:,:32]>=0)),all_clock_target_rows_passed=bool(np.all(saved['clock_full'][:,32:]>=0)),actual_float32_schedule=False,intermediate_geometry_checked=False,physical_admission=False))
 np.savez_compressed(O/'event_clock.npz',parameters=z,budgets=budgets,raw_ineq=saved['raw'],tighten=saved['tighten'],clock=saved['samples'],clock_raw=saved['clock_raw'],clock_full=saved['clock_full'],locations=saved['locations'],targets=saved['targets'],ulp=d['ulp'])
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({q.name:sha(q) for q in O.iterdir() if q.is_file()},indent=2));print(json.dumps(result['optimizer'],indent=2));print(json.dumps(result['event_clock'],indent=2))
if __name__=='__main__':main()
