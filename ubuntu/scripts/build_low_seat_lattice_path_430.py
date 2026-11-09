"""ONE integer-exact all-hard closest-path nonlinear construction.
New integer allocation is the proved428 minimax necessary-duration allocation
from saved424 EXPORTED ROOT, not any unvisited branch or chosen callback.
ALL894 continuous path coordinates vary jointly in the EXACT406-centered416
box. Every retained original/support/window/old-bound row is HARD with the
additional dimensionless2^-23 inward buffer. ZERO feasibility slack variables.
One30-iteration SLSQP call minimizes normalized squared change from the saved
root; no restart, homotopy, callback selection, alternate allocation or fallback.
All full geometry/clock/filter/runtime/physical admission gates remain separate.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,time
from scipy.optimize import minimize,Bounds
from low_seat_monotone_phase_410 import make_phase
from low_seat_lattice_projection_428 import rows,objective,BUFFER,GROUPS
from low_seat_budget_domain_422 import full_clock
from export_low_seat_lattice_430 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'lattice_path_430'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);prep=json.loads((D/'lattice_preparation_428/preparation.json').read_text());audit=json.loads((D/'lattice_preparation_audit_429/independent_audit.json').read_text());launch=json.loads((D/'lattice_path_launch_430.json').read_text());assert prep['passed'] and audit['passed'];hashes=launch['source_input_sha256']
 for p,h in hashes.items():assert sha(p)==h,p
 f=dict(np.load(D/'lattice_preparation_428/fixture.npz'));base=dict(np.load(D/'window_path_400/initial.npz'));d=make_phase();w=d['d'];p=d['p'];budgets=f['budgets'];reference=f['reference'];keep=f['keep'];scale=f['scale'];hard=np.ones(27708,bool);lo=f['lo'];hi=f['hi'];radius=f['radius'];history=[];started=time.monotonic();counts=dict(primal_requests=0,jacobian_requests=0,primal_cache_misses=0)
 def evaluate(x,derivative=False):
  z=np.r_[x,budgets];counts['jacobian_requests' if derivative else 'primal_requests']+=1
  if 'z' not in d['last'] or not np.array_equal(z,d['last']['z']):counts['primal_cache_misses']+=1
  return d['evaluate'](z,derivative)
 def constraints(x):
  ee,ii=evaluate(x);return rows(ii,d['last']['v'],scale,keep,lo,hi)
 def jac(x):
  ee,ii=evaluate(x);E,I=evaluate(x,True);return rows(ii,d['last']['v'],scale,keep,lo,hi,I,d['last']['G'])[1]
 def callback(x):
  ee,ii=evaluate(x);ar=rows(ii,d['last']['v'],scale,keep,lo,hi);n=len(history);value=objective(x,reference,radius)[0]
  rec=dict(callback=n,elapsed_s=time.monotonic()-started,objective=value,minimum_all_hard_row=float(ar.min()),maximum_scaled_model_violation=float(max(0.,np.max(-ii[keep]/scale[keep]))),hover_residual=float(abs(ee).max()),not_selected=True)
  np.savez_compressed(O/('callback_%02d.npz'%n),solver=x,z=np.r_[x,budgets],eq=ee,ineq=ii,v=d['last']['v'],hard_rows=ar);history.append(rec);(O/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(rec),flush=True)
 ee,ii=evaluate(reference);initialrows=rows(ii,d['last']['v'],scale,keep,lo,hi)
 revision=dict(scope=__doc__,reference='424 exportedroot final, newresearchinput only',budgets=budgets.tolist(),allocation_proof='lattice_preparation_428/allocation_proof.json',all_hard_rows=len(initialrows),free_path_coordinates=894,feasibility_slack_variables=0,objective='half sum squared normalized complete-path coordinate change',extra_inward_buffer=BUFFER,maximum_optimizer_calls=1,maximum_iterations=30,ftol=1e-10,continuous_box='EXACT416 centered406',initial_model_minimum=float(ii[keep].min()),initial_all_hard_minimum=float(initialrows.min()),initial_admitted=False,callbacks_not_selected=True,physical_admission=False)
 (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'initial.npz',**base);np.savez_compressed(O/'structural_initial.npz',z=np.r_[reference,budgets],reference=reference,radius=radius,eq=ee,ineq=ii,hard_rows=initialrows,lower=f['lower'],upper=f['upper'],lo=lo,hi=hi,keep=keep,scale=scale,hard=hard)
 try:
  fit=minimize(lambda x:objective(x,reference,radius)[0],reference,jac=lambda x:objective(x,reference,radius)[1],method='SLSQP',bounds=Bounds(f['lower'],f['upper']),constraints=[dict(type='ineq',fun=constraints,jac=jac)],callback=callback,options=dict(maxiter=30,ftol=1e-10,disp=True))
 except Exception as exc:
  (O/'termination.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),callbacks=len(history),restarted=False,physical_admission=False),indent=2));raise
 x=fit.x;ee,ii=evaluate(x);saved=dict(d['last']);ar=rows(ii,saved['v'],scale,keep,lo,hi);z=np.r_[x,budgets];samples,times=full_clock(saved['C'],saved['dt']);dc=np.diff(samples,axis=0);targets=10*samples[1:]-9*samples[:-1];rawclock=np.c_[.02*p['rates']-dc,.02*p['rates']+dc,p['command_hi']-targets,targets-p['command_lo']];clock=rawclock-np.r_[4*w['ulp'],4*w['ulp'],32*w['ulp'],32*w['ulp']]
 opt=dict(success=bool(fit.success),status=int(fit.status),message=str(fit.message),iterations=int(fit.nit),evaluations=int(fit.nfev),objective_nfev=int(fit.nfev),jacobian_njev=int(fit.njev),nearest_path_objective=float(fit.fun));ep=dict(p);ep['evaluate']=lambda _:(ee,ii)
 result=export_result(ep,saved['x'],opt,revision,hashes,O,0.,scale,keep,hard)
 cm=float(min((saved['v'][:786]-lo).min(),(hi-saved['v'][:786]).min()));boxmin=float(min((x-f['lower']).min(),(f['upper']-x).min()));numerical=bool(ar.min()>=-1e-8 and abs(ee).max()<=1e-8 and boxmin>=0);strict=bool(ar.min()>=0 and abs(ee).max()<=1e-8 and boxmin>=0)
 result.update(algorithm=dict(optimizer_calls=1,callback_count=len(history),callbacks_not_selected=True,history=history,all_constraints_hard=True,feasibility_slack_variables=0,minimum_buffered_hard_row=float(ar.min()),buffered_hard_numerical_passed=numerical,buffered_hard_strict_passed=strict,minimum_original_coordinate_bound=cm,minimum_box_bound=boxmin,request_counts=counts,count_scope='Wrapper primal/J requests and primal cache misses; not all internal FK/root operations',new_integer_construction=True,prior_tree_continued=False),event_clock=dict(event_budgets=budgets.tolist(),mandatory_events_exactly_integer=True,event_integer_error=0.,total_ticks=675,integer_time_s=13.5,minimum_original_clock_rate_row=float(rawclock[:,:32].min()),minimum_original_clock_target_row=float(rawclock[:,32:].min()),all_tightened_clock_rate_rows_passed=bool(np.all(clock[:,:32]>=0)),all_tightened_clock_target_rows_passed=bool(np.all(clock[:,32:]>=0)),actual_float32_schedule=False,physical_admission=False),window=dict(breakpoints=232,minimum_original_target_row=float(saved['env']['raw'].min()),minimum_tightened_target_row=float(saved['env']['tightened'].min()),all_strict_window_rows_passed=bool(np.all(saved['env']['tightened']>=0))))
 core=w['last'];np.savez_compressed(O/'event_clock.npz',parameters=saved['v'][:895],budgets=budgets,raw_ineq=core['raw'],tighten=core['tighten'],clock=samples,clock_raw=rawclock,clock_full=clock,times=times,targets=targets,ulp=w['ulp']);np.savez_compressed(O/'window.npz',**saved['env']);np.savez_compressed(O/'structural.npz',**{k:v for k,v in saved.items() if isinstance(v,np.ndarray)});np.save(O/'parameters.npy',z);np.save(O/'solver_final.npy',x);np.save(O/'hard_final.npy',ar)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file()},indent=2));print(json.dumps({k:result[k] for k in ['optimizer','event_clock','window']},indent=2),flush=True)
if __name__=='__main__':main()
