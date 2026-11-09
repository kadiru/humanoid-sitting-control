"""ONE bounded event-aligned complete nominal path solve; no physics/runtime.
895 free path coordinates plus one command/rate/headroom feasibility slack.
Reuse379 event budgets exactly; retain all original rows and explicit clock targets.
"""
from pathlib import Path
import json,hashlib,datetime,time,numpy as np
from scipy.optimize import minimize
from low_seat_event_clock_v2_386 import make_event,GROUPS
from low_seat_support_rows_371 import transformed
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'event_path_388'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);prep=json.loads((D/'event_clock_preparation_v2_386/preparation.json').read_text());audit=json.loads((D/'event_clock_preparation_audit_387/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
 for record in [prep['source_input_sha256'],audit['source_sha256']]:
  for path,h in record.items():assert sha(path)==h,path
 d=make_event();p=d['p'];f=np.load(D/'event_clock_preparation_v2_386/fixture.npz');assert np.array_equal(f['z'],d['z0']);scale=d['scale'];keep=d['keep'];hard=d['hard'];assert np.array_equal(f['scale'],scale) and np.array_equal(f['keep'],keep) and np.array_equal(f['hard'],hard)
 z0=np.r_[f['z'],float(f['initial_slack'])];history=[];start=time.monotonic();last_valid=z0.copy()
 def constraints(z):
  eq,iq=d['evaluate'](z[:-1]);return transformed(eq,iq,scale,keep,hard,z[-1])
 def derivatives(z):
  E,I=d['evaluate'](z[:-1],True);return np.c_[E,np.zeros(len(E))],np.c_[I[keep]/scale[keep,None],(~hard[keep]).astype(float)]
 def callback(z):
  nonlocal last_valid
  eq,iq=d['evaluate'](z[:-1]);last_valid=z.copy();row=dict(iteration=len(history)+1,elapsed_s=time.monotonic()-start,slack=float(z[-1]),normalized_hover_residual=float(eq[0]),minimum_inequality=float(iq.min()),maximum_scaled_violation=float(max(0.,np.max(-iq[keep]/scale[keep]),np.max(abs(eq)))),maximum_scaled_hard_violation=float(max(0.,np.max(-iq[keep&hard]/scale[keep&hard]),np.max(abs(eq)))),minimum_augmented_inequality=float(constraints(z)[1].min()),minimum_clock_rate_row=float(d['last']['clock_full'][:,:32].min()),minimum_clock_target_row=float(d['last']['clock_full'][:,32:].min()),maximum_kinematic_residual=float(abs(d['last']['Y'][:103,:14]).max()));history.append(row);np.save(O/'last_iterate.npy',z);(O/'iterations.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 initial=dict(np.load(D/'support_preparation_371/fixture.npz'));initial.update({k:f[k] for k in f.files if k not in ['E','I','fd','analytic']});np.savez_compressed(O/'initial.npz',**initial)
 hashpaths=[Path(__file__).resolve(),Path('scripts/export_low_seat_support_373.py').resolve(),Path('scripts/audit_low_seat_event_path_389.py').resolve(),D/'event_clock_preparation_v2_386/preparation.json',D/'event_clock_preparation_v2_386/fixture.npz',D/'event_clock_preparation_audit_387/independent_audit.json'];hashes=dict(prep['source_input_sha256']);hashes.update({str(q):sha(q) for q in hashpaths})
 revision=dict(scope=__doc__,source_input_sha256=hashes,optimizer='SLSQP',maximum_iterations=30,maximum_optimizer_runs=1,objective='one dimensionless command/rate/headroom slack, no secondary time objective',original_support_rows=20284,explicit_clock_target_rows=21600,clock_rate_final_rows=21600,optimizer_inequalities=len(scale),hard_rows=int((hard&keep).sum()),soft_rows=int((~hard&keep).sum()),hard_hover_equalities=1,free_path_variables=895,epigraph_slacks=1,event_knots=[0,16,64,99,104,114],event_budgets=d['budgets'].tolist(),movement_ticks=650,minimum_synthetic_quiet_ticks=25,total_ticks=675,interior_knots_need_not_align_to_clock=True,segment_floor_s=.02,original_physical_criteria_unchanged=True,full_geometry_mandatory_final_screen=True,exact_float32_filter_mandatory_separate_gate=True,all_intermediate_geometry_mandatory_separate_gate=True,positive_slack_never_admission=True,initial_original_feasibility_not_claimed=True,physical_admission=False)
 (O/'revision.json').write_text(json.dumps(revision,indent=2));gradient=np.r_[np.zeros(895),1.]
 try:
  fit=minimize(lambda z:float(z[-1]),z0,jac=lambda z:gradient,method='SLSQP',bounds=d['bounds']+[(0,None)],constraints=[dict(type='eq',fun=lambda z:constraints(z)[0],jac=lambda z:derivatives(z)[0]),dict(type='ineq',fun=lambda z:constraints(z)[1],jac=lambda z:derivatives(z)[1])],callback=callback,options=dict(maxiter=30,ftol=1e-10));z=fit.x;opt=dict(success=bool(fit.success),message=fit.message,iterations=int(fit.nit),evaluations=int(fit.nfev))
 except RuntimeError as ex:
  z=last_valid;opt=dict(success=False,message='terminated on bounded dependent-coordinate domain failure; last callback or initial retained',iterations=len(history),evaluations=None,domain_error=str(ex))
 eq,iq=d['evaluate'](z[:-1]);saved=dict(d['last']);opt['minimum_augmented_inequality']=float(transformed(eq,iq,scale,keep,hard,z[-1])[1].min())
 # Export fixed final x through the unchanged geometry/static model. The wrapped
 # evaluator returns this NEW formulation's complete rows, without another solve.
 ep=dict(p);ep['evaluate']=lambda x:(eq,iq);result=export_result(ep,saved['x'],opt,revision,hashes,O,float(z[-1]),scale,keep,hard)
 extra=dict(parameters=z[:-1],budgets=d['budgets'],raw_ineq=saved['raw'],tighten=saved['tighten'],clock=saved['samples'],clock_raw=saved['clock_raw'],clock_full=saved['clock_full'],locations=saved['locations'],targets=saved['targets'],ulp=d['ulp']);np.savez_compressed(O/'event_clock.npz',**extra)
 result['event_clock']=dict(event_budgets=d['budgets'].tolist(),movement_ticks=650,minimum_synthetic_quiet_ticks=25,total_ticks=675,integer_time_s=13.5,minimum_original_inequality=float(saved['raw'].min()),minimum_full_clock_row=float(saved['clock_full'].min()),minimum_original_clock_rate_row=float(saved['clock_raw'][:,:32].min()),minimum_original_clock_target_row=float(saved['clock_raw'][:,32:].min()),all_clock_rate_rows_passed=bool(np.all(saved['clock_full'][:,:32]>=0)),all_clock_target_rows_passed=bool(np.all(saved['clock_full'][:,32:]>=0)),actual_float32_schedule=False,intermediate_geometry_checked=False,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({q.name:sha(q) for q in O.iterdir() if q.is_file()},indent=2));print(json.dumps(result['optimizer'],indent=2));print(json.dumps(result['event_clock'],indent=2))
if __name__=='__main__':main()
