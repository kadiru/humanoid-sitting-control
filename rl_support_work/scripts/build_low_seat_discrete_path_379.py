"""ONE bounded complete integer-timed nominal path, zero physics/runtime.
Fixed650 movement ticks+25 quiet; optimize complete geometry/COM/forces, with inward numerical envelopes.
Contact, phase, planning, support and hover constraints receive NO feasibility slack.
"""
from pathlib import Path
import json,hashlib,datetime,time
import numpy as np
from scipy.optimize import minimize
from low_seat_discrete_problem_377 import make_discrete
from low_seat_support_rows_371 import rows,transformed
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'discrete_path_379'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);prep=json.loads((D/'discrete_preparation_v2_377/preparation.json').read_text());audit=json.loads((D/'discrete_preparation_audit_378/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
 for record in [prep['source_input_sha256'],audit['source_sha256']]:
  for path,h in record.items():assert sha(path)==h,path
 d=make_discrete();p=dict(d['p']);p['evaluate']=d['evaluate'];f=np.load(D/'discrete_preparation_v2_377/fixture.npz');x0=f['x'].copy();free=d['free'];scale=d['scale'];keep=d['keep'];hard=d['hard'];groups=d['groups'];proof=d['proof'];assert np.array_equal(x0,d['x0']) and np.array_equal(free,f['free']);assert np.array_equal(scale,f['scale']) and np.array_equal(keep,f['keep']) and np.array_equal(hard,f['hard'])
 z0=np.r_[x0[free],float(f['initial_slack'])];history=[];start=time.monotonic();last_valid=z0.copy()
 def expand(z):
  x=x0.copy();x[free]=z[:-1];return x
 def constraints(z):
  eq,iq=p['evaluate'](expand(z));return transformed(eq,iq,scale,keep,hard,z[-1])
 def derivatives(z):
  E,I=p['evaluate'](expand(z),True);return np.c_[E[:,free],np.zeros(len(E))],np.c_[I[keep][:,free]/scale[keep,None],(~hard[keep]).astype(float)]
 def callback(z):
  nonlocal last_valid
  x=expand(z);eq,iq=p['evaluate'](x);last_valid=z.copy();row=dict(iteration=len(history)+1,elapsed_s=time.monotonic()-start,slack=float(z[-1]),time_s=float(.5+x[1026:].sum()),normalized_hover_residual=float(eq[0]),minimum_raw_inequality=float(iq.min()),maximum_scaled_original_violation=float(max(0.,np.max(-iq[keep]/scale[keep]),np.max(abs(eq)))),maximum_scaled_hard_violation=float(max(0.,np.max(-iq[keep&hard]/scale[keep&hard]),np.max(abs(eq)))),minimum_augmented_inequality=float(constraints(z)[1].min()),maximum_kinematic_residual=float(abs(p['last']['Y'][:103,:14]).max()));history.append(row);np.save(O/'last_iterate.npy',z);(O/'iterations.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 initial=dict(np.load(D/'support_preparation_371/fixture.npz'));initial.update({k:f[k] for k in f.files});np.savez_compressed(O/'initial.npz',**initial)
 hashpaths=[Path(__file__).resolve(),Path('scripts/export_low_seat_support_373.py').resolve(),Path('scripts/audit_low_seat_discrete_path_380.py').resolve(),D/'discrete_preparation_v2_377/preparation.json',D/'discrete_preparation_v2_377/fixture.npz',D/'discrete_preparation_audit_378/independent_audit.json'];hashes=dict(prep['source_input_sha256']);hashes.update({str(q):sha(q) for q in hashpaths})
 preparation=dict(scope=__doc__,source_input_sha256=hashes,optimizer='SLSQP',maximum_iterations=30,maximum_optimizer_runs=1,objective='dimensionless command/rate/headroom/time epigraph slack only; no time objective',row_scale_groups=groups,removed_constant_rows=proof,original_inequalities=19760,added_support_inequalities=len(p['support_h']),all_inequalities=len(scale),hard_rows=int((hard&keep).sum()),soft_rows=int((~hard&keep).sum()),hard_hover_equalities=1,free_path_variables=len(free),epigraph_slacks=1,full_geometry_is_mandatory_final_screen=True,unchanged_physical_criteria=True,changed_research_target='fixed20ms counts; all103 nominal pose/COM and114force coordinates optimized, with stricter numerical interior and rounded-filter envelopes',movement_ticks=650,normal_ticks=101,minimum_synthetic_quiet_ticks=25,integer_budget_ticks=675,tick_allocation=prep['tick_allocation'],construction_reserves_are_not_physical_tolerances=True,exact_filter_is_mandatory_separate_gate=True,initial_feasibility_not_claimed=True,physical_admission=False)
 (O/'revision.json').write_text(json.dumps(preparation,indent=2));gradient=np.r_[np.zeros(len(free)),1.]
 try:
  fit=minimize(lambda z:float(z[-1]),z0,jac=lambda z:gradient,method='SLSQP',bounds=[p['bounds'][k] for k in free]+[(0,None)],constraints=[dict(type='eq',fun=lambda z:constraints(z)[0],jac=lambda z:derivatives(z)[0]),dict(type='ineq',fun=lambda z:constraints(z)[1],jac=lambda z:derivatives(z)[1])],callback=callback,options=dict(maxiter=30,ftol=1e-10));z=fit.x;opt=dict(success=bool(fit.success),message=fit.message,iterations=int(fit.nit),evaluations=int(fit.nfev))
 except RuntimeError as ex:
  z=last_valid;opt=dict(success=False,message='terminated on bounded dependent-coordinate domain failure; last callback or initial retained',iterations=len(history),evaluations=None,domain_error=str(ex))
 solution=expand(z);eq,iq=p['evaluate'](solution);opt['minimum_augmented_inequality']=float(transformed(eq,iq,scale,keep,hard,z[-1])[1].min());result=export_result(p,solution,opt,preparation,hashes,O,float(z[-1]),scale,keep,hard)
 result['discrete']=dict(segment_ticks=d['ticks'].tolist(),movement_ticks=650,synthetic_quiet_ticks=25,integer_budget_ticks=675,integer_duration_s=13.5,construction_inequalities=20284,actual_rounded_filter_schedule=False,all_intermediate_geometry_checked=False,physical_admission=False);raw_eq,raw_iq=d['p']['evaluate'](solution);result['discrete']['minimum_original_inequality']=float(raw_iq.min());np.savez_compressed(O/'discrete.npz',raw_eq=raw_eq,raw_ineq=raw_iq,tighten=d['tighten'],segment_ticks=d['ticks'],rate_roundoff=d['rate_per_tick_envelope'],target_roundoff=d['target_envelope']);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({q.name:sha(q) for q in O.iterdir() if q.is_file()},indent=2));print(json.dumps(result['optimizer'],indent=2));print(json.dumps({k:v for k,v in result['dense'].items() if k not in ['path','command_nodes']},indent=2))
if __name__=='__main__':main()
