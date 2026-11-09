"""Prepare one new feasibility-first complete nominal-path formulation, zero optimization/physics."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from low_seat_reduced_problem_365 import make_problem
from low_seat_feasibility_rows_365 import rows,transformed
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'feasibility_preparation_365'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);p=make_problem();seed=np.load(D/'reduced_path_v2_361/final.npz');initial=np.load(D/'reduced_path_v2_361/initial.npz');x=seed['x'].copy();eq,iq=p['evaluate'](x);E,I=p['evaluate'](x,True);free=p['free'];scale,keep,groups,proof=rows(p)
 errors=dict(eq=float(abs(eq-seed['eq']).max()),ineq=float(abs(iq-seed['ineq']).max()),full_coordinates=float(abs(p['last']['X']-seed['full_coordinates']).max()),local_outputs=float(abs(p['last']['Y']-seed['local_outputs']).max()))
 assert max(errors.values())<1e-10,errors
 for name,pname in [('null','null'),('base_weights','wbase'),('counts','counts'),('rates','rates'),('command_lo','command_lo'),('command_hi','command_hi'),('lower','lower'),('upper','upper')]:assert np.array_equal(initial[name],p[pname]),name
 assert np.array_equal(initial['free'],free)
 bounds=np.array(p['bounds']);assert np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]);assert np.array_equal(x[np.setdiff1d(np.arange(len(x)),free)],p['x0'][np.setdiff1d(np.arange(len(x)),free)])
 removed=np.flatnonzero(~keep);assert np.array_equal(I[removed],np.zeros_like(I[removed]));assert np.array_equal(iq[removed],np.array([v['value'] for v in sorted(proof,key=lambda a:a['index'])]))
 rng=np.random.default_rng(365);direction=np.zeros(len(x));direction[free]=rng.normal(size=len(free));direction/=np.linalg.norm(direction);h=1e-5
 ep,ip=p['evaluate'](x+h*direction);em,im=p['evaluate'](x-h*direction);eqfd=(ep-em)/(2*h);iqfd=(ip-im)/(2*h);eqan=E@direction;iqan=I@direction;err=float(max(abs(eqfd-eqan).max(),abs(iqfd-iqan).max()));assert err<2e-6,err
 slack=max(0.,float(np.max(-iq[keep]/scale[keep])),float(abs(eq).max()))+1e-6;aug=transformed(eq,iq,scale,keep,slack);assert aug.min()>0
 # Exact algebraic reconstruction is checked independently in366. Mixed-unit raw residuals are never ranked as a physical quantity.
 np.savez_compressed(O/'fixture.npz',x=x,free=free,bounds=bounds,eq=eq,ineq=iq,E=E[:,free],I=I[:,free],direction=direction,h=h,eq_fd=eqfd,ineq_fd=iqfd,eq_analytic=eqan,ineq_analytic=iqan,scale=scale,keep=keep,slack=slack,augmented=aug,null=p['null'],base_weights=p['wbase'],effort=p['effort'],rates=p['rates'])
 paths=[Path(__file__).resolve(),Path('scripts/low_seat_reduced_problem_365.py').resolve(),Path('scripts/low_seat_feasibility_rows_365.py').resolve(),Path('scripts/build_low_seat_reduced_path_v2_361.py').resolve(),D/'reduced_path_v2_361/final.npz',D/'reduced_path_v2_361/initial.npz',D/'reduced_path_v2_361/results.json',D/'reduced_path_audit_362/independent_audit.json']
 hashes={str(a):sha(a) for a in paths};old=json.loads((D/'reduced_path_v2_361/results.json').read_text())['source_input_sha256']
 for a,hsh in old.items():assert sha(a)==hsh;hashes[a]=hsh
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,source_input_sha256=hashes,scope=__doc__,reconstruction_errors=errors,raw_directional_error=err,unchanged_raw_derivative_bound=2e-6,groups=groups,removed_structural_rows=proof,all_original_inequalities=19760,solver_retained_original_inequalities=int(keep.sum()),original_equalities=1,augmented_inequalities=len(aug),free_original_variables=len(free),slack_variables=1,initial_slack=float(slack),initial_augmented_minimum=float(aug.min()),objective='minimize one dimensionless worst original row violation slack; no time objective or secondary solve',original_admission_constraints_unchanged=True,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,nominal_optimizer_runs=0)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({a.name:sha(a) for a in O.iterdir() if a.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['source_input_sha256','removed_structural_rows','groups']},indent=2))
if __name__=='__main__':main()
