"""ONE bounded feasibility-first complete nominal path. No physics, learning or admission.
Positive epigraph slack represents remaining original infeasibility, never permission.
"""
from pathlib import Path
import json,hashlib,datetime,time
import numpy as np
from scipy.optimize import minimize
from low_seat_reduced_problem_365 import make_problem
from low_seat_feasibility_rows_365 import rows,transformed
from export_low_seat_feasibility_367 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'feasibility_path_367'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);prep=json.loads((D/'feasibility_preparation_365/preparation.json').read_text());audit=json.loads((D/'feasibility_row_audit_366/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
 for record in [prep['source_input_sha256'],audit['source_sha256']]:
  for path,h in record.items():assert sha(path)==h,path
 p=make_problem();f=np.load(D/'feasibility_preparation_365/fixture.npz');x0=f['x'].copy();free=p['free'];scale,keep,groups,proof=rows(p);assert np.array_equal(scale,f['scale']) and np.array_equal(keep,f['keep']);s0=float(f['slack']);z0=np.r_[x0[free],s0];history=[];start=time.monotonic();last_valid=z0.copy()
 def expand(z):
  x=x0.copy();x[free]=z[:-1];return x
 def fun(z):
  eq,iq=p['evaluate'](expand(z));return transformed(eq,iq,scale,keep,z[-1])
 def jac(z):
  E,I=p['evaluate'](expand(z),True);A=np.r_[I[keep][:,free]/scale[keep,None],E[:,free],-E[:,free]];return np.c_[A,np.ones(len(A))]
 def callback(z):
  nonlocal last_valid
  x=expand(z);eq,iq=p['evaluate'](x);last_valid=z.copy();row=dict(iteration=len(history)+1,elapsed_s=time.monotonic()-start,slack=float(z[-1]),time_s=float(.5+x[798:].sum()),normalized_hover_residual=float(eq[0]),minimum_raw_inequality=float(iq.min()),maximum_scaled_original_violation=float(max(0.,np.max(-iq[keep]/scale[keep]),np.max(abs(eq)))),minimum_augmented_inequality=float(fun(z).min()),maximum_kinematic_residual=float(abs(p['last']['Y'][:103,:14]).max()));history.append(row);np.save(O/'last_iterate.npy',z);(O/'iterations.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 originalinitial=np.load(D/'reduced_path_v2_361/initial.npz');np.savez_compressed(O/'initial.npz',**{k:originalinitial[k] for k in originalinitial.files},starting_final_x=x0,starting_slack=s0,scale=scale,keep=keep)
 hashpaths=[Path(__file__).resolve(),Path('scripts/export_low_seat_feasibility_367.py').resolve(),Path('scripts/audit_low_seat_feasibility_path_368.py').resolve(),D/'feasibility_preparation_365/preparation.json',D/'feasibility_preparation_365/fixture.npz',D/'feasibility_row_audit_366/independent_audit.json'];hashes=dict(prep['source_input_sha256']);hashes.update({str(q):sha(q) for q in hashpaths})
 preparation=dict(scope=__doc__,source_input_sha256=hashes,optimizer='SLSQP',maximum_iterations=30,maximum_optimizer_runs=1,objective='dimensionless epigraph slack only',row_scale_groups=groups,removed_constant_rows=proof,all_original_inequalities=19760,retained_solver_original_rows=int(keep.sum()),original_equality_count=1,free_path_variables=694,epigraph_slacks=1,augmented_inequalities=int(keep.sum()+2),full_geometry_is_mandatory_final_screen=True,unchanged_criteria=True,physical_admission=False)
 (O/'revision.json').write_text(json.dumps(preparation,indent=2));gradient=np.r_[np.zeros(len(free)),1.];assert fun(z0).min()>0
 try:
  fit=minimize(lambda z:float(z[-1]),z0,jac=lambda z:gradient,method='SLSQP',bounds=[p['bounds'][k] for k in free]+[(0,None)],constraints=[dict(type='ineq',fun=fun,jac=jac)],callback=callback,options=dict(maxiter=30,ftol=1e-10));z=fit.x;opt=dict(success=bool(fit.success),message=fit.message,iterations=int(fit.nit),evaluations=int(fit.nfev))
 except RuntimeError as ex:
  z=last_valid;opt=dict(success=False,message='terminated on dependent-coordinate domain failure; last valid callback retained',iterations=len(history),evaluations=None,domain_error=str(ex))
 solution=expand(z);eq,iq=p['evaluate'](solution);opt['minimum_augmented_inequality']=float(transformed(eq,iq,scale,keep,z[-1]).min());result=export_result(p,solution,opt,preparation,hashes,O,float(z[-1]),scale,keep)
 (O/'SHA256.json').write_text(json.dumps({q.name:sha(q) for q in O.iterdir() if q.is_file()},indent=2));print(json.dumps(result['optimizer'],indent=2));print(json.dumps({k:v for k,v in result['dense'].items() if k not in ['path','command_nodes']},indent=2))
if __name__=='__main__':main()
