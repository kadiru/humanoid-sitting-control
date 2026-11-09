"""Complete bounded Schur/HSD witness construction in original torque-projected coordinates."""
from low_seat_torque_execution_v2_524 import *
from low_seat_schur_solver_532 import SPEC as LP_SPEC
SPEC=dict(SPEC)
SPEC.pop('staged_recovery',None)
SPEC.update(scope=__doc__,method='hsd-schur-dr',lp_subsolver=LP_SPEC,
 rationale='Distinct fixed homogeneous self-dual splitting with state-local matrix-free Schur/PCG resolvents; original torque projection and all physical gates retained. No HiGHS retry or old cut continuation.',
 budget_rationale='24major/48actual HSD LP invocations/120maps,45s TOTAL per child algorithm,40GiB address space/120sparent.4096DR/256CG per K/16384CGtotal; one method, no retry.',
 numerical_limits='Unscaled numerical primal/dual/gap1e-9, independent original LP1e-7. All physical and finite tolerances unchanged.',
 initial_assembly='Exact preserved524 initial context and full primary assembly plus525 independent proof inherited by hash; zero initial reassembly/normalization. This is a new algorithm from472INITIAL, not524 continuation or recovery.',
 bounded_trace='Only factors, last attempted K equation, last completed DR step and scalar ledgers. Immediate nonfinite decoded candidate termination. No all-iteration vector accumulation.',
 failure_policy='First absent/uncertified projected LP or original backsubstitution/witness failure terminates entire run. No timeout trust series, internal-state selection or method fallback.',physical_admission=False)
def solve_projected(folder,d,b,s,projections,ceiling=None):
 
 if folder==D/'schur_path_532/major_00/primary':
  assert not (folder/'source').exists() and not (folder/'projected').exists();p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 role='primary' if ceiling is None else 'witness';source,z,ledger,scope=projections.get(folder,p,role);child=folder/'projected';started=time.monotonic();timed=False
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_schur_lp_532.py')),str(child)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 called=(child/'hsd_started.json').exists();meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0,crossover_iterations=0)
 meta.update(actual_lp_call=called,worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-started,method='hsd-schur-dr',projection=scope,projected_candidate=meta['candidate'],backsubstitution_attempted=False);meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  q=np.load(child/'witness.npy');pv=violation(z,q);meta.update(projected_linear_violation=pv)
  if pv<=1e-7 and np.all(np.isfinite(q)):
   meta['backsubstitution_attempted']=True;w=reconstruct(source,ledger,q,folder)
   if w is not None:
    v=violation(p,w);sv=violation(source,w);np.savez_compressed(folder/'full_residuals.npz',original=p['matrix']@w-p['rhs'],source=source['matrix']@w-source['rhs'],projected=z['matrix']@q-z['rhs']);meta.update(linear_violation=v,source_linear_violation=sv,linear_pass=bool(max(pv,v,sv)<=1e-7),objective=float(p['cost']@w),reconstruction_unmodified=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
