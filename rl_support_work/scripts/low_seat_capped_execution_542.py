"""Complete bounded native capped-maximum PDHG path construction, all auxiliary torques retained."""
from low_seat_lifted_execution_492 import *
from low_seat_capped_solver_542 import SPEC as LP_SPEC
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='capped-maximum-native-PDHG',lp_subsolver=LP_SPEC,resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='ALL78790 original native directions including21840 auxiliary torques; only scalar epigraph/eta handled analytically. No522 projection/backsubstitution,504 transform,532 Schur/HSD or old internal vector.',
 initial_assembly='Preserved524 complete primary/context/intervals and525 proof inherited after EXACT472 initial state/canonical490/drop/trust and complete input identity. No initial assembly/root or old metric/proof repeated.',
 equations='Frozen540 PDHG with fixed exact-certified diagonal steps, theta1, zero initialization, finite weighted proximal screens and first eligible once-rounded exact scalar. No tuning, averaging, restart or fallback.',
 native_witness='Returned native values stay BITWISE unchanged. Source row duals map to full original row positions, with zeros on omitted exact-implied rows. Full source AND FULL490 numerical primal/equality/bound/stationarity/gap checks<=1e-9, stricter than unchanged full linear1e-7, before return. Scalar rounding slack is disclosed, not repaired.',
 changed_input='A different actual state/trust/ACTUAL witness ceiling assembles complete original system, exact-original quotient with independent implication proof, capped split and independent copy proof, exact metric and independent norm proof before iteration. These are new input calculations. Unsupported native domain/equality rejected.',
 budget_rationale='24major/48actual LP invocations/120maps/5shared dyadics;45s TOTAL includes child input load/validation/new preparation/proof/PDHG/scalar/full KKT;40GiB/120sparent.4096PDHG each. First failure ends entire run; no timeout trust series.',
 finite_policy='Primary is never mapped; complete original direction supplies primary modeled phi, actual interior ceiling and prediction. Dyadics use full original direction then exact unscaled interval eta>0 before immutable490 finite encoding. Fresh mechanics/J only for changedX; no root postprocessing.',
 finite_limits='Finite proximal screens are not exact maps or a summable-error convergence theorem. Exact-real norm/reference theorem is not a45s performance or physical feasibility guarantee.',physical_admission=False)
def save_problem(folder,p):
 folder.mkdir(exist_ok=False)
 for name in ['matrix','equality']:sparse.save_npz(folder/(name+'.npz'),p[name])
 np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']})
def load_problem(folder):
 return dict(matrix=sparse.load_npz(folder/'matrix.npz').tocsr(),equality=sparse.load_npz(folder/'equality.npz').tocsr(),**dict(np.load(folder/'arrays.npz')))
def violation(p,w):
 if not np.all(np.isfinite(w)):return float('inf')
 return max(0.,float((p['matrix']@w-p['rhs']).max()),float(abs(p['equality']@w-p['equality_rhs']).max()),float((p['bounds'][:,0]-w).max()),float((w-p['bounds'][:,1]).max()))
class NativeSystems:
 def __init__(self):self.inherited_systems=0;self.new_systems=0;self.new_proof_rows=0

def solve_native(folder,d,b,s,systems,ceiling=None):
 if folder==D/'capped_path_542/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_capped_lp_542.py')),str(child)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'pdhg_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='capped-maximum-native-PDHG',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'solve/witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['source']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
