"""Full native witness construction using the exact546 box-support row reduction."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='exact-box-native-IPM',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='All78790 original native directions plus original epigraph/eta,78791columns. No torque/scalar elimination, mixed basis, clipping or reconstruction.',
 structural_method='New546 exact multirow box-support reduction, independently proved547, after inherited516/517 exact original ray quotient. Every unary premise retained. Reduced primary402487rows/2049085nnz; all full source criteria remain exactly implied. This is the mathematical change; the backend is existing fixed HiGHS-IPM, not a newly claimed solver method.',
 initialization='ONE cold scipy linprog highs-ipm call per input, no x0/basis/old internal state. Installed SciPy/HiGHS implementation and version bound at source freeze. PresolveTrue, maxiter4096, primal/dual/ipm_optimality1e-9. No method/parameter fallback, adaptive metric or restart.',
 equations='min c.w subject Aw<=b,Ew=d,l<=w<=u. Return only status0 finite unedited native+scalar w and nonnegative row/bound duals with full numerical primal/dual stationarity and absolute duality gap<=1e-9 in retained546, source516 and FULL490 systems. Duals lift by zero on exactly implied omitted rows; no coefficient rescaling.',
 initial_assembly='Inherit524 complete initial primary/context/intervals and525 independent proof after472INITIAL/canonical490/drop/trust/source identities. Inherit546/547 reference subset/certificates after ALL full/source/reduced/domain/hash checks; no old assembly/quotient/box proof repeated.',
 changed_input='Changed actual state, trust or ACTUAL witness ceiling gets fresh complete original assembly, new original quotient and box-support certificates with independent exact proofs BEFORE any iteration, within child45s clock. No use of old illustrative witness ceiling.',
 native_witness='ALL78791 coordinates unchanged bitwise. No midpoint/epigraph reconstruction. Source and FULL490 checks remain mandatory since exact-real implications do not bound floating residual amplification. No exact rational feasibility/optimality claim from numerical KKT.',
 budget_rationale='24major/48actual LP/120maps(max5dyadics1..1/16),45sTOTAL per child including startup,input load/identity,new reduction+proof,backend and fullKKT. Remaining wall allowance passed once to backend; deadline checked between operations and any overshoot reported.40GiB/120sparent. First absence/failure/deadline terminates entire run; no trust series after timeout.',
 finite_policy='Same complete original direction gives primary modeled phi,ACTUAL interior ceiling(mu+theta)/2,prediction. Primary never mapped. Full interior direction dyadics require exact original unscaled interval eta>0 before immutable490 encoder. ChangedX fresh model/J, no15-root correction.',
 limits='Structural row reduction is not measured solver speed/conditioning improvement or global feasibility. All original/inward/BOTH consistency signs/frozen units/physical guards and actual geometry/filter/runtime/PPO gates unchanged.',physical_admission=False)
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
 if folder==D/'box_path_548/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_box_lp_548.py')),str(child)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='exact-box-native-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['source']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
