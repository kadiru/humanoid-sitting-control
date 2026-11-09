"""Full native witness construction using552 finite-binary64 bounds and an explicit original dual lift."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='finite-bound-native-IPM',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='ALL78790 native directions plus original scalar remain78791. No primal reconstruction, clipping or repair.',
 structural_method='552 singleton absorption into directed binary64 coordinate bounds, inheriting546/547 exact box support and516/517 quotient. Exact finite-binary64 feasible vectors preserved; new real polytope is an inward subset, not generally equal. Numerical tolerance sets and original real optima need not agree.',
 initialization='One cold installed scipy linprog highs-ipm, presolveTrue,maxiter4096,primal/dual/ipm_optimality1e-9. No x0/basis/warmstart/restart/fallback/backend or parameter sweep. Structural domain representation is new; IPM is not newly invented.',
 equations='min c.w subject mixed A.w<=b,E.w=d,l64<=w<=u64. Return only finite status0 unedited w passing numerical KKT<=1e-9 in552,546,516,FULL490.',
 dual_lift='For selected singleton a*w_j<=b with endpoint q=b/a, transfer nonnegative bound mu to exact mu/abs(a),round multiplier ONCE binary64; direct-bound owner copies mu. Exact coefficient stationarity preserved before rounding. Original dual value=new dual value-delta, delta=sum(muL*(l64-L)+muU*(U-u64))>=0. Original signed gap therefore grows by delta; save exact penalty and each multiplier rounding coefficient/RHS error. FULL original KKT must pass; no real-LP equivalence or objective-error claim.',
 initial_assembly='Inherit524 complete primary/context/intervals and525proof after exact472INITIAL/490canonical/drop/trust identity; reference552/553/546/547/516/517 inherited after full source/domain/hash checks. No old calculation repeated.',
 changed_input='New actual state/trust/ACTUAL ceiling needs new full assembly,quotient,box endpoints/certificates,binary64 bounds and independent exact input proofs before LP within45s. No illustrative reference-witness ceiling reuse.',
 native_witness='All78791 returned coordinates bitwise unchanged. Source dual algebra alone is lifted; original inequalities/equalities/bounds/stationarity/absolute gap<=1e-9 in all four systems mandatory. Exact implications do not bound floating residual amplification.',
 budget_rationale='24major/48LP/120maps,max5dyadics1..1/16;45sTOTAL child before imports/loads,includinginput proofs,backend,exact dual lift and fullKKT;40GiB/120sparent. Remaining wall allowance passed once,operation overshoot reported. First absent/uncertified/nonfinite/deadline terminal entire run, no timeout trust series.',
 finite_policy='Same original direction gives modeledphi,ACTUAL(mu+theta)/2 ceiling,prediction;primarynevermapped. Exact original unscaled intervaleta>0 beforeimmutable490encoder;newXfreshmechanics/J,no15rootdrift. Trust1..1/32/2^-20guards/acceptance.1 unchanged.',
 limits='No performance,conditioning,globalfeasibility or exact-real optimality claim. All physical/unit/geometry/filter/runtime/retention/PPO gates unchanged.',physical_admission=False)
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
 if folder==D/'float_path_554/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_float_lp_554.py')),str(child)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='finite-bound-native-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['source']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
