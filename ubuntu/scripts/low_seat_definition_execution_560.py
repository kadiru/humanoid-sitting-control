"""Full original-coordinate witness construction through the exact558 definition extension."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='exact-definition-extended-IPM',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='All78791 ORIGINAL coordinates retained first, scalar at78790, plus57329 free zero-cost defining variables:136120 columns on inherited558. Original546 native bounds remain unbounded;552 finite bounds are not used.',
 structural_method='Exact558/559 common local-mechanics expression sharing: paired native a.x and -a.x become +z and -z, plus mandatory a.x-z=0. No original inequality/variable removed. New nonzero E58695x136120/773732nnz. Primary inequality402487x136120/730937nnz. Exact real projection/objective equivalence; no universal binary64 extension or tolerance-set equivalence.',
 initialization='ONE cold installed scipy linprog highs-ipm call per input; no x0,basis,warmstart,restart,fallback. PresolveTrue,maxiter4096,primal/dual/IPMoptimality1e-9. Same fixed backend; genuine mathematical change is558 sparse extension, not method/parameter retry.',
 equations='min c.w subject extended Aw<=b,NONZERO extended Ew=d,l<=w<=u. Require status0 finite extended vector; preserve original78791 prefix bitwise, discard ONLY appended definitions. Extended inequality duals are source546 duals unchanged; preserve original E/bound dual prefixes, discard ONLY appended definition duals. nu=yplus-yminus follows exact z-stationarity, but finite errors may amplify; independently check all FOUR extended/546/516/FULL490 primal/equality/bound/stationarity/absolute-gap<=1e-9. Source/full row duals zero on inherited implied rows. New free bound duals exactlyzero. No rounding/repair/reconstruction.',
 initial_assembly='Inherit524 complete initial primary/context/92267intervals and525 proof after472INITIAL/canonical490/drop/trust/source identity. Inherit516/517,546/547,558/559 only by full source/domain/hash identity. No completed proof, pair decision, definition construction or old fixture repeated.',
 changed_input='Fresh complete original assembly, quotient, box support and definition extension plus independent exact input proofs for changed actual state/trust/ACTUAL interior ceiling. All preparation charged to45s clock, including parent assembly/context before child imports. Old illustrative reference ceiling never actual result.',
 native_witness='Original78791 primal values exactly the backend prefix, no native/scalar reconstruction. Exact algebra does not bound floating original residuals; full checks mandatory. Numerical KKT does not imply exact rational feasibility/optimality or objective error<=1e-9.',
 budget_rationale='24major/48actualLP/120maps/5dyadics,45sTOTAL charged from parent preparation start and before child scientific imports. Includes new assembly/input/reduction/proofs/backend/allKKT. Remaining allowance passed once; checks between operations with actual overshoot reported.40GiB/120sparent childcap. First absent/nonfinite/uncertified/deadline terminal entire run; no timeout trust series.',
 finite_policy='SAME original full direction sets primary modeled phi,ACTUAL ceiling(mu+theta)/2,prediction. Primary never mapped. Exact unscaled interval eta>0 before immutable490 finite encoder. NewX fresh mechanics/J, no15-root correction; all actual signed/unit/normal/hover/geometry/filter/runtime/PPO gates unchanged.',
 limits='More variables/equalities and fewer coefficients are structural counts, not performance/conditioning/feasibility evidence. No physical controls or authority added by algebraic definitions. All original/inward rows,BOTHconsistency signs/frozen units/domain/physical guards remain.',physical_admission=False)
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
 origin=getattr(systems,'pending_start',None)
 if origin is None:origin=time.monotonic()
 elif hasattr(systems,'pending_start'):del systems.pending_start
 if folder==D/'definition_path_560/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 write(child/'clock_origin.json',dict(parent_monotonic_origin=origin,parent_preparation_elapsed_s=start-origin,all_new_preparation_charged=True))
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_definition_lp_560.py')),str(child),repr(origin)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='exact-definition-extended-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['source']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
