"""Complete primary572 zero-face plus actual full-original interior construction574."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='primary-zero-face-native-IPM',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='ALL78791 original native+scalar optimizer primal coordinates. Scalar78790. No coordinate substitution/reconstruction/repair;21840tau auxiliaries remain modeloutputs,not controls.',
 structural_method='Primary572 exposed objective-zero face t=0; ALL68250 relaxed signed consistency pairs become AFFINE equalities with copied originalRHS. Potentially empty/proper original feasible-set subset,NOTwholeLP equivalence. No original optimum-zero or feasibility claim.',
 initialization='ONE cold installed scipy linprog highs-ipm per genuinely new input; no x0/basis/warmstart/restart/fallback. FixedpresolveTrue,maxiter4096,primal/dual/IPM1e-9. Initialprepared572 only; mathematical change is zero-face restriction,not backendretry.',
 primary_equations='min t subject retained572Aw<=b,NONZEROAFFINEEw=d,originalbounds,t=0. Require status0/success/finite result.x and scalar alreadyEXACTzero(+/-0),all78791values BITWISE unchanged. Inherit573 analytic lowerbound dual(lower_t1,otherduals0). Independently FULLface/570/546/516/490 primal/eq/bound/stationarity/abs-gap<=1e-9 inside1e-7. No scalar set/clip/round/normalize/repair. NumericKKT NOTexactfeasibility/optimality/objectiveerrorbound.',
 interior_equations='Only after eligible primary, SAMEfulloriginaldirection gives computedtheta/phi and ACTUAL(mu+theta)/2ceiling. No modeledphi0 assumption. Interior is min -eta with ALLfreshFULL490 original rows,E,bounds; no zero-eta face and no old illustrativeceiling. ColdsameIPM once. StandardSciPy signs y=-ineqmarginal,nu=-eqmarginal,upper=-uppermarginal,lower=lowermarginal. FULLoriginalKKT<=1e-9.',
 initial_assembly='Inherit524 fullprimary/context/92267intervals and525proof afterexact472INITIAL/490canonicalaux/drop/trust identity. Bind570/571,546/547,516/517,572/573 and559pair proof byhash. No initialassembly/face/algebra/analyticdual proof replay.',
 changed_input='For a genuinelychanged state/trust or ACTUALinteriorceiling, assemble FULL490 and independently verify completeassembly/intervals BEFORELP. Conservatively retainALLoriginalrows: NOquotient/box/coupled reduction is applied to changed inputs, hence no old-state reduction proof is reused or new endpoint reduction needed. Newprimary alone constructsface directlyfromFULLsignedpairs with independent exactnative/scalar/RHS/copyproof. Newinterior uses untouchedFULLsystem. Allwithin45sTOTAL.',
 clock='First45sLP clock starts at mainprocess entry BEFOREscientificimports/globalinputchecks/loads/context/copies,then passes sharedLinuxmonotonicorigin tochild. Later45s clocksstartbeforeparentnewcontext/assembly. Everynewinputproof/backend/fullKKTcharged. Onlyremainingallowancepassedonce;overshootreported. Parentchildwall120s,child40GiB.',
 budget_rationale='24major48actualLP120maps5dyadics1..1/16; trust1..1/32,guards2^-20,halveonlyfinite/nonlinearreject,double<=1onlyratio>.75,accept>=.1sameeffectiveencodedfeaturedelta. Firstabsent/nonfinite/uncertified/infeasible/deadline terminalENTIRErun; no timeouttrustseries/nonzero-epigraphface sequence.',
 finite_policy='PrimaryNEVERmapped. Sameoriginaldirection sets modeledphi/actualceiling/prediction. Interior originaldirectiondyadics then exactunscaledintervaleta>0 beforeimmutable490encoder;newXfreshmechanics/J,no15root. Allsignedrows/7units1e-13/normal1e-13/hover1e-8/geometry/filter/runtime/72retention/PPOgatesunchanged.',
 limits='Face may be empty though originalLP permits positiveobjective. FailureNOToriginal/physicalimpossibility. Moreequalities/fewerstoredcoefficientsNOTspeed/conditioning/deadline guarantee. Analytic dual provesonlylowerbound0 untilprimalpasses. No physical/PPOpromotion.',physical_admission=False)
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
 if folder==D/'zero_face_path_574/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 write(child/'clock_origin.json',dict(parent_monotonic_origin=origin,parent_preparation_elapsed_s=start-origin,all_new_preparation_charged=True))
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_zero_face_lp_574.py')),str(child),repr(origin)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='primary-zero-face-native-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['full']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
