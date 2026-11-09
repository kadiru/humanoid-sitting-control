"""Full original-coordinate witness construction through the exact564 native dual-balance formulation."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='exact-native-dual-balance-IPM',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='Optimizer primal p=(y,beta,alpha)>=0 has m+2 variables. Original78791 native+scalar coordinates,including21840tau auxiliaries,are equality-marginal slots; originalscalar slot78790. They are NOT optimizer-primal columns.',
 structural_method='Exact564/565 native LP dualization using saved wholeclock/command/local-mechanics transpose; B=[A.T,+e78790,-e78790],h=-c,q=[b,U,-L]. Original546 domain,ALLrow multipliers/BOTHconsistency signs preserved. No552 finitebounds,558 definitions,522 torqueprojection or native coordinate substitution.',
 initialization='ONE cold installed scipy linprog highs-ipm per new input,NOx0,basis,warmstart,restart,fallback. Fixed presolveTrue,maxiter4096 and primal/dual/IPMoptimality1e-9. Mathematical change is564 dualization,not backend/parameter retry.',
 equations='min q.p subject B.p=h,p>=0; no inequality matrix. Require status0/success/finite p and equality marginals. ORIGINALw=BITWISE COPY scipy eqlin.marginals,NO NEGATION,notresult.x. Original y=p[:m],upperScalar=p[m],lowerScalar=p[m+1]; nativebound/oldE duals0. Newnu=-w,newlower=lower.marginals,newupper=-upper.marginals=0 at infinity. Newbalance is originalstationarity; newstationarity encodesoriginalslacks; newgap equalsoriginalsignedgap. Independently all FOUR D/546/516/FULL490 primal/equality/bound/stationarity/absolute-gap<=1e-9,nonnegative finite duals/infiniteboundduals0. No candidate reconstruction/rounding/clipping/repair.',
 initial_assembly='Inherit524 fullinitial/context/92267intervals and525proof after472INITIAL/canonical490/drop/trust/source identity. Inherit516/517,546/547,564/565 onlybyfullsource/domain/hash identity; no completed transpose/identity/proof/fixture repeated.565 originalaudit serialization failure and stagedwitness-only recovery preserved.',
 changed_input='Fresh fulloriginal assembly,quotient,boxsupport,newdualconstruction and independent exact proofs BEFORE LP for changedactualstate/trust/ACTUALceiling. All preparation inside45sTOTAL fromparentprep beforechildimports. Oldillustrativewitnessceiling neveractualresult.',
 native_witness='All78791 originalvalues bitwise backend equality-marginals. Independent alloriginal residuals required; floating association may differ from exact coefficient identities. NumericalKKT is not exact rational feasibility/optimality/objectiveerror<=1e-9.',
 budget_rationale='24major/48actualLP/120maps/5dyadics,45sTOTAL fromparentprep beforeimports,includesnewassembly/proofs/backend/allKKT. Onlyremaining allowance passedonce; actualovershoot reported.40GiB/120sparent. Firstabsent/nonfinite/uncertified/deadline terminalENTIRErun; no timeouttrustseries.',
 finite_policy='SAME completeoriginaldirection primaryphi/ACTUAL(mu+theta)/2ceiling/prediction; primaryNEVERmapped. Exactoriginalunscaledintervaleta>0 beforeimmutable490encoder; newXfreshmechanics/J,no15rootdrift. Allactualsigned/7unitclosure/normal/hover/geometry/filter/runtime/PPOgates unchanged.',
 limits='Structural dimensions do not establish speed/conditioning/feasibilitygain. Auxiliary torques remain modeloutputs,not actuatorauthority. No physical criterion removed. Originalreal domain retained; no universal finite solver-return or tolerance-equivalence claim.',
 marginal_reference='https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.linprog-highs-ipm.html ; sensitivity/sign convention only,installed1.17 runtime files independently frozen; no performance theorem imported.',physical_admission=False)
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
 if folder==D/'dual_path_566/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 write(child/'clock_origin.json',dict(parent_monotonic_origin=origin,parent_preparation_elapsed_s=start-origin,all_new_preparation_charged=True))
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_dual_lp_566.py')),str(child),repr(origin)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='exact-native-dual-balance-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['source']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
