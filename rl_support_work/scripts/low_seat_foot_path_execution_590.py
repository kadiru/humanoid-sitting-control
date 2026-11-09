"""Complete586/588 coupled-foot surrogate plus actual full-original interior construction590."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(**{'method': 'coupled-foot-surrogate-zero-face-IPM', 'resource_address_space_GiB': 40, 'per_lp_worker_wall_cap_s': 120, 'coordinate_system': 'Initialprimary14636 ORIGINAL retained coordinates:10540priorcontrols+4095rootxy/f directions+scalaroriginal78790at14635. Exact586 coupledfoot graph determines16380leg directions;old578graph determines47775auxiliaryoutputs. No actualstate/root/mechanics change. Prior5862730exact6x6inverses inherited;ZERO newinversions.', 'structural_method': '586 exact rational coupled-foot elimination followed by588 separatelyaudited numerical surrogate. ALL265967 inequalities and5462 AFFINE equalities retained. A1891613/E52506coefficients,total1944119;833627fewerthan578but780759morethan572. No performanceclaim. Exact586 projection onlyto572zeroFACE,possiblyempty/proper;not wholeoriginalLP equivalence.', 'initialization': 'ONE cold installed scipy.linprog highs-ipm per genuinelynewinput;presolveTrue,maxiter4096,primal/dual/IPM1e-9,no x0/basis/warmstart/restart/fallback. Initial588input inheritedbyhash;new physicalleg dimensionreduction,not old582restart.', 'primary_equations': 'min t subject588Ahat.v<=bhat,NEW rounded AFFINE Ehat.v=dhat,EXPLICIT14636 projectedfiniteOUTERbounds,t=0equality. ONLY inequalityblock has sufficientREALinner implication. Complete numericinput is APPROXIMATE EQUALITY SURROGATE,NOT certifiedREALinner/equivalent. Defectintervals boundencodingerroronly,not backenddot/residual. Requirestatus0/success/finite14636backendvalues,scalarALREADY0(+/-0);ALLretainedBITWISE,no setting/clip/repair/normalization. Evaluate ALL586 rationalA AND E exactlyatunchangedbackendpoint. Then exactfootgraphs,old578auxgraphonEXACTUNROUNDED31016intermediate,roundEACH64155eliminatedoriginalvalueONCE. Check ALL578rationalA atFINALroundedpoint and578E/bounds/stationarity/gap separately. Fullnumeric588/exact586/exact578/572/570/546/516/FULL490 ALLprimal/equality/bound/requiredstationarity/absolute-gap<=1e-9 inside1e-7. Inherit573analyticlower_t1others0;no newdualproof. NumericalKKT not exactfeasibility/optimality/objectiveerrorbound.', 'interior_equations': 'SAMEcompleteoriginal78791primarydirection computesmodeledtheta/phi andACTUAL(mu+theta)/2ceiling/prediction;neverassumephi0fromt0. PrimaryNEVERmapped. InteriorfreshFULLoriginal490 min-eta ALLoriginalrows/zeroE/originalbounds,no zeroFACEgraph. SamecoldIPMonce;originalbackend78791BITWISE unchanged,backendduals y=-ineq,nu=-eq,upper=-upper,lower=lower;fullKKT.', 'initial_assembly': 'Inherit524initialfullprimary/context/92267intervals and525proof aftercomplete472INITIALX/Y/J+490canonicalaux/drop/trust identity. Bind517547559571573579581587589 proofhashes. No initialassembly/root/oldgraph/inversion/encoding/proof arithmetic replay.', 'changed_input': 'Genuinelynewstate/trust/ACTUALceiling usesfreshFULL490 andindependentfullassembly/intervalproofBEFORELP inside45sTOTAL. ConservativeFULLoriginalvariables/rows,NOoldstate reductions reused. NewprimaryaddsONLYzeroFACEwithindependentnewnative/scalar/RHS/copyproof;newinterioruntouchedFULL. No sequenceofnonzeroepigraphfaces.', 'clock': 'First45sTOTAL fromMAINentry BEFOREscientificimports/globalhashes/loads/context/preparation;laterbeforeparentnewassembly. SharedLinuxmonotonicorigin passedchild. Allpreparation/backend/exactpoint/compositelift/allKKT charged,onlyremainingbackendallowancepassedonce. Checksbetweenoperations,overshootreportednotextension.40GiBchildAS/120sparentchildwall.', 'budget_rationale': '24major48actualLP120maps5dyadics1..1/16;trust1..1/32guards2^-20,halveonlyfinite/nonlinearreject,double<=1onlyratio>.75,accept>=.1sameeffectiveencodedfeaturedelta. Firstabsent/nonfinite/uncertified/infeasible/deadline TERMINATES ENTIREconstruction,no retry.', 'finite_policy': 'Neveruseoriginalinfiniteboundswith588compensation. RetainedbackendvaluesBITWISE;exactunroundedlegsfeedoldauxgraph;ALL64155eliminatedcoordsroundONCE. No universalbinary64graphextension. Exactunscaledintervaleta>0beforeimmutable490encoder,newXfreshmechanics/J,no15root. Allsignedrows/7units1e-13/normal1e-13/hover1e-8/geometry/filter/runtime/72retention/PPOgates unchanged.', 'partial_return': 'Persist NEW pointresidual/graphvalues as processed; incomplete work never impliespassedcertificate. Independent591 checks savedpartialnewpoint arithmetic ifterminated,doesnot redo completedoldproofs or inferunsavedwork.', 'limits': 'Fewercolumns not speed/conditioning/convergenceevidence. Surrogate/zeroFACE failure not originalglobalinfeasibility. No internal/rejectedvectors,physicalpromotion,guard/tolerance/budgetchanges.', 'physical_admission': False})
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
 if folder==D/'foot_path_590/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 write(child/'clock_origin.json',dict(parent_monotonic_origin=origin,parent_preparation_elapsed_s=start-origin,all_new_preparation_charged=True))
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_foot_path_lp_590.py')),str(child),repr(origin)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='coupled-foot-surrogate-zero-face-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=bool(meta.get('coordinate_reconstruction_attempts',0)))
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['full']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=not bool(meta.get('coordinate_reconstruction_attempts',0)),retained_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
