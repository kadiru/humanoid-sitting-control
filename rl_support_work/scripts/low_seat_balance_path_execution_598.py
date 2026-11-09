"""Complete594/596 condensed inner zero-face plus actual full-original interior construction598."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(**{'method': 'condensed-balance-inner-zero-face-IPM', 'resource_address_space_GiB': 40, 'per_lp_worker_wall_cap_s': 120, 'coordinate_system': 'Initialprimary10541 ORIGINAL retained coordinates:10540priorcontrols+scalaroriginal78790at10540. Exact594graph determines4095rootxy/f directions,586graph16380legs,578graph47775auxiliaryoutputs. All20475physicaldirections remainuniquelydetermined andmustpassfullchecks. Savedcoefficientgraphs NOTnonlinear15-root/IK/drift/force-onlysolve. Prior5941365exact3x3+5862730exact6x6inverses inherited;ZEROnewinversion.', 'structural_method': '594 exact moment/gap condensation of586 followedby596 independentlycertified binary64input. ALL265967inequalities and1367equalities retained. A6102946/E1coefficients,total6102947 INCREASE4158828over586/4939587over572;no speed/conditioningclaim. Exact594REALprojection onlyto572zeroFACE,possiblyempty/proper,notwholeLP equivalence. This eliminates4095additionalphysicaldirectioncols/equalities,not590restart/same-matrixreencoding.', 'initialization': 'ONEcold installedscipy.linprog highs-ipm per genuinelynewinput;presolveTrue,maxiter4096,primal/dual/IPM1e-9,no x0/basis/warmstart/restart/fallback. New596input inheritedbyhash with594595/596597 proofaudits,not replayed.', 'primary_equations': 'min t subject596Ahat.v<=bhat,EXACTE.v=0,EXPLICIT10541projectedfiniteOUTERbounds,t=0equality. Complete numericREALinput sufficientINNERsubsetof594zeroFACEinsidebox,possiblyempty/proper;NOTequivalence/finite/tolerance/primalcertificate. Status0/success/finite10541backendv/scalarALREADY0 required. ALLretainedBITWISE/no setting/clip/repair/normalization. EvaluateALL594exactrationalA/E atunchangedv,thenEXACT594rootxy/f->586legs atUNROUNDED14636->578aux atUNROUNDED31016;roundEACH68250originaleliminatedcoordinateONCE. ALL586and578rationalA/E/fullbounds atFINALonce-roundedoriginalprojections mandatory. Numeric596/exact594/586/578/572/570/546/516/FULL490 ALLNINEsystems primal/E/bound/requiredstationarity/absolute-gap<=1e-9inside1e-7.573analyticlower_t1others0 inherited,no newdualproof. NumericalKKT NOTexactfeasibility/optimality/objectiveerrorbound.', 'interior_equations': 'SAMEcompleteoriginal78791primarydirection computesmodeledphi/theta/ACTUAL(mu+theta)/2ceiling/prediction;doNOTassumephi0fromt0. PrimaryNEVERmapped. ActualinteriorfreshFULLoriginal490 min-eta,alloriginalrows/oldzeroE/originalbounds;zeroFACEgraphsinvalid. SamecoldIPMonce,backend78791r.xBITWISE,dualsy=-ineq,nu=-eq,upper=-upper,lower=lower;fullKKT.', 'initial_assembly': 'Inherit524initialfullprimary/context/92267intervals and525proof aftercomplete472INITIALX/Y/J+490canonicalaux/drop/trustidentity. Bind517547559571573579581587589595597proofs. ZEROinitialassembly/root/oldgraph/inversion/encoding/proofreplay.', 'changed_input': 'Genuinelynewstate/trust/ACTUALceiling freshFULL490 andindependentfullassembly/intervalproof BEFORELPinside45sTOTAL. ConservativeFULLoriginalvars/rows;NOoldstatereductionsreused. NewprimaryaddsONLYzeroFACEwithindependentnewnative/scalar/RHS/copyproof;interioruntouchedFULL. No nonzeroepigraphfacesequence.', 'clock': 'First45sTOTAL startsMAINentry BEFOREscientificimports/globalhashes/loads/context/preparation;laterbeforeparentnewassembly. SharedLinuxmonotonicoriginchild. ALLhashchecks/prep/backend/exactpoint/compositegraph/allKKTcharged;onlyremainingbackendallowancepassedONCE. Checksbetweenoperations;overshootreportednotextension.40GiBchildAS/120sparentchildwall. Denseexactpoint/threegraphcost not waived orassignedoutside45s; no guaranteedcompletion/performanceclaim.', 'budget_rationale': '24major48actualLP120maps5dyadics1..1/16;trust1..1/32guards2^-20,halveonlyfinite/nonlinearreject,double<=1onlyratio>.75,accept>=.1SAMEeffectiveencodedfeaturedelta. Firstabsent/nonfinite/uncertified/infeasible/deadline terminatesENTIREconstruction,noretry.', 'finite_policy': 'Neveruseoriginalinfiniteboundswith596compensation. Retained10541backendvaluesBITWISE/scalaralready0. UnroundedEXACTrootxy/f feeds586legs;unroundedEXACTrootandlegsfeed578aux. ALL68250removedcoordsroundONCE. Fulloriginalchecks;no universalbinary64extension. Exactunscaledeta>0beforeimmutable490encoder,newXfreshmechanics/J,no15root. Allsignedrows/7units1e-13/normal1e-13/hover1e-8/geometry/filter/runtime/72retention/PPOgatesunchanged.', 'partial_return': 'PersistNEWexactpointresiduals androot/leg/auxvalues asprocessed;incompleteprefixnevercertificate. Independent599 audits savedNEWpartialpointarithmetic withoutreplayingoldcoefficientproofs. Counts/partiallogspreserved onterminalfailure;no selecting partialvector.', 'limits': 'FeweroptimizercolumnsbutdenserA notspeed/conditioning/convergenceevidence. Innerinput/zeroFACE failure notoriginalglobalinfeasibility. No oldinternal/rejectedvectors/physicalpromotion/guard/tolerance/budgetchanges.', 'physical_admission': False})
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
 if folder==D/'balance_path_598/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 write(child/'clock_origin.json',dict(parent_monotonic_origin=origin,parent_preparation_elapsed_s=start-origin,all_new_preparation_charged=True))
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_balance_path_lp_598.py')),str(child),repr(origin)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='condensed-balance-inner-zero-face-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=bool(meta.get('coordinate_reconstruction_attempts',0)))
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['full']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=not bool(meta.get('coordinate_reconstruction_attempts',0)),retained_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
