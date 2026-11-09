"""Complete compensated580 graph-zero-face plus actual full-original interior construction582."""
from low_seat_lifted_execution_492 import *
from pathlib import Path
from fractions import Fraction as F
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(method='compensated-graph-zero-face-IPM',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='Initialprimary31016original retained coordinates includingALL20475physicalX,10540priorcontrols,scalaroriginal78790atreduced31015.47775auxiliary command/torque/COM outputs exactgraphliftthenroundONCE. AuxiliaryoutputsNOTactuatorauthority;noinverse/factor/15root. ChangedinputsconservativeFULLoriginal78791.',
 structural_method='578 exact rational graph elimination on572zeroFACE plus580 certified binary64 INNER approximation with explicitOUTER coordinatebox andinwarderror-compensatedRHS. WholeoriginalLP/finite/tolerance-equivalence NOTclaimed;mayempty/proper. No newgraph/encoding/analyticproof oninitialinput.',
 initialization='ONE coldinstalled scipy.linprog highs-ipm per genuinelynewinput;fixedpresolveTrue,maxiter4096,primal/dual/IPM1e-9,no x0/basis/warmstart/restart/fallback. Initial580inputinheritedbyhash,notanother574run.',
 primary_equations='min t subject580Ahat.u<=bhat,retainedAFFINEE.u=d,EXPLICITfinite580outerbounds,scalarzero equality. Neverreplace580boundsbyinfiniteoriginalbounds. Require status0/success/finite originalretained result.x scalarALREADY0(+/-0),ALL31016valuesBITWISE unchanged. Evaluate ALL578rationalrows exactlyatreturnedpoint; lift47775outputs byexactgraphsum/divisionthenroundONCE,rejectnonfinite/fullchecks,no clip/repair/normalize. Inherit573analyticlowerbound(lower_t1others0). ALLencoded580/exactrational578/572/570/546/516/FULL490 primal/eq/bound/stationarity/absolute-gap<=1e-9 inside1e-7. KKTNOTexactoptimality/objectiveerrorbound.',
 interior_equations='SAMEcomplete78791direction computesmodeledtheta/phi andACTUAL(mu+theta)/2ceiling;doNOTassumephi0fromt0. PrimaryNEVERmapped. InteriorusesfreshFULLoriginal490 min-eta,ALLoriginalrows/zeroE/originalbounds;zero-facegraphdoesNOTapply. ColdsameIPMonce,original78791backendvaluesBITWISE;backenddual signs y=-ineq,nu=-eq,upper=-upper,lower=lower;fullKKT<=1e-9.',
 initial_assembly='Inherit524initialfullprimary/context/92267intervals and525proof after472INITIALX/Y/J+490canonicalaux/drop/trust identity. Bind517547559571573579581 proofhashes;no initialassembly/graph/encoding/oldproof replay.',
 changed_input='Genuinelychangedstate/trust/ACTUALceiling usesfreshFULL490 plusindependentcompleteassembly/intervalproof beforeLP;conservativeFULLoriginalvariables/rows,NOold-stategraph/box/coupled/quotient/encodingproofused. NewprimaryONLYbuildszeroFACEwithindependentnewnative/scalar/RHS/copyproof;newinterioruntouchedFULL. Allcharged45sTOTAL.',
 clock='First45sLP clockatMAINentry BEFOREscientificimports/globalinputhashes/loads/context/copies;laterclockbeforeparentnewassembly. SharedLinuxmonotonicorigin passedchild. Parentprep,inputproofs,backend,allpointlifts/rationalchecks/KKTcharged. Onlyremainingbackendallowancepassedonce;checksbetweenoperations,overshootreported.40GiBchild/120sparentchildwall.',
 budget_rationale='24major48actualLP120maps5dyadics1..1/16;trust1..1/32guards2^-20,halveonlyfinite/nonlinearreject,double<=1onlyratio>.75,accept>=.1sameeffectiveencodedfeaturedelta. Firstabsent/nonfinite/uncertified/infeasible/deadline TERMINATES ENTIREconstruction;no timeouttrustseries/otherface/solverretry.',
 finite_policy='PrimaryNEVERmapped;samecompleteoriginaldirectionforphi/ceiling/prediction. Interiordyadics thenexactoriginalunscaledintervaleta>0beforeimmutable490encoder,newXfreshmechanics/J,no15root. Allactualsignedrows/7units1e-13/normal1e-13/hover1e-8/geometry/filter/runtime/72retention/PPOgates unchanged.',
 limits='Fewercolumns butdenserrowsNOTspeed/conditioning/deadlineevidence. EncodedinnerfacefailureNOToriginalLP/physicalinfeasibility. No completedstateproofreplay/internalvectors/physicalpromotion.',physical_admission=False)
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
 if folder==D/'graph_path_582/major_00/primary':
  p=load_problem(folder);assert read(folder/'stage.json')==dict(ceiling=None,primary=True)
 else:
  p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
 child=folder/'lp';child.mkdir();start=time.monotonic();timed=False
 write(child/'clock_origin.json',dict(parent_monotonic_origin=origin,parent_preparation_elapsed_s=start-origin,all_new_preparation_charged=True))
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_graph_path_lp_582.py')),str(child),repr(origin)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0)
 meta.update(actual_lp_call=(child/'scipy_started.json').exists(),worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='compensated-graph-zero-face-IPM',native_candidate=meta['candidate'],backsubstitution_attempted=False)
 meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if (child/'input_proof.json').exists():
  proof=read(child/'input_proof.json');systems.inherited_systems+=int(proof['inherited']);systems.new_systems+=int(not proof['inherited']);systems.new_proof_rows+=proof.get('new_quotient_proof_rows',0)
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  raw=np.load(child/'witness.npy');checks=read(child/'original_checks.json');v=violation(p,raw)
  if checks['passed'] and v<=1e-7:
   w=raw;np.save(folder/'witness.npy',w);meta.update(linear_violation=v,source_linear_violation=checks['full']['primal'],linear_pass=True,objective=float(p['cost']@w),native_values_unchanged=not bool(meta.get('coordinate_reconstruction_attempts',0)),retained_values_unchanged=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
