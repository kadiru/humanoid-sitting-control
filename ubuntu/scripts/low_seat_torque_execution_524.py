"""Bounded original-coordinate existential torque projection with exact binary64 backsubstitution."""
from low_seat_lifted_execution_492 import *
from low_seat_quotient_execution_518 import same_problem
from low_seat_exact_row_quotient_v2_516 import quotient
from low_seat_torque_projection_522 import project,row
from certify_low_seat_torque_system_525 import certify_new
from fractions import Fraction as F
from pathlib import Path
import subprocess,sys,shutil
SPEC=dict(SPEC)
SPEC.update(scope=__doc__,method='highs-ipm',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 coordinate_system='Original490 coordinates with21840 auxiliary torque increments existentially eliminated.56950 retained directions plus epigraph/eta. No504 transform or new inverse.',
 rationale='The audited522/523 disjoint interval projection reduces actual variable count, with exact source implication provenance. It is not a new basis or cut queue. Witness coefficients can increase; no speed or timeout-cause claim.',
 reconstruction='Retained optimal solver coordinates stay bitwise unchanged. Exact Fraction source endpoint intersection for every eliminated auxiliary. Empty intersection rejects. Choose exact midpoint and round ONCE to nearest binary64; if that float is nonfinite or lies outside the exact closed intersection reject. No nextafter/clipping/vector repair. This deliberately rejects nonrepresentable singleton intervals. Exact rational endpoints, midpoint, rounding error and owning rows are saved.',
 full_checks='Projected numeric, source representative, and complete ORIGINAL490 inequalities/equalities/bounds must ALL pass1e-7. No transformed504 system is used or claimed checked. Both torque consistency signs/unit-scales preserved existentially. Fully reconstructed direction supplies modeled phi and all dyadic trials; primary is never mapped.',
 lifecycle='Exact complete reference primary/witness inherits516original subset/517proof and522projection/523proof after full matrix/RHS/cost/bound/equality and source identity. Any changed full LP gets its own original exact quotient and projection, independently certified BEFORE its solve. Old examples/proofs never rerun.',
 solver_failure_policy='First absent/nonoptimal/uncertified LP, empty exact interval, nonrepresentable chosen float or resource failure terminates the entire run. One method, no retry or timeout-driven trust changes.',
 budget_rationale='24major,48actual LP,120finite maps,5shared dyadics1..1/16;45s native1e-9, full linear1e-7,40GiB child/120s wall. Same original guards/trust1..1/32/ratio. No old budget extended.',
 finite_policy='Original complete direction before scaling, exact unscaled interval eta>0 before immutable490 decoder. SAME effective encoded delta for prediction. Fresh changed actualX mechanics/J; no15-root. Unit closures and all actual signed/geometry/filter/runtime gates unchanged.',
 physical_admission=False)

def save_problem(folder,p):
 folder.mkdir(exist_ok=False)
 for name in ['matrix','equality']:sparse.save_npz(folder/(name+'.npz'),p[name])
 np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']})
def load_problem(folder):
 return dict(matrix=sparse.load_npz(folder/'matrix.npz').tocsr(),equality=sparse.load_npz(folder/'equality.npz').tocsr(),**dict(np.load(folder/'arrays.npz')))
def violation(p,w):
 if not np.all(np.isfinite(w)):return float('inf')
 return max(0.,float((p['matrix']@w-p['rhs']).max()),float(abs(p['equality']@w-p['equality_rhs']).max()),float((p['bounds'][:,0]-w).max()),float((w-p['bounds'][:,1]).max()))
class Projections:
 def __init__(self):
  ex=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];self.el=10540+np.flatnonzero((ex%75>=55)&(ex%75<71));assert len(self.el)==21840
  self.inherited_systems=0;self.new_systems=0;self.new_proof_rows=0;self.new_pairs=0
 def get(self,folder,p,role):
  source=folder/'source';reduced=folder/'projected';old=D/'lifted_feasibility_preparation_490'/role;inherited=same_problem(p,old)
  if inherited:
   q=D/'exact_row_quotient_preparation_516'/role;ids=np.load(q/'row_ids.npy');assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed'];assert read(D/'torque_projection_audit_523/independent_audit.json')['passed']
   src=dict(p,matrix=p['matrix'][ids],rhs=p['rhs'][ids]);assert same_problem(src,q/'original');save_problem(source,src);np.save(source/'row_ids.npy',ids);shutil.copy2(q/'original_ledger.npz',source/'ledger.npz')
   sealed=D/'torque_projection_preparation_522'/role;shutil.copytree(sealed,reduced);assert np.array_equal(np.load(reduced/'source_row_ids.npy'),ids)
   counts=read(D/'torque_projection_preparation_522/preparation.json')['systems'][role];self.inherited_systems+=1
   proof=dict(inherited=True,implication_manifest_sha256=sha(D/'exact_row_quotient_audit_517/SHA256.json'),projection_manifest_sha256=sha(D/'torque_projection_audit_523/SHA256.json'),prior_proofs_repeated=False)
  else:
   led=quotient(p['matrix'],p['rhs']);ids=led['representatives'];src=dict(p,matrix=p['matrix'][ids],rhs=p['rhs'][ids]);save_problem(source,src);np.save(source/'row_ids.npy',ids);np.savez_compressed(source/'ledger.npz',**led)
   reduced.mkdir();counts=project(src['matrix'].copy(),src,self.el,reduced);ret=np.load(reduced/'projection_ledger.npz')['retained_columns'];assert src['equality'][:,self.el].nnz==0
   sparse.save_npz(reduced/'equality.npz',src['equality'][:,ret]);np.save(reduced/'source_row_ids.npy',ids)
   proof=certify_new(folder,source,reduced,self.el);proof['inherited']=False;self.new_systems+=1;self.new_proof_rows+=proof['implication_rows_verified'];self.new_pairs+=proof['pairs_verified']
  scope=dict(role=role,inherited=inherited,full_rows=p['matrix'].shape[0],source_rows=len(ids),full_columns=p['matrix'].shape[1],**counts,proof=proof)
  write(folder/'projection_scope.json',scope)
  write(folder/'certified_system_hashes.json',{str(x.relative_to(folder)):sha(x) for base in [source,reduced] for x in base.rglob('*') if x.is_file()})
  return src,load_problem(reduced),dict(np.load(reduced/'projection_ledger.npz')),scope

def reconstruct(source,ledger,q,folder):
 A=source['matrix'];b=source['rhs'];ret=ledger['retained_columns'];el=ledger['eliminated_columns'];C=A.tocsc();w=np.zeros(A.shape[1]);w[ret]=q;exact={int(c):F(float(v)) for c,v in zip(ret,q)};records=[];fail=[]
 for c in el:
  c=int(c);start,end=C.indptr[c:c+2];low=[];up=[]
  for i in C.indices[start:end]:
   i=int(i);r=row(A,i);a=r.pop(c);assert all(j in exact for j in r);rest=sum((v*exact[j] for j,v in r.items()),F(0));endpoint=(F(float(b[i]))-rest)/a
   (low if a<0 else up).append((endpoint,i))
  lo,li=max(low,key=lambda x:(x[0],-x[1]));hi,ui=min(up,key=lambda x:(x[0],x[1]));rec=dict(column=c,lower=pair(lo),upper=pair(hi),lower_row=li,upper_row=ui)
  if lo>hi:rec.update(passed=False,reason='empty exact source intersection');fail.append(c)
  else:
   mid=(lo+hi)/2
   try:value=float(mid)
   except OverflowError:value=float('inf')
   ok=bool(np.isfinite(value) and lo<=F(value)<=hi);rec.update(midpoint=pair(mid),rounded=value,rounding_error=pair(F(value)-mid) if np.isfinite(value) else None,passed=ok,reason=None if ok else 'once-rounded midpoint outside exact interval or nonfinite')
   if ok:w[c]=value
   else:fail.append(c)
  records.append(rec)
 write(folder/'backsubstitution.json',dict(policy=SPEC['reconstruction'],checked=len(records),failed_columns=fail,passed=not fail,records=records,retained_values_unchanged=True))
 if fail:return None
 assert np.array_equal(w[ret],q);np.save(folder/'witness.npy',w);return w

def solve_projected(folder,d,b,s,projections,ceiling=None):
 p=assemble(d,b,s,ceiling);save_problem(folder,p);write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None));role='primary' if ceiling is None else 'witness';source,z,ledger,scope=projections.get(folder,p,role);child=folder/'projected';started=time.monotonic();timed=False
 with (child/'worker.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_torque_lp_524.py')),str(child)],stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
 called=(child/'scipy_started.json').exists();meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='child resource failure',candidate=False,success=False,iterations=0,crossover_iterations=0)
 meta.update(actual_lp_call=called,worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-started,method='highs-ipm',projection=scope,projected_candidate=meta['candidate'],backsubstitution_attempted=False);meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
 if rc==0 and not timed and meta['status']==0 and meta['candidate']:
  q=np.load(child/'witness.npy');pv=violation(z,q);meta.update(projected_linear_violation=pv)
  if pv<=1e-7 and np.all(np.isfinite(q)):
   meta['backsubstitution_attempted']=True;w=reconstruct(source,ledger,q,folder)
   if w is not None:
    v=violation(p,w);sv=violation(source,w);np.savez_compressed(folder/'full_residuals.npz',original=p['matrix']@w-p['rhs'],source=source['matrix']@w-source['rhs'],projected=z['matrix']@q-z['rhs']);meta.update(linear_violation=v,source_linear_violation=sv,linear_pass=bool(max(pv,v,sv)<=1e-7),objective=float(p['cost']@w),reconstruction_unmodified=True)
 meta['candidate']=w is not None;meta['terminal_failure']=not(rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False));write(folder/'solver.json',meta);return w,meta
