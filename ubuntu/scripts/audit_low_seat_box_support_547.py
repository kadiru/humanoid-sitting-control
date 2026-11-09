"""Independent positive-conic coefficient/RHS certificates and ALL negative decisions; no producer import."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime,time,resource
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
 assert read(D/'box_support_preparation_exit_546.json')['actual_exit']==0
 for p,h in read(D/'box_support_launch_546.json')['source_input_sha256'].items():assert sha(p)==h,p
 prep=D/'box_support_preparation_546'
 for p,h in read(prep/'SHA256.json').items():assert sha(prep/p)==h,p
 assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed']
 out=D/'box_support_audit_547';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  src=D/'exact_row_quotient_preparation_516'/role/'original';A=sparse.load_npz(src/'matrix.npz').tocsr();z=dict(np.load(src/'arrays.npz'));o=prep/role
  B=sparse.load_npz(o/'matrix.npz').tocsr();bz=dict(np.load(o/'arrays.npz'));keep=np.load(o/'retained_source_positions.npy');dec=np.load(o/'decisions.npy');owners=np.load(o/'bound_owners.npy');ep=[[None if q is None else F(q) for q in r] for r in read(o/'endpoints.json')]
  assert len(ep)==A.shape[1] and owners.shape==(A.shape[1],2) and len(dec)==A.shape[0] and np.all(dec<=3)
  # Independent premises a*x_j<=b, with signed unit rows for direct variable bounds.
  prem={};all_unary=set(np.flatnonzero(np.diff(A.indptr)==1).tolist());seen_sides=set()
  for j in range(A.shape[1]):
   for side in [0,1]:
    owner=int(owners[j,side]);q=ep[j][side]
    if owner==-2:assert q is None;continue
    if owner==-1:
     assert np.isfinite(z['bounds'][j,side]);a=F(1 if side else -1);b=a*F(float(z['bounds'][j,side]))
    else:
     assert owner in all_unary and dec[owner]==0;k=A.indptr[owner];assert A.indices[k]==j
     a=F(float(A.data[k]));b=F(float(z['rhs'][owner]));assert (a>0)==bool(side)
    assert a*q==b;prem[j,side]=(a,b);seen_sides.add((j,side))
   lo,hi=ep[j];assert lo is None or hi is None or lo<=hi
  # Verify tightness, not production endpoint selection; ties need not be reselected.
  for i in all_unary:
   k=A.indptr[i];j=int(A.indices[k]);a=F(float(A.data[k]));side=int(a>0);assert (j,side) in prem
   assert a*ep[j][side]<=F(float(z['rhs'][i]))
  for j in range(A.shape[1]):
   for side in [0,1]:
    if np.isfinite(z['bounds'][j,side]):
     assert (j,side) in prem;q=F(float(z['bounds'][j,side]));assert ep[j][side]<=q if side else ep[j][side]>=q
  # Every absent endpoint must really lack both a direct bound and a singleton premise.
  actual_sides={(int(A.indices[A.indptr[i]]),int(A.data[A.indptr[i]]>0)) for i in all_unary}
  actual_sides.update((j,s) for j in range(A.shape[1]) for s in (0,1) if np.isfinite(z['bounds'][j,s]))
  assert actual_sides==seen_sides
  verified=np.zeros(A.shape[0],dtype=np.uint8);terms=0;positive=negative=untested=0
  with (o/'exact_decisions.jsonl').open() as ledger:
   for i in range(A.shape[0]):
    if time.monotonic()-start>240:raise TimeoutError('240s independent certificate audit cap')
    lo,hi=A.indptr[i:i+2]
    if hi-lo<=1:assert dec[i]==0;verified[i]=1;continue
    choices=[(int(j),int(a>0)) for j,a in zip(A.indices[lo:hi],A.data[lo:hi])]
    if any(c not in prem for c in choices):assert dec[i]==3;untested+=1;verified[i]=1;continue
    row,slack=json.loads(next(ledger));assert row==i;bound=F(0)
    for k,c in zip(range(lo,hi),choices):
     original_a=F(float(A.data[k]));pa,pb=prem[c];multiplier=original_a/pa
     assert multiplier>0 and multiplier*pa==original_a
     bound+=multiplier*pb;terms+=1
    deficit=F(float(z['rhs'][i]))-bound;assert deficit==F(slack)
    if deficit>=0:assert dec[i]==2;positive+=1
    else:assert dec[i]==1;negative+=1
    verified[i]=1
   assert ledger.read()==''
  assert np.all(verified==1) and np.array_equal(keep,np.flatnonzero(dec!=2))
  expected=A[keep].tocsr()
  assert B.shape==expected.shape and np.array_equal(B.indptr,expected.indptr) and np.array_equal(B.indices,expected.indices) and np.array_equal(B.data,expected.data)
  assert set(bz)==set(z)
  for k in z:assert np.array_equal(bz[k],z[k][keep] if k=='rhs' else z[k]),k
  assert sha(o/'equality.npz')==sha(D/'lifted_feasibility_preparation_490'/role/'equality.npz')
  assert np.array_equal(np.load(o/'original_row_ids.npy'),np.load(src.parent/'row_ids.npy')[keep])
  ps=read(prep/'preparation.json')['systems'][role]
  assert positive==ps['proved_removed_rows'] and negative==ps['negative_box_support_tests'] and untested==ps['missing_endpoint_untested'] and terms==ps['exact_support_terms']
  np.save(out/(role+'_all_decisions_verified.npy'),verified)
  systems[role]=dict(source_rows=A.shape[0],retained_rows=B.shape[0],source_nonzeros=A.nnz,retained_nonzeros=B.nnz,positive_conic_certificates=positive,negative_sufficient_tests=negative,missing_endpoint_untested=untested,exact_conic_coefficient_products=terms,finite_bound_premises=len(prem),all_unary_premises_explicit=True,all_decisions_verified=True,retained_coefficient_rhs_copy_errors=0,cost_bounds_equalities_unchanged=True,variables_removed=0)
  print(json.dumps(dict(role=role,**systems[role])),flush=True)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,systems=systems,elapsed_s=time.monotonic()-start,prior517_proof_inherited_not_repeated=True,production_imported=False,lp_calls=0,small_examples=0,linear_solves=0,model_calls=0,root_calls=0,finite_maps=0,new_solver=False,new_candidate=False,physical_admission=False,limits='Exact-real source redundancy only. Negative tests are not nonredundancy/infeasibility certificates. No numerical witness, performance claim or actual-model/physical admission.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(passed=True,elapsed_s=r['elapsed_s'])),flush=True)
if __name__=='__main__':main()
