"""Independent exact proofs for NEW548 original systems only; no old517/547 proof rerun."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
from scipy import sparse
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def entries(A,i):
 st,en=A.indptr[i:i+2];return {int(j):F(float(v)) for j,v in zip(A.indices[st:en],A.data[st:en]) if v}
def prove(full,arr,A,z,ids,ledger,src,o,stats,out,check):
 same(A,full[ids]);assert np.array_equal(z['rhs'],arr['rhs'][ids]);chosen=set(map(int,ids));flags=np.zeros(full.shape[0],np.uint8)
 for i,j in enumerate(ledger['representative']):
  if i%128==0:check()
  if j<0:assert not entries(full,i) and F(float(arr['rhs'][i]))>=0
  elif i!=j:
   assert int(j) in chosen;v=entries(full,i);u=entries(full,int(j));assert v.keys()==u.keys() and v;c=min(v);ratio=v[c]/u[c];assert ratio>0 and all(v[k]*u[c]==u[k]*v[c] for k in v) and F(float(arr['rhs'][i]))>=ratio*F(float(arr['rhs'][j]))
  else:assert int(j) in chosen
  flags[i]=1
 np.save(out/'new_quotient_verified.npy',flags)
 for k in ['cost','bounds','equality_rhs']:assert np.array_equal(z[k],arr[k])
 B=sparse.load_npz(o/'matrix.npz').tocsr();bz=dict(np.load(o/'arrays.npz'));keep=np.load(o/'retained_source_positions.npy');dec=np.load(o/'decisions.npy');owners=np.load(o/'bound_owners.npy');ep=[[None if q is None else F(q) for q in r] for r in read(o/'endpoints.json')]
 assert len(ep)==A.shape[1] and owners.shape==(A.shape[1],2) and len(dec)==A.shape[0] and np.all(dec<=3)
 # Independent premises a*x_j<=b, with signed unit rows for direct variable bounds.
 prem={};all_unary=set(np.flatnonzero(np.diff(A.indptr)==1).tolist());seen_sides=set()
 for j in range(A.shape[1]):
  if j%128==0:check()
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
  if j%128==0:check()
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
   check()
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
 assert sha(o/'equality.npz')==sha(src/'equality.npz')
 assert np.array_equal(np.load(o/'original_row_ids.npy'),np.load(src/'row_ids.npy')[keep])
 ps=stats
 assert positive==ps['proved_removed_rows'] and negative==ps['negative_box_support_tests'] and untested==ps['missing_endpoint_untested'] and terms==ps['exact_support_terms']
 np.save(out/'new_box_verified.npy',verified)
 return dict(passed=True,new_quotient_proof_rows=len(flags),new_box_proof_rows=len(verified),new_positive_box_implications=positive,new_negative_box_tests=negative,new_exact_conic_products=terms,original_copy_error=0,old_proofs_repeated=False)
