"""Independent exact certificates for genuinely NEW projected LP systems; never replays522/523."""
from audit_low_seat_torque_projection_523 import entries,same,Q,read,sha
from pathlib import Path
import numpy as np
from scipy import sparse
import json,itertools

def certify_new(full,source,projected,expected_el):
 A=sparse.load_npz(source/'matrix.npz').tocsr();a=dict(np.load(source/'arrays.npz'));B=sparse.load_npz(projected/'matrix.npz').tocsr();b=dict(np.load(projected/'arrays.npz'));led=dict(np.load(projected/'projection_ledger.npz'));el=led['eliminated_columns'];ret=led['retained_columns'];unt=led['unchanged_rows'];pairs=led['pairs'];P=projected.parent;role=projected.name
 fullA=sparse.load_npz(full/'matrix.npz').tocsr();fulla=dict(np.load(full/'arrays.npz'));fullE=sparse.load_npz(full/'equality.npz');source_ids=np.load(source/'row_ids.npy');rep=np.load(source/'ledger.npz')['representative'];chosen=set(map(int,source_ids))
 same(A,fullA[source_ids]);assert np.array_equal(a['rhs'],fulla['rhs'][source_ids])
 for k in ['cost','bounds','equality_rhs']:assert np.array_equal(a[k],fulla[k])
 flags=np.zeros(len(rep),bool)
 for i,j in enumerate(rep):
  v=entries(fullA,i)
  if j<0:assert not v and Q(float(fulla['rhs'][i]))>=0
  else:
   assert int(j) in chosen
   if i!=j:
    u=entries(fullA,int(j));assert v and set(u)==set(v);c=min(v);ratio=v[c]/u[c];assert ratio>0
    assert all(v[k]*u[c]==u[k]*v[c] for k in v)
    assert Q(float(fulla['rhs'][i]))-ratio*Q(float(fulla['rhs'][j]))>=0
  flags[i]=True
 np.save(source/'new_implication_proof.npy',flags)
 assert np.array_equal(el,expected_el) and len(el)==21840;assert np.array_equal(ret,np.setdiff1d(np.arange(A.shape[1]),el));assert np.all(a['cost'][el]==0) and np.all(a['bounds'][el,0]==-np.inf) and np.all(a['bounds'][el,1]==np.inf)
 incidence=A[:,el].tocsr();assert np.all(np.diff(incidence.indptr)<=1);assert np.array_equal(unt,np.flatnonzero(np.diff(incidence.indptr)==0))
 same(B[:len(unt)],A[unt][:,ret]);assert np.array_equal(b['rhs'][:len(unt)],a['rhs'][unt]);assert np.array_equal(b['cost'],a['cost'][ret]) and np.array_equal(b['bounds'],a['bounds'][ret]) and np.array_equal(b['equality_rhs'],a['equality_rhs'])
 E=fullE;assert E[:,el].nnz==0;same(sparse.load_npz(projected/'equality.npz'),E[:,ret]);assert np.array_equal(np.load(projected/'source_row_ids.npy'),source_ids)
 C=A.tocsc();rowcache={};cursor=0;dest=len(unt);flags=np.zeros(len(pairs),dtype=np.bool_);coefs=0;statuses=[0,0,0]
 inverse={int(c):i for i,c in enumerate(ret)}
 for c in el:
  start,end=C.indptr[c:c+2];rr=C.indices[start:end];vv=C.data[start:end];lower=rr[vv<0];upper=rr[vv>0];assert len(lower)>0 and len(upper)>0
  rowcache={int(i):entries(A,int(i)) for i in rr}
  for low,high in itertools.product(lower,upper):
   col,li,ui,output,status,power=map(int,pairs[cursor]);assert (col,li,ui)==(int(c),int(low),int(high))
   L=rowcache[li];U=rowcache[ui];al=L[col];au=U[col];assert al<0<au
   # Independent endpoint subtraction, then positive denominator product.
   factor=au*(-al);rest={j:(U.get(j,Q(0))/au-L.get(j,Q(0))/al)*factor for j in set(L)|set(U)};assert rest[col]==0;rest={j:q for j,q in rest.items() if q};rhs=(Q(float(a['rhs'][ui]))/au-Q(float(a['rhs'][li]))/al)*factor
   assert all(j in inverse for j in rest)
   maximum=Q(0);bounded=True
   for j,q in rest.items():
    endpoint=a['bounds'][j,1 if q>0 else 0]
    if not np.isfinite(endpoint):bounded=False;break
    maximum+=q*Q(float(endpoint))
   expect=1 if not rest and rhs>=0 else 2 if bounded and maximum<=rhs else 0;assert status==expect;statuses[status]+=1
   if status:
    assert output==-1 and power==0
   else:
    assert output==dest;scale=Q(2)**power;largest=max([abs(rhs)]+[abs(q) for q in rest.values()])*scale;assert Q(1,2)<largest<=1
    got=entries(B,output);assert set(got)=={inverse[j] for j in rest}
    for j,q in rest.items():
     rounded=float(q*scale);assert np.isfinite(rounded) and rounded!=0;assert got[inverse[j]]==Q(rounded);coefs+=1
    assert b['rhs'][output]==float(rhs*scale) and (rhs==0 or b['rhs'][output]!=0);dest+=1
   flags[cursor]=True;cursor+=1
 np.savez_compressed(projected/'new_pair_proof.npz',verified_pairs=flags,source_rows=unt,eliminated_columns=el)
 assert cursor==len(pairs) and dest==B.shape[0] and flags.all();result=dict(pairs_verified=cursor,generated_coefficients_verified=coefs,kept_pairs=statuses[0],constant_tautologies=statuses[1],variable_bound_implications=statuses[2],matrix_rhs_copy_error=0,rounded_exact_pair_error=0,shape=list(B.shape),nnz=B.nnz,source_quotient_proof_inherited_not_repeated=False)
 result.update(implication_rows_verified=len(rep),old_proofs_repeated=False,new_examples_verified=0)
 (projected/'new_system_proof.json').write_text(json.dumps(result,indent=2))
 return result
