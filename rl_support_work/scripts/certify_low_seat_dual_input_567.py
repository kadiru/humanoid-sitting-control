"""Independent NEW-only dual input proof; no producer/optimizer or immutable564/565 replay."""
from pathlib import Path
import numpy as np
from scipy import sparse
import json
read=lambda p:json.loads(Path(p).read_text())
def bits(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def prove_dual(src,new,out,check):
 A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'))
 B=sparse.load_npz(new/'equality.npz').tocsr();z=dict(np.load(new/'arrays.npz'));layout=read(new/'layout.json')
 m,n=A.shape;j=78790
 assert n==78791 and A.has_canonical_format and B.has_canonical_format
 assert B.shape==(n,m+2) and B.nnz==A.nnz+2 and np.isfinite(B.data).all() and np.all(B.data!=0)
 assert z['cost'].shape==(m+2,) and z['equality_rhs'].shape==(n,) and z['bounds'].shape==(m+2,2)
 assert E.shape==(1366,n) and E.nnz==0 and np.all(a['equality_rhs']==0)
 assert np.all(a['cost'][:j]==0) and abs(a['cost'][j])==1
 assert np.all(np.isneginf(a['bounds'][:j,0])) and np.all(np.isposinf(a['bounds'][:j,1]))
 L,U=a['bounds'][j];assert np.isfinite([L,U]).all() and L<=U
 assert np.all(z['bounds'][:,0]==0) and np.all(np.isposinf(z['bounds'][:,1]))
 bits(z['cost'][:m],a['rhs']);assert z['cost'][m]==U and z['cost'][m+1]==-L
 bits(z['equality_rhs'],-a['cost'])
 for dest,name in [('original_row_ids.npy','original_row_ids.npy'),('source516_positions.npy','retained_source_positions.npy')]:bits(np.load(new/dest),np.load(src/name))
 slots=np.load(new/'original_coordinate_slots.npy');assert np.array_equal(slots,np.arange(n,dtype=np.int64))
 assert layout['original_scalar_slot']==j and layout['scalar_upper_multiplier_column']==m and layout['scalar_lower_multiplier_column']==m+1
 # Every new balance row is exactly the original stationarity coefficient list.
 # No product/re-solve or old implication proof is used.
 C=A.tocsc();native_terms=0
 for k in range(n):
  if k%256==0:check()
  c0,c1=C.indptr[k:k+2];b0,b1=B.indptr[k:k+2];bi=B.indices[b0:b1];bv=B.data[b0:b1]
  retained=bi<m;assert np.array_equal(bi[retained],C.indices[c0:c1]);bits(bv[retained],C.data[c0:c1]);native_terms+=int(c1-c0)
  if k==j:
   assert np.array_equal(bi[~retained],[m,m+1]) and np.array_equal(bv[~retained],[1.,-1.])
  else:assert retained.all()
 # Independently view each multiplier column: q_i-B_i.T*w is exactly
 # the original row slack; upper/lower columns give U-t and t-L.
 T=B.tocsc();slack_terms=0
 for i in range(m):
  if i%256==0:check()
  a0,a1=A.indptr[i:i+2];t0,t1=T.indptr[i:i+2]
  assert np.array_equal(T.indices[t0:t1],A.indices[a0:a1]);bits(-T.data[t0:t1],-A.data[a0:a1]);slack_terms+=int(a1-a0)
 for i,sgn in [(m,1.),(m+1,-1.)]:
  t0,t1=T.indptr[i:i+2];assert np.array_equal(T.indices[t0:t1],[j]) and np.array_equal(T.data[t0:t1],[sgn])
 # Gap coefficient identities hold as exact dyadics, with nu=-w.
 assert np.array_equal(-z['equality_rhs'],a['cost'])
 assert np.array_equal(z['cost'][:m],a['rhs']) and z['cost'][m]==U and z['cost'][m+1]==-L
 assert native_terms==slack_terms==A.nnz
 np.save(out/'new_dual_coordinate_slots_verified.npy',slots)
 np.save(out/'new_dual_multiplier_slots_verified.npy',np.arange(m,dtype=np.int64))
 return dict(passed=True,dual_shape=list(B.shape),dual_nonzeros=B.nnz,original_stationarity_coefficient_identities=native_terms,original_slack_coefficient_identities=slack_terms,scalar_bound_column_identities=4,signed_gap_coordinate_coefficient_identities=n,signed_gap_multiplier_coefficient_identities=m+2,retained_copy_error_exact=0,original_primal_from_equality_marginals=True,optimizer_primal_is_multiplier_vector=True,original_domain_unchanged=True,candidate_evaluated=False)
