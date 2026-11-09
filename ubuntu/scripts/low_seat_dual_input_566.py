"""Fresh-input-only dual construction566; immutable564 is inherited, never transposed again."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from low_seat_box_input_548 import quotient,derive
def derive_dual(src,out,check):
 A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'));ids=np.load(src/'original_row_ids.npy');pos=np.load(src/'retained_source_positions.npy')
 m,n=A.shape;j=n-1
 assert n==78791 and A.has_canonical_format and np.all(A.data!=0) and np.isfinite(A.data).all()
 assert E.shape==(1366,n) and E.nnz==0 and np.all(a['equality_rhs']==0)
 assert np.all(a['cost'][:j]==0) and a['cost'][j] in (-1.,1.)
 assert np.all(np.isneginf(a['bounds'][:j,0])) and np.all(np.isposinf(a['bounds'][:j,1]))
 L,U=map(float,a['bounds'][j]);assert np.isfinite([L,U]).all() and L<=U
 check()
 C=sparse.csr_matrix(([1.,-1.],([j,j],[0,1])),shape=(n,2))
 B=sparse.hstack([A.T,C],format='csr');q=np.r_[a['rhs'],U,-L];h=-a['cost'];bounds=np.c_[np.zeros(m+2),np.full(m+2,np.inf)]
 assert B.has_canonical_format and B.nnz==A.nnz+2
 sparse.save_npz(out/'equality.npz',B);np.savez_compressed(out/'arrays.npz',cost=q,equality_rhs=h,bounds=bounds)
 np.save(out/'original_row_ids.npy',ids);np.save(out/'source516_positions.npy',pos);np.save(out/'original_coordinate_slots.npy',np.arange(n,dtype=np.int64))
 meta=dict(source_shape=list(A.shape),source_nonzeros=int(A.nnz),source_equality_shape=list(E.shape),dual_primal_variables=m+2,dual_equality_shape=list(B.shape),dual_equality_nonzeros=int(B.nnz),dual_inequality_rows=0,nonnegative_row_multiplier_variables=m,scalar_upper_multiplier_column=m,scalar_lower_multiplier_column=m+1,original_coordinate_equality_marginal_slots=n,original_scalar_slot=j,original_native_bound_multiplier_variables=0,zero_old_equality_multiplier_variables=0,source_coefficients_rounded=0,new_transpose_coefficient_copies=int(A.nnz),new_bound_column_coefficients=2)
 (out/'layout.json').write_text(json.dumps(meta,indent=2));check();return meta
