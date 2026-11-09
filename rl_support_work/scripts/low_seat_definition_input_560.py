"""NEW input-only exact definition lifting; immutable reference558 is never replayed."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def derive_definition(src,out,check):
 A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));ids=np.load(src/'original_row_ids.npy');positions=np.load(src/'retained_source_positions.npy')
 n=A.shape[1];assert n==78791 and A.has_canonical_format and np.all(A.data!=0) and np.isfinite(A.data).all()
 assert len(set(map(int,ids)))==len(ids) and E.shape[1]==n and E.nnz==0
 loc={int(v):i for i,v in enumerate(ids)};pairs=[];decisions=np.zeros(68250,dtype=np.int8);scalar=[];terms=0
 for i in range(68250):
  check();p=loc.get(331495+i);q=loc.get(331495+68250+i)
  if p is None or q is None:continue
  ps,pe=A.indptr[p:p+2];qs,qe=A.indptr[q:q+2];pj=A.indices[ps:pe];pv=A.data[ps:pe];qj=A.indices[qs:qe];qv=A.data[qs:qe]
  pm=pj<n-1;qm=qj<n-1;assert np.array_equal(pj[pm],qj[qm]) and np.array_equal(pv[pm],-qv[qm]),'original consistency pair is not exact opposite'
  if pm.sum()<4:decisions[i]=1;continue
  decisions[i]=2;z=n+len(pairs);pairs.append((i,p,q,z,int(pm.sum())));terms+=int(pm.sum())
  scalar.append((float(pv[~pm][0]) if (~pm).any() else 0.,float(qv[~qm][0]) if (~qm).any() else 0.))
 pair=np.asarray(pairs,dtype=np.int64).reshape(-1,5);k=len(pair);assert k>0,'no positive storage-reduction certificate'
 rows=np.zeros(A.shape[0],dtype=bool);rows[pair[:,1]]=True;rows[pair[:,2]]=True;C=A.tocoo();keep=~rows[C.row]
 br=C.row[keep].tolist();bc=C.col[keep].tolist();bv=C.data[keep].tolist();er=[];ec=[];ev=[]
 for v,g in zip(pair,scalar):
  check();i,p,q,z,count=map(int,v);j=z-n
  for row,sgn,gamma in [(p,1.,g[0]),(q,-1.,g[1])]:
   br.append(row);bc.append(z);bv.append(sgn)
   if gamma:br.append(row);bc.append(n-1);bv.append(gamma)
  b,e=A.indptr[p:p+2]
  for col,val in zip(A.indices[b:e],A.data[b:e]):
   if col<n-1:er.append(j);ec.append(int(col));ev.append(float(val))
  er.append(j);ec.append(z);ev.append(-1.)
 B=sparse.csr_matrix((bv,(br,bc)),shape=(A.shape[0],n+k));F=sparse.csr_matrix((ev,(er,ec)),shape=(k,n+k));EE=sparse.vstack([sparse.hstack([E,sparse.csr_matrix((E.shape[0],k))]),F],format='csr')
 assert A.nnz+E.nnz-B.nnz-EE.nnz==terms-3*k and terms-3*k>0
 aa=dict(arr,cost=np.r_[arr['cost'],np.zeros(k)],bounds=np.vstack([arr['bounds'],np.tile([-np.inf,np.inf],(k,1))]),equality_rhs=np.r_[arr['equality_rhs'],np.zeros(k)])
 sparse.save_npz(out/'matrix.npz',B);sparse.save_npz(out/'equality.npz',EE);np.savez_compressed(out/'arrays.npz',**aa);np.save(out/'pairs.npy',pair);np.save(out/'decisions.npy',decisions);np.save(out/'original_row_ids.npy',ids);np.save(out/'source516_positions.npy',positions)
 return dict(source_shape=list(A.shape),source_inequality_nonzeros=int(A.nnz),source_equality_shape=list(E.shape),extended_shape=list(B.shape),extended_inequality_nonzeros=int(B.nnz),extended_equality_shape=list(EE.shape),extended_equality_nonzeros=int(EE.nnz),definition_variables=k,source_variables_retained=n,original_scalar_column=n-1,source_inequalities_retained=A.shape[0],definition_native_coefficients=terms,net_nonzeros_removed=terms-3*k,original_ID_pairs_considered=68250,missing_one_or_both_source_rows=int((decisions==0).sum()),present_low_degree_pairs_not_lifted=int((decisions==1).sum()),exact_opposite_pairs_lifted=k,real_projection_equivalence=True,finite_binary64_extension_guaranteed=False,source_coefficients_rounded=0)

from low_seat_box_input_548 import quotient,derive
