"""Independent exact scaling/support check for genuinely new projected systems only."""
import numpy as np
from scipy import sparse

def certify(A,a,E,norm,local,separator,ptr):
 assert np.all(E.data==0) and np.all(a['equality_rhs']==0)
 A0=norm['original'];As=norm['matrix'];rp=norm['row_power'];cp=norm['column_power']
 assert np.array_equal(A0[:A.shape[0]].indptr,A.indptr) and np.array_equal(A0[:A.shape[0]].indices,A.indices) and np.array_equal(A0[:A.shape[0]].data,A.data)
 expected=[]
 for j,(lo,hi) in enumerate(a['bounds']):
  if np.isfinite(hi):expected.append((j,1,hi))
  if np.isfinite(lo):expected.append((j,-1,-lo))
 assert len(expected)+A.shape[0]==A0.shape[0]
 for k,(j,sign,rhs) in enumerate(expected):
  i=A.shape[0]+k;p,q=A0.indptr[i:i+2];assert q-p==1 and A0.indices[p]==j and A0.data[p]==sign and norm['original_rhs'][i]==rhs and j in set(separator)
 assert np.array_equal(norm['original_rhs'][:len(a['rhs'])],a['rhs']) and np.array_equal(norm['original_cost'],a['cost'])
 assert np.array_equal(As.indices,A0.indices) and np.array_equal(As.indptr,A0.indptr)
 def exact(v,w,power):
  assert np.all(np.isfinite(v)) and np.all(np.isfinite(w));nz=v!=0;assert np.array_equal(nz,w!=0)
  mv,ev=np.frexp(v[nz]);mw,ew=np.frexp(w[nz]);assert np.array_equal(mv,mw) and np.array_equal(ew-ev,np.broadcast_to(power,v.shape)[nz])
 rid=np.repeat(np.arange(A0.shape[0]),np.diff(A0.indptr));exact(A0.data,As.data,rp[rid]+cp[A0.indices]);exact(norm['original_rhs'],norm['rhs'],rp);exact(a['cost'],norm['cost'],cp)
 assert np.array_equal(np.sort(np.r_[local,separator]),np.arange(A.shape[1])) and len(np.unique(np.r_[local,separator]))==A.shape[1]
 owner=np.full(A.shape[1],-1,dtype=int)
 for k,(lo,hi) in enumerate(zip(ptr[:-1],ptr[1:])):owner[local[lo:hi]]=k
 # Independent min/max owner per row; no old partition classification or factor replay.
 lo=np.full(A0.shape[0],len(ptr));hi=np.full(A0.shape[0],-1);ids=owner[A0.indices];mask=(ids>=0)&(A0.data!=0)
 np.minimum.at(lo,rid[mask],ids[mask]);np.maximum.at(hi,rid[mask],ids[mask]);assert np.all((hi<0)|(lo==hi))
 return dict(passed=True,exact_dyadic_scaling=True,all_source_rows_and_bounds_retained=True,local_support_valid=True,new_system_only=True,old_proofs_repeated=False)
