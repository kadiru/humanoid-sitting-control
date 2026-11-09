"""Independent association and coefficient-copy audit of the newly assembled input."""
from common import *

def identical(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def sparse_identical(a,b):
 a=a.tocsr();b=b.tocsr();assert a.shape==b.shape
 for n in ('indptr','indices','data'):identical(getattr(a,n),getattr(b,n))
def prove(origin):
 import numpy as np
 from scipy import sparse
 from fractions import Fraction as F
 frozen=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr();old=dict(np.load(PRIOR/'full_arrays.npz',allow_pickle=False))
 state=np.load(PRIOR/'state.npz',allow_pickle=False);rows=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz',allow_pickle=False)
 im=read(PRIOR/'intervals.json');point=read(PRIMARY/'ceiling.json');ceiling=point['ceiling']
 assert read(PRIMARY/'audit.json')['candidate_accepted'] and read(PRIMARY/'result.json')['execution_and_audit_passed']
 M=sparse.load_npz(O/'matrix.npz').tocsr();E=sparse.load_npz(O/'equality.npz').tocsr();a=dict(np.load(O/'arrays.npz',allow_pickle=False))
 n=78790;k=331495;m=68250;h=92267;start=k+2*m;end=652529
 assert M.shape==(993712,n+1) and M.nnz<=10412995
 sparse_identical(M[:end,:n],frozen[:,:n]);assert M[:end,n:].nnz==0
 sparse_identical(M[end:end+h,:n],frozen[start:start+h,:n])
 sparse_identical(M[end+h:end+2*h,:n],frozen[start+h:end,:n])
 tail=M[end:end+2*h,n:].tocoo();assert tail.nnz==2*h and np.array_equal(tail.row,np.arange(2*h))
 assert np.all(tail.col==0) and np.all(tail.data==1)
 sparse_identical(E,sparse.load_npz(PRIOR/'full_equality.npz'));assert E.nnz==0 and E.shape==(1366,n+1)
 keep=rows['keep'];soft=rows['soft'];assert int(keep.sum())==k and state['h'].shape==(m,)
 expected=np.concatenate((np.add(np.subtract(state['r'][keep],state['guard'][keep]),np.multiply(soft[keep],ceiling)),np.add(np.negative(state['h']),ceiling),np.add(state['h'],ceiling)))
 identical(a['rhs'][:start],expected);identical(a['rhs'][start:end],old['rhs'][start:end])
 for index in range(h):
  if index%256==0:check(origin)
  l=F(*im['lower'][index]);u=F(*im['upper'][index]);w=F(*im['widths'][index]);assert w>0 and u-l==w
  assert max(abs(v).bit_length() for key in ('lower','upper','widths') for v in im[key][index])<=4096
  assert a['rhs'][end+index]==float(u/w) and a['rhs'][end+h+index]==-float(l/w)
 assert np.array_equal(a['cost'],np.r_[np.zeros(n),-1.])
 from reserve_bridge import cache,sparse_same
 import common as c
 guards,guard_rhs,witness=cache(c,origin,interior=True)
 expected_bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),.5]]
 for col,w in witness.items():
  expected_bounds[col,0]=max(expected_bounds[col,0],w.lower)
  expected_bounds[col,1]=min(expected_bounds[col,1],w.upper)
 identical(a['bounds'],expected_bounds)
 assert np.array_equal(a['bounds'][-1],[0.,.5]);identical(a['equality_rhs'],old['equality_rhs'])
 assert np.all(a['equality_rhs']==0) and np.isfinite(a['rhs']).all() and np.isfinite(M.data).all()
 G=sparse.load_npz(PRIOR/'target_matrix.npz'); grhs=np.load(PRIOR/'target_rhs.npy',allow_pickle=False)
 sparse_identical(M[837063:837065],G); assert M[837063:837065,n:].nnz==0
 identical(a['rhs'][837063:837065],grhs)
 sparse_same(M[837065:],guards);identical(a['rhs'][837065:],guard_rhs)
 result=dict(target_rows_retained=True,passed=True,new_state=True,ceiling=ceiling,primary_point_arithmetic_repeated=False,old_proofs_replayed=False,all_original_rows_retained=True,added_interior_rows=184534,interval_rows=92267,files_sha256={n:sha(O/n) for n in ('matrix.npz','equality.npz','arrays.npz','ceiling.json')})
 write(O/'input_audit.json',result);return result
