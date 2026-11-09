"""Build the new-state interior LP from the accepted full input; no old proof replay."""
from common import *

def construct(base, arrays, state, soft, ceiling, interval, k, m):
 import numpy as np
 from scipy import sparse
 from fractions import Fraction as F
 n=base.shape[1]-1; first_closed=k+2*m
 count=len(interval['widths']); assert base.shape[0]==first_closed+2*count
 directions=base[:,:n].tocsr()
 top=sparse.hstack([directions,sparse.csr_matrix((base.shape[0],1))],format='csr')
 hn=directions[first_closed:first_closed+count]
 ones=sparse.csr_matrix(np.ones((count,1)))
 matrix=sparse.vstack([top,sparse.hstack([hn,ones]),sparse.hstack([-hn,ones])],format='csr')
 rhs=arrays['rhs'].copy()
 rhs[:k]=state['r'][soft['keep']]-state['guard'][soft['keep']]+soft['soft'][soft['keep']]*ceiling
 rhs[k:k+m]=-state['h']+ceiling;rhs[k+m:k+2*m]=state['h']+ceiling
 upper=[float(F(*u)/F(*w)) for u,w in zip(interval['upper'],interval['widths'])]
 lower=[-float(F(*l)/F(*w)) for l,w in zip(interval['lower'],interval['widths'])]
 rhs=np.r_[rhs,upper,lower]
 cost=np.r_[np.zeros(n),-1.]
 bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),.5]]
 return matrix,dict(rhs=rhs,cost=cost,bounds=bounds,equality_rhs=arrays['equality_rhs'].copy())

def main_input(origin):
 import numpy as np
 from scipy import sparse
 bind_predecessor(PRIMARY,origin,check)
 point=read(PRIMARY/'ceiling.json'); assert point['eligible']
 assert point['mu']==read(ACCEPTED/'decision.json')['summary']['phi']==0.08605330623059242
 assert point['ceiling']==(point['mu']+point['theta'])/2
 M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr()
 E=sparse.load_npz(PRIOR/'full_equality.npz').tocsr()
 arrays=dict(np.load(PRIOR/'full_arrays.npz',allow_pickle=False))
 state=dict(np.load(PRIOR/'state.npz',allow_pickle=False))
 rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz',allow_pickle=False))
 im=read(PRIOR/'intervals.json')
 assert M.shape==(652529,78791) and M.nnz==2614306
 assert E.shape==(1366,78791) and E.nnz==0 and im['trust']==1
 matrix,a=construct(M,arrays,state,rows,point['ceiling'],im,331495,68250)
 G=sparse.load_npz(PRIOR/'target_matrix.npz').tocsr()
 grhs=np.load(PRIOR/'target_rhs.npy',allow_pickle=False)
 assert G.shape==(2,78791) and G.nnz==12 and G[:,-1:].nnz==0
 matrix=sparse.vstack([matrix,G],format='csr');a['rhs']=np.r_[a['rhs'],grhs]
 assert matrix.shape==(837065,78791) and matrix.nnz<=3050586
 import common as c
 from reserve_bridge import append
 matrix,a=append(c,origin,matrix,a,interior=True)
 sparse.save_npz(O/'matrix.npz',matrix);sparse.save_npz(O/'equality.npz',E)
 np.savez(O/'arrays.npz',**a)
 write(O/'ceiling.json',dict(point,primary_sha256=sha(PRIMARY/'witness.npy'),primary_arithmetic_repeated=False))
 check(origin);return matrix,E,a

