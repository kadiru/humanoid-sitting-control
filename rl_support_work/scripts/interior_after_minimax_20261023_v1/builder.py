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
 summary=read(PRIMARY/'summary.json');audit=read(PRIMARY/'audit.json');result=read(PRIMARY/'result.json')
 assert summary['candidate_accepted'] and audit['passed'] and audit['candidate_accepted'] and result['candidate_accepted']
 assert not summary['physical_success'] and not audit['physical_admission']
 ceiling=read(PRIMARY/'ceiling.json')
 assert ceiling['eligible'] and ceiling['mu']==audit['mu'] and ceiling['theta']==audit['theta']
 assert ceiling['ceiling']==summary['subsequent_interior_ceiling']==0.061901382259917585
 proof=read(PRIOR/'input_proof.json')
 assert proof['passed'] and proof['new_state_full_input_proved'] and not proof['old_state_reductions_used']
 assert read(ACCEPTED/'audit.json')['passed'] and read(ACCEPTED/'result.json')['trial_merit_accepted']
 for name in ('full_matrix.npz','full_arrays.npz','full_equality.npz','state.npz','A.npz','C.npz','H.npz','intervals.json'):
  assert sha(PRIOR/name)==proof['file_sha256'][name],name
  check(origin)
 M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr();E=sparse.load_npz(PRIOR/'full_equality.npz').tocsr()
 assert M.shape==(652529,78791) and M.nnz==2614306 and E.shape==(1366,78791) and E.nnz==0
 arrays=dict(np.load(PRIOR/'full_arrays.npz',allow_pickle=False));state=dict(np.load(PRIOR/'state.npz',allow_pickle=False))
 rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz',allow_pickle=False));im=read(PRIOR/'intervals.json')
 assert im['trust']==1 and len(im['widths'])==92267
 out,a=construct(M,arrays,state,rows,ceiling['ceiling'],im,331495,68250)
 assert out.shape==(837063,78791) and out.nnz<=3050574
 sparse.save_npz(O/'matrix.npz',out);sparse.save_npz(O/'equality.npz',E);np.savez(O/'arrays.npz',**a)
 write(O/'ceiling.json',dict(ceiling,primary_sha256=sha(PRIMARY/'witness.npy'),primary_arithmetic_repeated=False))
 check(origin)
 return out,E,a
