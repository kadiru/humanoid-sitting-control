"""Independent NEW-input full assembly/face identity proof. No optimizer/producer import."""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy import sparse
read=lambda p:json.loads(Path(p).read_text())
def equal(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def csr(a,b):
 a=a.tocsr();b=b.tocsr();assert a.shape==b.shape
 assert np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices);equal(a.data,b.data)
def prove_full(folder,check):
 from audit_low_seat_graph_path_steps_583 import context,interval,matrix,rat
 check();d=context();S=folder.parent;state=dict(np.load(S/'state.npz'));im=read(S/'intervals.json');drops=list(map(rat,im['drops']));check()
 H,l,u,width,cl,cu,owners=interval(d,state['features'],drops,im['trust']);check()
 for name,value in [('lower',l),('upper',u),('widths',width),('closed_lower',cl),('closed_upper',cu)]:assert list(map(rat,im[name]))==value
 assert owners==im['owners'];csr(H,sparse.load_npz(S/'H.npz'));check()
 role=read(folder/'stage.json');M,rhs,cost,bounds,A,C,r,h,guard=matrix(d,state['features'],state['actualY'],state['J'],H,l,u,width,cl,cu,role['ceiling']);check()
 a=dict(np.load(folder/'arrays.npz'));csr(M,sparse.load_npz(folder/'matrix.npz'))
 for key,v in [('rhs',rhs),('cost',cost),('bounds',bounds)]:equal(v,a[key])
 csr(A,sparse.load_npz(S/'A.npz'));csr(C,sparse.load_npz(S/'C.npz'))
 for key,v in [('r',r),('h',h),('guard',guard)]:equal(v,state[key])
 E=sparse.load_npz(folder/'equality.npz');assert E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
 check();return dict(passed=True,new_full_assembly_proved=True,quotient_box_coupled_reductions_used=False,all_original_rows_retained=True)
def prove_face(A,a,E,B,b,H,out,check):
 n=78791;t=78790;k=331495;m=68250;p=np.arange(k,k+m);q=np.arange(k+m,k+2*m)
 assert A.shape[1]==n and H.shape==(E.shape[0]+m+1,n)
 equal(a['cost'],b['cost']);equal(a['bounds'],b['bounds']);assert np.count_nonzero(a['cost'])==1 and a['cost'][t]==1 and a['bounds'][t,0]==0
 verified=np.zeros(m,np.uint8)
 for i in range(m):
  if i%128==0:check()
  x=A.getrow(p[i]);y=A.getrow(q[i]);hi=H.getrow(E.shape[0]+i)
  xp=x.indices<t;yp=y.indices<t
  assert np.array_equal(x.indices[xp],y.indices[yp]) and np.array_equal(x.indices[xp],hi.indices)
  equal(x.data[xp],-y.data[yp]);equal(x.data[xp],hi.data)
  assert x.indices[-1]==y.indices[-1]==t and x.data[-1]==y.data[-1]==-1
  assert F(float(a['rhs'][p[i]]))+F(float(a['rhs'][q[i]]))==0
  assert F(float(b['equality_rhs'][E.shape[0]+i]))==F(float(a['rhs'][p[i]]));verified[i]=1
 keep=np.r_[np.arange(k),np.arange(k+2*m,A.shape[0])];csr(B,A[keep]);equal(b['rhs'],a['rhs'][keep]);csr(H[:E.shape[0]],E);equal(b['equality_rhs'][:E.shape[0]],a['equality_rhs'])
 last=H.getrow(H.shape[0]-1);assert last.nnz==1 and last.indices[0]==t and last.data[0]==1 and b['equality_rhs'][-1]==0
 np.save(out/'new_face_pairs_verified.npy',verified);check();return dict(passed=True,new_pair_identities=m,whole_original_equivalence=False,zero_face_equivalence=True)
