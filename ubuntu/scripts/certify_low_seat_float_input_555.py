"""Independent finite-domain proof for genuinely NEW554 inputs; reference553 never replayed."""
from certify_low_seat_box_input_549 import prove as prove_box
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import struct,json
read=lambda p:json.loads(Path(p).read_text())
def neighbor(v,above):
 b=int.from_bytes(struct.pack('>d',v),'big')
 if v==0:b=1 if above else (1<<63)|1
 elif (v>0)==above:b+=1
 else:b-=1
 v=struct.unpack('>d',b.to_bytes(8,'big'))[0];assert np.isfinite(v);return F(v)
def prove_hull(box,hull,out,check):
 A=sparse.load_npz(box/'matrix.npz').tocsr();a=dict(np.load(box/'arrays.npz'));B=sparse.load_npz(hull/'matrix.npz').tocsr();b=dict(np.load(hull/'arrays.npz'));keep=np.load(hull/'retained546_positions.npy');rem=np.load(hull/'removed546_positions.npy');ep=read(box/'endpoints.json');bounds=b['bounds'];flags=np.zeros_like(bounds,dtype=np.uint8)
 assert np.all(np.isfinite(bounds)) and np.all(bounds[:,0]<=bounds[:,1])
 for j in range(len(bounds)):
  check();l,u=map(F,ep[j]);lo,hi=map(float,bounds[j]);assert neighbor(lo,False)<l<=F(lo)<=F(hi)<=u<neighbor(hi,True);flags[j,:]=1
 assert np.array_equal(keep,np.flatnonzero(np.diff(A.indptr)!=1)) and np.array_equal(rem,np.flatnonzero(np.diff(A.indptr)==1));uf=np.zeros(len(rem),np.uint8)
 for k,i in enumerate(rem):
  check();p=A.indptr[i];j=int(A.indices[p]);c=F(float(A.data[p]));assert c*F(float(bounds[j,int(c>0)]))<=F(float(a['rhs'][i]));uf[k]=1
 C=A[keep].tocsr();assert B.shape==C.shape and np.array_equal(B.indptr,C.indptr) and np.array_equal(B.indices,C.indices) and np.array_equal(B.data,C.data)
 assert np.array_equal(b['rhs'],a['rhs'][keep]) and np.array_equal(b['cost'],a['cost']) and np.array_equal(b['equality_rhs'],a['equality_rhs'])
 assert np.array_equal(np.load(hull/'bound_owners516.npy'),np.load(box/'bound_owners.npy')) and np.array_equal(np.load(hull/'source516_positions.npy'),np.load(box/'retained_source_positions.npy')[keep])
 np.save(out/'new_float_endpoints_verified.npy',flags);np.save(out/'new_float_unaries_verified.npy',uf)
 return dict(new_float_endpoint_proofs=int(flags.size),new_float_unary_proofs=len(uf),real_equivalence_claimed=False,finite_binary64_equivalence=True)
