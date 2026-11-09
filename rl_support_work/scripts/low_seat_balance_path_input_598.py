"""NEW-input-only primary zero-face construction; saved572 is inherited byhash."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_problem(p):return sparse.load_npz(p/'matrix.npz').tocsr(),dict(np.load(p/'arrays.npz')),sparse.load_npz(p/'equality.npz').tocsr()
def same(A,B):
 A=A.tocsr();B=B.tocsr();return A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and A.data.tobytes()==B.data.tobytes()
def fresh_face(A,a,E,out,check):
 # All FULL original rows retained except the signed consistency pairs.
 # No quotient/box/coupled reduction or previous-state proof is used here.
 n=78791;t=78790;k=331495;m=68250;assert A.shape[1]==n and E.shape==(1366,n)
 p=np.arange(k,k+m);q=np.arange(k+m,k+2*m);C=A[p,:t].tocsr()
 assert same(C,-A[q,:t]) and np.all(A[p,t].toarray()==-1) and np.all(A[q,t].toarray()==-1)
 assert np.array_equal(a['rhs'][p],-a['rhs'][q]) and a['bounds'][t,0]==0
 check();keep=np.r_[np.arange(k),np.arange(k+2*m,A.shape[0])]
 B=A[keep].tocsr();H=sparse.vstack([E,sparse.hstack([C,sparse.csr_matrix((m,1))]),sparse.csr_matrix(([1.],([0],[t])),shape=(1,n))],format='csr')
 b=dict(a,rhs=a['rhs'][keep],equality_rhs=np.r_[a['equality_rhs'],a['rhs'][p],0.])
 sparse.save_npz(out/'matrix.npz',B);sparse.save_npz(out/'equality.npz',H);np.savez_compressed(out/'arrays.npz',**b);np.save(out/'retained_full_positions.npy',keep)
 check();return B,b,H
