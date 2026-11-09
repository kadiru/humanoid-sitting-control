"""Saved-matrix local residual-coordinate substitution; no LP/model/map.

All original rows are multiplied by an invertible affine coordinate map.
The computed closure block is retained literally, including numerical defects;
it is never replaced by an ideal identity and no small entry is thresholded.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import sparse

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
N0=10540
BLOCK=50
SPEC=dict(
 scope='NEW saved-matrix residual-coordinate preparation, not a solver or candidate.',
 rationale='492 returned no primal witness. Re-express its local direct-state/auxiliary block in normalized mechanics-residual coordinates; this is not evidence of the timeout cause or promised speed improvement.',
 reference='Immutable490 canonical auxiliary features and EXACT sealed472 INITIAL actual X/Y/J. No484 or478 final/rejected point or492 internal solver state.',
 algebra='Write h+C0*u+L*v for the68250 normalized linearized consistency rows. L consists of1365 local50x50 blocks. Compute Q approximately inverse L. Define old u=new u, old v=-Q*C0*u+Q*z-Q*h. This affine map is bijective when every Q block is nonsingular; exact inverse L is NOT required for bijectivity.',
 structure='The new local solve uses50 independent residual right-hand sides perstate, not the old10-control nullspace/root/derivative fixture. It is saved-array linear algebra only.',
 defect='Retain literal C*S and h+C*a, including ALL nonzero roundoff terms. Do NOT replace with [0,I] or zero. Both consistency signs, epigraph coefficients and unit scales remain.',
 rows='Every original row and its order, hard/soft/keep labels, full control and feature intervals,206 drop bounds including pivot,1411 original primal equalities and1366 retained homogeneous tangent rows remain. All buffers and original440 +/- .02 domain unchanged.',
 dimensions='78790 variables remain; this is a local block coordinate substitution, NOT dimension reduction or removal of consistency residual freedom.',
 finite='No new finite map is executed or validated here. A future witness must first be reconstructed to the UNMODIFIED old-coordinate direction. Original untransformed LP residual <=1e-7 and original exact unscaled interval certificate eta>0 remain mandatory before immutable490 decoding.',
 numerical='Sparse products and reconstruction are floating arithmetic. Exact real affine substitution is the formal equivalence; binary64 matrices are not asserted bitwise-equivalent LPs. All original witness and actual nonlinear checks must reject any numerical discrepancy; no feasibility tolerance changes.',
 witness='The saved490 witness ceiling is an illustrative assembly fixture only. Both primary and illustrative witness assemblies are transformed without solving either. Future use needs actual primary predicted phi, not the fixture ceiling.',
 admission='Auxiliary/linear residual feasibility is not actual mechanics, geometry, force, command, quiet or physical admission. No posthoc15-root, clipping, normalization, dropped row or buffer waiver.',
 future='A new bounded execution algorithm, reconstruction certification and independent actual-step auditor still need design/freeze.492 remains complete and may not be restarted or extended.',
 model_calls=0,root_calls=0,ad_calls=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def maximum(m):return float(np.max(abs(m.data))) if m.nnz else 0.
def load():
 o=D/'lifted_feasibility_preparation_490'
 r=dict(np.load(o/'reference.npz'))
 C=sparse.load_npz(o/'consistency_chain.npz').tocsr()
 L=C[:,N0:].tocsr(); co=L.tocoo()
 assert np.all(co.row//BLOCK==co.col//BLOCK)
 local=np.zeros((len(r['active']),BLOCK,BLOCK))
 local[co.row//BLOCK,co.row%BLOCK,co.col%BLOCK]=co.data
 return o,r,C,local
def transform(C,local,h):
 Q=np.linalg.solve(local,np.broadcast_to(np.eye(BLOCK),local.shape))
 Qs=sparse.block_diag(list(Q),format='csr');C0=C[:,:N0]
 U=(-Qs@C0).tocsr()
 S=sparse.bmat([[sparse.eye(N0,format='csr'),None],[U,Qs]],format='csr')
 a=np.r_[np.zeros(N0),-Qs@h]
 return S,a,Q,Qs,U
def substitute(A,rhs,S,a):
 V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr')
 return (A@V).tocsr(),rhs-A[:,:-1]@a

