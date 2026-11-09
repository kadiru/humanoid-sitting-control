"""Partial kinematic-residual coordinates with native command/torque/COM increments.

Saved-matrix preparation only. This is a new triangular coordinate map, not a
thresholded version of502, a mechanics projection, or a finite decoder.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
N0=10540;N=78790;BLOCK=50;STATES=1365
KIN=np.r_[np.arange(14),46]
AUX=np.array([i for i in range(50) if i not in set(KIN)])
SPEC=dict(
 scope='NEW partial residual/native auxiliary coordinate preparation. No optimizer, model, root, finite map, actual path or execution budget.',
 rationale='Full496 residual coordinates spread each native command/torque/COM auxiliary across all local residuals.502 localizes normal-gap incidence but still has23.7M primary coefficients. Keep35 native auxiliary increments perstate and transform only the15 rootxy/f/leg coordinates. This controls a different source of fill without deleting any constraint or coefficient from an old matrix.',
 seed='ONLY sealed472INITIAL actualX/Y/J and490canonical auxiliary, immutable original490 chains,496savedQ/L,497saved directed certificate, and503audited502 integer gap incidence. No old accepted/rejected/internal solver vector.',
 partition='The50 local outputs are split into15 kinematic outputs (feet12,moments2,gap1) and35 commands16/torques16/COM3. The50 local direct variables are15 rootxy/f/legs and35 native output auxiliaries. The15kinematic rows of L have EXACTzero coefficients on the35 nativeauxiliary columns.',
 inverse_inheritance='Take R=Q[root15,kin15] from immutable496. Since L[kin,aux35]=0 EXACTLY, (K*R-I15) is the kin/kin principal submatrix of (L*Q-I50), where K=L[kin,root15]. Inherit the existing497 outward per-entry intervals and full selected-row norm bounds (<1); no new inverse or repeated directed product certificate. Thus R is nonsingular. Nonsingularity ofQ alone would NOT justify its submatrix.',
 map='old_u=T_joint*new_u; old_root=-Rblock*Ckin0*T_joint*new_u +Rblock*zkin -Rblock*hkin; old_aux=new_aux. Same78790 coordinates:10540joint plus20475kinematic-residual plus47775nativeauxiliary. All206normal-drop bounds and endpoint contract remain.',
 invertibility='After grouping local slots, the new exact-real map is block triangular with invertible T_joint, certified R blocks, and identity native auxiliaries. Rounded lower-left products do not affect invertibility. This is a new basis definition, not dropping tiny entries from oldQ orSn.',
 shift='NEW shift acts only on15root variables perstate. It is not the old496a. newzero is not olddirectionzero. Original direction must be reconstructed BEFORE any shared dyadic scaling.',
 consistency='Every literal C*Snew and h+C*anew entry is retained. Both signs of ALL68250 normalized mechanics-consistency rows and frozen unit scales remain. Nativeauxiliary closure is fully coupled, not forced tozero, replaced by an ideal identity, or omitted. No posthoc15-root.',
 rows='All331888final/331495retained inequalities,156647HARD/noepigraph,174848rate/window,1411normal/hoverfinal,1366homogeneous equalities,11906fullcontrolboxes,80155featuretrust intervals,206dropbounds including pivot,27708old/support/window+1572bounds retained. Same392constantcontact+timeidentity omittedoptimizerONLY. Original440 +/-.02box and8eps/4ULP/32ULP/4302^-23reserves unchanged.',
 numerical='Sparse products are binary64; no bitwise equivalence to old transformed LPs is asserted. Future transformed AND reconstructed ORIGINAL LP witnesses must pass unchanged1e-7. Exact original interval eta>0 required before unchanged490 finite encoding. No returned-vector edits, coefficient threshold, clipping or normalization.',
 finite='New coordinates are continuous directions only. The existing205binary64drop/exactFractionpivot endpoint,468encoding,207Ugaptrust,2e-13+2Uactualgapguard and120-maploadbound remain separate. No finite map is executed/certified by this preparation.',
 reference='SavedJ/R are valid only at EXACT initialstate. ChangedX would need fresh mechanics/J and valid new local map/proof; no execution wrapper or resource/LP budget is designed here.',
 witness='Reference witness ceiling.09895564659603738 is an inherited assembly fixture, not a solved value. Both complete primary/witness matrices are preserved for arithmetic audit only.',
 scope_limits='Fewer stored coefficients if observed do not demonstrate solver speed or feasibility, explain prior timeouts, or admit actual commands/geometry. Off-consistency auxiliaries remain infeasible research variables. Original actual/physical/PPO gates remain.',
 model_calls=0,root_calls=0,ad_calls=0,local_inverse_solves=0,directed_inverse_product_proofs=0,lp_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def maximum(A):return float(np.max(abs(A.data))) if A.nnz else 0.
def construct(C,h,Q,T):
 kinrows=(np.arange(STATES)[:,None]*50+KIN).ravel()
 roots=(np.arange(STATES)[:,None]*50+np.arange(15)).ravel()
 auxiliaries=(np.arange(STATES)[:,None]*50+np.arange(15,50)).ravel()
 R=Q[:,:15,:][:,:,KIN].copy();RB=sparse.block_diag(list(R),format='csr')
 E=sparse.csr_matrix((np.ones(len(roots)),(roots,np.arange(len(roots)))),shape=(STATES*50,len(roots)))
 Iaux=sparse.csr_matrix((np.ones(len(auxiliaries)),(auxiliaries,auxiliaries)),shape=(STATES*50,STATES*50))
 T0=T[:N0,:N0].tocsr();U=(-RB@C[kinrows,:N0]).tocsr()
 S=sparse.bmat([[T0,sparse.csr_matrix((N0,STATES*50))],[E@(U@T0),(E@RB@E.T)+Iaux]],format='csr')
 a=np.r_[np.zeros(N0),E@(-RB@h[kinrows])]
 return S,a,R,kinrows,roots,auxiliaries
def substitute(A,rhs,S,a):
 V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr')
 return (A@V).tocsr(),rhs-A[:,:-1]@a
