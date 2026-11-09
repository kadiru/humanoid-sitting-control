"""Saved-array preparation for lifted state/command collocation, not a solver.

Every original row acts on the SAME 75 features [X24,Y51]. X and Y are now
independent optimization variables until Y=Mechanics(X) is satisfied. Thus
command/rate/target/planning rows are affine; auxiliary feasibility alone has
NO physical meaning. No root retraction is used or executed here.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
DEP=np.array([0,1,2]+list(range(4,16)))
IND=np.array([3]+list(range(16,24)))
EQS=np.r_[np.arange(14),47]
LOCAL_CONTROLS=np.array([71,3]+list(range(16,24)))
EXTRA=np.r_[DEP,24+np.array([i for i in range(51) if i!=47])]
ALGEBRA_BOUND=2e-10
SPEC=dict(
    scope='New lifted feature-space transcription; preparation only, no executable optimizer or candidate.',
    seed='EXACT sealed472 INITIAL X/Y/J. No484 final or rejected trial initializes it.',
    features='Unchanged75 features per state: X24 followed by all51 auxiliary mechanics outputs.',
    variables='Existing10540 g/p/conserved-load/exact-drop coordinates plus65 independent features at each1365 nonfixed state;99265 total.',
    extra_features='15 rootxy/f/leg coordinates plus50 auxiliary outputs other than gap. Commands and torques are independent auxiliaries until consistency is satisfied.',
    closure='51 nonlinear equations auxiliaryY-Mechanics(X)=0 and14 affine auxiliary foot/moment equations=0 at each nonfixed state;88725 equations.',
    closure_units='Feetposition m; orientation0.1*vee rad-scaled; horizontalmoments normalizedBW*m; normal normalizedBW; command rad; torque Nm; gap andCOM m. Mixed residuals are not a physical severity ranking.',
    original_equalities='All1411 original normal/hover equations remain separately visible. Their lifted tangents are NOT zero; after closure/nullspace reduction they recover the old analytic zero tangents. Original absolute-normal1e-13 and normalized-hover1e-8 primal criteria remain. Redundant conservation equations require explicit numerical treatment by any future solver, not impossible restoration of an admissible roundoff residual.',
    hard_rows='All331888 final rows and27708 original/support/window plus1572 old coordinate bounds retained. Same331495 optimizer keep mask,156647 hard and174848 rate/window soft labels. No hard slack introduced by this preparation.',
    domain='Original440 +/- .02 full11906 f/p/load box;206 normal-drop bounds including exact pivot;45 fixed states, inactive exactzero loads, aliases and all labels remain. Added feature variables do not expand physical authority.',
    finite_encoding='No finite lifted decoder exists yet. Existing exact-drop and conserved-load encodings are not rerun. A future decoder must preserve all original bounds, exact endpoints, fixed/alias/inactive states and the load rounding contract; this linear basis is not a quantized-map derivative claim.',
    nonlinear_policy='An off-consistency lifted point is an INFEASIBLE research iterate, never a complete actual path or command schedule. Every admitted final must freshly recompute actual mechanics, all original/buffered rows, geometry and independent work. Auxiliary row success or small aggregate closure is insufficient; actual signed criteria may not be waived.',
    retained_buffers='8eps contact/phase/support;4ULP per20ms rates;32ULP targets;430 additional2^-23 buffer;normal-gap2e-13+2*2^-59 guard unchanged.',
    numerical_scope='Saved Jacobian algebra at the exact inherited472 initial state only; no FK,AD,root,finite-work,LP,trajectory optimizer,filter or physics.',
    algebra_comparison_bound=ALGEBRA_BOUND,
    solver_designed=False,solver_budget_frozen=False,physical_admission=False)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def load_saved():
    seed=dict(np.load(D/'free_gap_path_472/initial.npz'))
    mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
    rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
    eq=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz'))
    cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'))
    return dict(seed=seed,mesh=mesh,rows=rows,eq=eq,cm=cm,
        M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz').tocsr(),
        E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz').tocsr(),
        B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsr(),
        C=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/original_control_box_matrix.npz').tocsr(),
        gap_layout=read(D/'gap_coordinate_preparation_v3_466/layout.json'))

def lift_basis(d):
    K=len(d['seed']['X']);B=d['B'].tocoo();active=np.flatnonzero(~d['mesh']['fixed']);n0=d['B'].shape[1]
    rr=(B.row//10)*75+LOCAL_CONTROLS[B.row%10];cc=B.col;vv=B.data
    extra_rows=(active[:,None]*75+EXTRA[None,:]).ravel()
    P=sparse.csr_matrix((np.r_[vv,np.ones(len(extra_rows))],
        (np.r_[rr,extra_rows],np.r_[cc,n0+np.arange(len(extra_rows))])),shape=(K*75,n0+len(extra_rows)))
    return P,active,extra_rows

def closure_blocks(J):
    K=len(J);L=np.zeros((K,65,75));L[:,:51,:24]=-J;L[:,:51,24:]=np.eye(51)
    L[:,51:,24:38]=np.eye(14)
    return L

def reduced_null(J):
    # NEW elimination of this lifted65-equation block, using only saved J.
    # The15x15 Schur block is the old free-gap kinematic block, not a root call.
    A=J[:,EQS][:,:,DEP];rhs=np.concatenate([np.tile(np.eye(15)[-1],(len(J),1))[:,:,None],-J[:,EQS][:,:,IND]],axis=2)
    T=np.zeros((len(J),24,10));T[:,DEP,:]=np.linalg.solve(A,rhs);T[:,IND,1:]=np.eye(9)
    raw=np.einsum('nij,njk->nik',J,T);G=np.concatenate([T,raw],axis=1)
    # Analytic defining identities, not dropping small matrix entries.
    G[:,24:38,:]=0.;G[:,71,:]=0.;G[:,71,0]=1.
    return T,G,A,rhs,raw

def max_sparse(m):return float(np.max(np.abs(m.data))) if m.nnz else 0.
