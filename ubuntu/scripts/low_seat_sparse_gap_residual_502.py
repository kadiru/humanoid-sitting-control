"""NEW normal-gap incidence coordinates for the saved residual formulation.

Arithmetic preparation only: no LP, finite encoding, mechanics, root or solve.
The normal-drop increment basis is replaced by interior gap increments. Both
anchor increments are zero. The same exact drop endpoint contract is retained.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
N0=10540
N=78790
SPEC=dict(
 scope='NEW saved-matrix sparse normal-gap incidence preparation; no execution algorithm, LP, candidate or finite decoder.',
 motivation='498 residual-coordinate primary returned no witness and has32076557 nonzeros. Cumulative normal-drop incidence spreads a local gap through many coordinate columns. Replace that incidence with interior-gap increments before multiplying original rows. This is an algebraically justified new representation, not a retry, claimed timeout cause or demonstrated solver speedup.',
 seed='ONLY immutable490 canonical reference and EXACTsealed472INITIAL actualX/Y/J;496stored S/a and497nonsingularity proof inherited, no local inverse/root repeated. No old accepted/rejected path or internal solver vector.',
 coordinates='78790 coordinates remain. Only205 independent normal-drop coordinates become205 interior normal-gap increments; the other10335 old joint coordinates and68250 local residual coordinates retain their identities. This is NOT dimension reduction or extra physical authority.',
 difference='For207ordered normal states, set delta_g0=delta_g206=0 and delta_dj=delta_gj-delta_g(j+1). Store all independent drops except original fixedpivot203. Missing pivot is exactly minus their sum, giving the same206differences and exactly zero total increment.',
 inverse='Recover interior delta_gi as negative prefix sum of ALL drops including the dependentpivot. Integer coefficients {-1,0,1}; both205x205 products are exactly identity. No matrix inverse or old solve is performed.',
 composition='Form NEW stored Sn=Sold*T from the SAVED496transform, then compose each ORIGINAL490matrix with Sn. No old32M primary is rerun or selected. Every computed nonzero, including tiny literal closure defects, is retained. No thresholding or idealized closure replacement.',
 invertibility='Sn has upper-left integer invertible T0, lower-right unchanged blockQ, and zero upper-right. Its rounded lower-left block does not affect invertibility. Prior497 directed proof of each exact-binary-rational Q nonsingular is inherited; no second proof/solve. Formal exact-real affine substitution is bijective even though sparse products round.',
 numerical='The new binary64 row matrices are not asserted bitwise-equivalent to old496/498 binary64 products with a different association. A future returned direction must reconstruct ORIGINALt=Sn*z+a, without editing, and pass original UNTRANSFORMED LP at unchanged1e-7, plus exact unscaled positive interval certification before any immutable490 finite map.',
 line_search='Nonzero affine a remains; newzero is NOT olddirectionzero. Reconstruct original complete direction BEFORE dyadic scaling and recertify that same scaled direction. No direct finite gap-state assignment or new rounding contract is asserted here.',
 constraints='ALL331888final/331495retained rows,156647HARD/noepigraph and174848rate/window labels,68250two-sided normalizedconsistency rows,1411normal/hoverfinal and1366homogeneousequalities,11906fullboxes,80155featuretrust and206dropintervals includingpivot,27708old/support/window+1572bounds retained. Original440 +/- .02globalbox and all8eps/4ULP/32ULP/4302^-23reserves unchanged.',
 bounds='Normal-gap increments are free coordinates only inside the complete transformed original domain. No independent gap-only box substitutes for drop bounds, full phase/kinematic/planning/command/rate/target constraints or exact finite-domain certificate.',
 reference_only='SavedS andJ valid only for the exact inherited initial state. FuturechangedX requiresfreshmechanics/J and fresh valid residual transform; no posthoc15root, force-only solve, coefficient deletion or physical criterion change.',
 witness='Both primary and illustrative witness matrices transformed as arithmetic.490fixtureceiling.09895564659603738 is not a solved primary result. Futurewrapper must use actual primary reduction.',
 admission='No new actualX/Y/J, gap/drop/force target, complete path or scientific fixture. Sparse size reduction if observed is storage arithmetic only; no computational speed or feasibility claim. All original actual gates and physical/PPO restrictions remain.',
 model_calls=0,root_calls=0,ad_calls=0,local_inverse_solves=0,directed_inverse_certificates=0,lp_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def maximum(A):return float(np.max(abs(A.data))) if A.nnz else 0.
def incidence(layout):
 old=[(j,c['edge']) for j,c in enumerate(layout['columns']) if c['kind']=='normal_drop'];pivot=layout['pivot'];assert len(old)==205 and pivot==203
 cols=np.array([j for j,e in old]);edges=np.array([e for j,e in old]);assert len(set(edges))==205 and set(edges)==set(range(206))-{pivot}
 T=sparse.eye(N,format='lil');Ti=sparse.eye(N,format='lil')
 for row,e in old:
  T.rows[row]=[];T.data[row]=[]
  if e>0:T[row,int(cols[e-1])]=1.
  if e<205:T[row,int(cols[e])]=-1.
 # New gap i lives in slot cols[i-1], independent of the old edge at that slot.
 for i in range(1,206):
  row=int(cols[i-1]);Ti.rows[row]=[];Ti.data[row]=[]
  for col,e in old:
   v=-int(e<i)+int(pivot<i)
   if v:Ti[row,col]=float(v)
 return T.tocsr(),Ti.tocsr(),cols,edges

def substitute(A,rhs,S,a):
 V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr')
 return (A@V).tocsr(),rhs-A[:,:-1]@a
