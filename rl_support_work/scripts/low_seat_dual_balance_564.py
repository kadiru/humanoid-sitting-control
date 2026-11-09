"""Exact dual balance formulation of saved native546 LPs; no numerical solve."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(
 scope='NEW exact LP dual balance formulation of the saved546 ORIGINAL native systems. Preparation only; no numerical algorithm, LP, robot witness or actual path.',
 source='Immutable546 ORIGINAL-coordinate matrix/RHS/cost/bounds and zero E;517/547 implications inherited by hash. No558 definition variables,552 finite-bound inward domain,522 eliminated torque,504 basis,532/542 internal state or rejected robot state. Actual472 INITIAL X/Y/J,490 canonical auxiliary and exact drops unchanged.',
 structural_change='Exchange original native-coordinate primal feasibility with nonnegative row-multiplier feasibility and78791 native stationarity balances. ALL original546 inequality multipliers become optimizer primal variables; original78791 robot-direction/scalar coordinates become the78791 equality-marginal slots. This is LP dualization, not a native coordinate basis, renamed row group/definition or backend change. It uses the saved whole-clock/command/local-mechanics incidence through its exact transpose and the zero-native-cost/unbounded-native-domain structure. It adds no actuator authority.',
 dimensions='Original native dimension78790 includes21840 auxiliary torque outputs; scalar at78790. Original native direct bounds unbounded and cost zero. Only original scalar lower/upper are finite. Original1366 E rows and RHS are identically zero. Thus no native bound multipliers or nontrivial old equality multipliers are needed. ALL original inequality rows including BOTH consistency signs/frozen units retain individual nonnegative dual variables; all original coordinates retain individual balance equations.',
 equations='Original P: min c.w, A.w<=b, l<=w<=u, E.w=d=0 with E=0. New D: min b.y+U*beta-L*alpha, A.T*y+e_scalar*beta-e_scalar*alpha=-c, (y,beta,alpha)>=0. New equality B=[A.T,+e_scalar,-e_scalar], RHS h=-c, cost q=[b,U,-L]. No inequality matrix. All coefficient/RHS/cost transfers copied or exactly sign-negated; no division, multiplication, endpoint construction or rounding.',
 weak_duality='For any feasible original w and new nonnegative p=(y,beta,alpha), c.w+q.p=y.(b-A.w)+beta*(U-w_scalar)+alpha*(w_scalar-L)>=0. Therefore -q.p is an original lower bound. No actual feasibility, attainment or objective equality is asserted without a certified primal-dual pair.',
 return_contract='Future solver must return status0/success/finite p AND finite equality marginals lambda. Recover ORIGINAL w by a BITWISE copy of scipy eqlin.marginals lambda, without negation, scalar/torque reconstruction, rounding, normalization, clipping or repair. These are backend equality marginals, NOT the new optimizer primal vector. y=p[:m], beta=p[m], alpha=p[m+1] unchanged; original native bound duals0, original zero-E duals0. Scalar upper/lower duals beta/alpha. New lower marginals s correspond to b-A.w, U-w_scalar, w_scalar-L; new upper marginals exactly0.',
 kkt_identity='Using the common stationarity convention cost+B.T*nu+upper-lower=0, new nu=-lambda=-w. New primal balance residual B.p+c is exactly original stationarity. New stationarity q-B.T*w-s=0 identifies original primal slacks, with s>=0. New signed gap q.p+h.nu equals original signed gap c.w+b.y+U*beta-L*alpha because h=-c and nu=-w. Conversely an exact original KKT pair supplies new nu=-w and lower s=[b-A.w,U-w_scalar,w_scalar-L]. This is a coefficient identity, not a computed primal/dual witness or an exact numerical-optimality claim.',
 finite_limit='A numerical equality marginal is an original-coordinate candidate only after independent finite/nonnegative/bitwise and exhaustive gates. New KKT does not substitute for full original KKT due to floating association/residual amplification. No finite-binary64 feasible-set equivalence or universally returned candidate is claimed. The original real domain is retained without552 inward bounds.',
 mandatory_checks='Future new D plus546 plus516 plusFULL490 primal/equality/bound violations,stationarity and absolute gap<=1e-9 inside unchanged1e-7; all original bounds/1366 equalities checked even though algebraically zero. Original546 row duals zero-embed through516 and490 IDs. No native/auxiliary/scalar candidate modification. Exact original unscaled interval eta>0 before490 encoder; primary NEVER mapped.',
 new_input='Changed actual state,trust or ACTUAL primary-derived ceiling requires fresh complete original assembly,valid quotient/box proof and new dual construction/independent proof before LP inside TOTAL45s from parent preparation origin. Saved reference witness ceiling.09895564659603738 remains illustrativeONLY. No old completed proof/assembly calculation repeated.',
 future='No566 numerical algorithm/child/full-path wrapper/sourcefreeze/budget/candidate exists. A future complete bounded method must handle a nonnegative multiplier primal of m+2 dimensions,78791 NONZERO equality balances,and robot candidates from equality marginals,not solver.x. Freeze independent return/actual-state auditor before any solve. Original45s,40GiB,120sparent,24major48LP120maps5dyadics,trust1..1/32 and all guards/gates unchanged.',
 audit='Independent565 imports no producer or optimizer; verifies exact transpose bijection per stored coefficient, both scalar-bound columns/signs/costs, all cost/RHS/domain/row-ID copies and symbolic original-primal/new-dual stationarity and signed-gap coefficient identities. No old547/517 proof, oldfixture, LP, model,root or finite-map replay.',
 resources=dict(preparation_wall_s=240,independent_audit_wall_s=240,address_space_GiB=40,checks_between_operations=True,peak_memory_measured=False,LP_budget_unchanged_s=45),
 performance='More optimizer primal variables and fewer equality rows than the original inequality count do not establish speed,conditioning,feasibility or deadline success.',physical_admission=False,new_candidate=False)
def check(start):
 if time.monotonic()-start>240:raise RuntimeError('240s symbolic preparation deadline; no continuation')
def derive(role,out,start):
 src=D/'box_support_preparation_546'/role
 A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'));ids=np.load(src/'original_row_ids.npy');pos=np.load(src/'retained_source_positions.npy')
 m,n=A.shape;j=n-1
 assert n==78791 and A.has_canonical_format and np.all(A.data!=0) and np.isfinite(A.data).all()
 assert E.shape==(1366,n) and E.nnz==0 and np.all(a['equality_rhs']==0)
 assert np.all(a['cost'][:j]==0) and a['cost'][j] in (-1.,1.)
 assert np.all(np.isneginf(a['bounds'][:j,0])) and np.all(np.isposinf(a['bounds'][:j,1]))
 L,U=map(float,a['bounds'][j]);assert np.isfinite([L,U]).all() and L<=U
 check(start)
 C=sparse.csr_matrix(([1.,-1.],([j,j],[0,1])),shape=(n,2))
 B=sparse.hstack([A.T,C],format='csr');q=np.r_[a['rhs'],U,-L];h=-a['cost'];bounds=np.c_[np.zeros(m+2),np.full(m+2,np.inf)]
 assert B.has_canonical_format and B.nnz==A.nnz+2
 sparse.save_npz(out/'equality.npz',B);np.savez_compressed(out/'arrays.npz',cost=q,equality_rhs=h,bounds=bounds)
 np.save(out/'original_row_ids.npy',ids);np.save(out/'source516_positions.npy',pos);np.save(out/'original_coordinate_slots.npy',np.arange(n,dtype=np.int64))
 meta=dict(source_shape=list(A.shape),source_nonzeros=int(A.nnz),source_equality_shape=list(E.shape),dual_primal_variables=m+2,dual_equality_shape=list(B.shape),dual_equality_nonzeros=int(B.nnz),dual_inequality_rows=0,nonnegative_row_multiplier_variables=m,scalar_upper_multiplier_column=m,scalar_lower_multiplier_column=m+1,original_coordinate_equality_marginal_slots=n,original_scalar_slot=j,original_native_bound_multiplier_variables=0,zero_old_equality_multiplier_variables=0,source_coefficients_rounded=0,new_transpose_coefficient_copies=int(A.nnz),new_bound_column_coefficients=2)
 (out/'layout.json').write_text(json.dumps(meta,indent=2));check(start);return meta
