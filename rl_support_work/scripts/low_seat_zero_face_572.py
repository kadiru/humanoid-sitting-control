"""Exact objective-zero face and its analytic lower-bound dual. No primal solve."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
N=78791;T=78790;K=331495;M=68250
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(
 scope='NEW sufficient-witness construction on the ORIGINAL PRIMARY objective-zero face. Exact structural preparation only. No numerical algorithm, LP, robot primal point, actual state, or interior-witness construction is executed.',
 rationale='The original primary cost is exactly t with original lower bound t>=0. Therefore any EXACT original-feasible point with t=0 attains the known lower bound. Search this specific exposed face directly, using hard linearized mechanics equalities instead of two-sided epigraph inequalities. This changes the search problem materially: it is a potentially empty proper subset, NOT an equivalent reformulation of the whole original LP or a claim that its optimum is zero.',
 distinction='Not another basis/row-group/definition/dual rename or20-row pruning retry. The new mathematical restriction is t=0, and ALL68250 relaxed mechanics pairs become exact affine equalities on that face. No496 inverse/residual substitution, no15-root, no558 definition variables, no552 inward bounds, no522 torque elimination, no538 scalar elimination, and no old internal/rejected vector. All78791 native-plus-scalar coordinates remain original.',
 source='Immutable570 PRIMARY subset, with571/547/517 implications inherited byhash. Native opposite-coefficient identities of all68250 consistency pairs are inherited from559 after exact546-source/hash/ID identity, not re-proved. Saved558/559 decisions include both10921 low-degree and57329 lifted pairs. No558 extension matrices are used. Actual472 INITIAL X/Y/J plus490 canonical auxiliary/exact drops remain unchanged.',
 equations='Keep every non-consistency570 inequality literally, with original cost t and original bounds. Append scalar equality t=0. For pair C_i*x-t<=b_i and -C_i*x-t<=-b_i replace BOTH inequalities by C_i*x=b_i, using copied stored native coefficients and the original positive RHS (not zeroing it). Keep all1366 original zero equality rows/RHS. NewE is nonzero. No variable added/removed, no native/scalar coordinate substitution or reconstruction.',
 exact_face_theorem='Given inherited opposite native coefficients, NEW exact checks require both original scalar coefficients=-1 and opposite RHS. At t=0 the two inequalities are equivalent to C_i*x=b_i. Conversely a face equality plus t=0 restores BOTH original signed inequalities. Thus the new feasible set is exactly the original real feasible set INTERSECT {t=0}, after inherited570/546/516 implications. This is not equality with the original whole feasible set. No existence, attainment, rank, consistency or feasibility claim.',
 finite='No new coordinates or coefficient products/divisions/rounding are introduced: coefficients are copied or scalar entries omitted under explicit t=0. Exact feasible binary64 points of the original ZERO FACE are preserved, including either signed scalar zero. No tolerance-feasible-set equivalence claimed. A future returned78791-vector stays bitwise unchanged; scalar must already equal zero. Never set/round/clip/repair/normalize it after the solver.',
 analytic_dual='Construct ONE NEW analytic original lower-bound dual: ALL inequality and equality duals zero, ALL upper duals zero, lower_t=1 and other lower duals zero. Original cost=e_t and lower_t bound0 imply c+A.T*y+E.T*nu+upper-lower=0 and dual objective0 EXACTLY. This is a genuine dual-feasible lower-bound vector only, NOT a primal-dual pair, primal witness, feasibility proof or full KKT evaluation. Its signed gap with any original primal w is w_t. The same dual is valid for the zero-face problem with new equality duals zero.',
 numerical_future='A future bounded primary face solver must require finite successful returned ORIGINAL coordinates, already-zero t, and independent FULLface/570/546/516/FULL490 inequality/equality/bound checks and required stationarity/absolute-gap<=1e-9 inside unchanged1e-7. The analytic dual may be used unchanged; backend marginals are not a substitute for full original primal checks. Numerical KKT does not establish exact feasibility/optimality or objective-error<=1e-9. No returned primal vector is evaluated in this preparation.',
 failure='Failure, infeasibility, no finite witness, uncertified point or deadline of a future zero-face attempt terminates that construction. It does NOT establish original LP or physical infeasibility, and does not authorize a sequence of nonzero epigraph faces, changed bounds/tolerances or restarted old solvers.',
 whole_path='Primary NEVER mapped. The SAME complete original direction must supply modeled phi, ACTUAL(mu+theta)/2 interior ceiling and prediction. Only then may a separately frozen actual interior LP and exact unscaled interval eta>0 precede immutable490 finite encoding; newX fresh mechanics/J and all signed/unit/normal/hover/geometry/filter/runtime/PPO gates. The old illustrative witness ceiling is not used. Changed actual state/trust requires valid fresh original assembly/input proofs inside unchanged45s total from parent preparation origin.',
 audit='Independent573 imports no producer/optimizer. Inherit559 native pair proofs by bound hashes; independently check NEW scalar/RHS face identities, native equality copies, all retained inequality/cost/domain/oldE/ID copies, and every analytic-dual sign/zero/coefficient identity and dual value. No endpoint/old implication/old signed native pair proof or candidate/model replay.',
 resources='ONE primary-only preparation and ONE independent audit, each240s total from before scientific imports,40GiB address-space cap, deadline checks between operations, peak unmeasured. Symbolic preparation only, not45s LP extension. No reference-witness zero-eta face: eta has a different objective and this theorem is specific to the primary epigraph.',
 limits='Fewer inequalities and more equalities do not establish speed, conditioning or a successful future solve. Exact face restriction may be infeasible even when original primary has positive feasible objective. All physical/controller criteria unchanged; no physical admission, motion, effort or PPO gain.',
 numerical_solver_exists=False,primal_witness_exists=False,physical_admission=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(t):
 if time.monotonic()-t>240:raise TimeoutError('240s zero-face symbolic preparation cap')
def build(out,t):
 src=D/'coupled_support_preparation_570/primary';A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'));ids=np.load(src/'original_row_ids.npy')
 positions546=np.load(src/'retained_546_positions.npy');positions516=np.load(src/'source_516_positions.npy')
 flags=np.load(D/'definition_audit_559/primary_decisions_verified.npy');decisions=np.load(D/'definition_preparation_558/primary/decisions.npy')
 assert flags.shape==(M,) and np.all(flags==1) and np.all((decisions==1)|(decisions==2))
 # This resolves already-certified row IDs; it does not re-decide558 eligibility or recompare opposite native coefficients.
 loc={int(v):i for i,v in enumerate(ids)};plus=np.array([loc[K+i] for i in range(M)],dtype=np.int64);minus=np.array([loc[K+M+i] for i in range(M)],dtype=np.int64)
 assert A.shape[1]==N and E.shape==(1366,N) and E.nnz==0 and not np.any(a['equality_rhs'])
 expected=np.zeros(N);expected[T]=1;assert np.array_equal(a['cost'],expected) and a['bounds'][T,0]==0 and np.isfinite(a['bounds'][T,1])
 assert np.isneginf(a['bounds'][:T,0]).all() and np.isposinf(a['bounds'][:T,1]).all()
 scalar_plus=A[plus,T].toarray().ravel();scalar_minus=A[minus,T].toarray().ravel();assert np.all(scalar_plus==-1) and np.all(scalar_minus==-1)
 assert np.array_equal(a['rhs'][plus],-a['rhs'][minus]);check(t)
 C=A[plus,:T].tocsr();new_rows=sparse.hstack([C,sparse.csr_matrix((M,1))],format='csr');scalar=sparse.csr_matrix(([1.],([0],[T])),shape=(1,N))
 EE=sparse.vstack([E,new_rows,scalar],format='csr');keep_mask=np.ones(A.shape[0],dtype=bool);keep_mask[plus]=False;keep_mask[minus]=False;keep=np.flatnonzero(keep_mask);B=A[keep].tocsr()
 sparse.save_npz(out/'matrix.npz',B);sparse.save_npz(out/'equality.npz',EE)
 np.savez_compressed(out/'arrays.npz',rhs=a['rhs'][keep],cost=a['cost'],bounds=a['bounds'],equality_rhs=np.r_[a['equality_rhs'],a['rhs'][plus],0.])
 np.save(out/'retained_570_positions.npy',keep);np.save(out/'source_546_positions.npy',positions546[keep]);np.save(out/'source_516_positions.npy',positions516[keep]);np.save(out/'original_row_ids.npy',ids[keep]);np.save(out/'equality_pair_570_positions.npy',np.c_[plus,minus]);np.save(out/'equality_original_positive_ids.npy',ids[plus])
 # This new vector is dual-feasible only. No primal coordinate is constructed or evaluated.
 full=D/'lifted_feasibility_preparation_490/primary';full_a=dict(np.load(full/'arrays.npz'));fullE=sparse.load_npz(full/'equality.npz')
 assert np.array_equal(full_a['cost'],a['cost']) and np.array_equal(full_a['bounds'],a['bounds'])
 lower=np.zeros(N);lower[T]=1
 np.savez_compressed(out/'analytic_lower_bound_dual.npz',lower=lower,upper=np.zeros(N),full490_inequality=np.zeros(len(full_a['rhs'])),full490_equality=np.zeros(fullE.shape[0]),face_inequality=np.zeros(B.shape[0]),face_equality=np.zeros(EE.shape[0]))
 layout=dict(original_scalar_column=T,original_columns=N,old_equality_rows=1366,consistency_equality_rows=M,scalar_zero_equality_row=1366+M,primary_only=True,source='570/primary',analytic_dual_value=0,analytic_dual_is_not_primal_witness=True,whole_original_LP_equivalence=False,exact_zero_face_equivalence=True,universal_finite_extension_needed=False,returned_coordinate_edit_allowed=False)
 (out/'layout.json').write_text(json.dumps(layout,indent=2));check(t)
 return dict(source_inequality_shape=[int(x) for x in A.shape],source_inequality_nonzeros=int(A.nnz),face_inequality_shape=[int(x) for x in B.shape],face_inequality_nonzeros=int(B.nnz),face_equality_shape=[int(x) for x in EE.shape],face_equality_nonzeros=int(EE.nnz),original_equality_rows=1366,inherited_opposite_native_pairs=M,new_scalar_coefficient_checks=2*M,new_opposite_RHS_checks=M,new_native_equality_coefficients_copied=int(C.nnz),inequalities_replaced_by_equalities=2*M,new_affine_consistency_equalities=M,new_scalar_zero_equalities=1,net_stored_nonzeros_removed=int(A.nnz-B.nnz-EE.nnz),original_variables_retained=N,variables_added=0,variables_removed=0,analytic_lower_bound_dual_vectors=1,original_cost_bounds_unchanged=True,exact_zero_face_equivalence=True,whole_original_feasible_set_equivalence=False,zero_face_feasibility_known=False,original_optimum_zero_established=False,native_opposite_pair_proofs_repeated=0,old_proofs_repeated=0)
