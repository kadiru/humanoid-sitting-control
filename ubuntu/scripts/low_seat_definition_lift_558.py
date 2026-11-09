"""Exact common-mechanics expression lifting; no coordinate substitution, LP or model evaluation."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(
 scope='NEW exact extended formulation factoring each eligible saved pair of opposite native mechanics rows through ONE new definition variable. Preparation only; not a solver, robot witness or actual path.',
 source='Immutable546 ORIGINAL-coordinate source subset and original490 row IDs; inherit547/517 proofs byhash. ALL78791 source variables, INCLUDING21840tau auxiliaries and original scalar at78790, remain unchanged. New definition variables appended AFTER original scalar.',
 pair_policy='Only original490 consistency IDs k+i and k+m+i, k=331495,m=68250,i ascending, present together in546 are eligible. Verify identical native index pattern and EXACT opposite binary64 coefficients; scalar coefficients copied separately. Lift iff native length>=4, so net stored coefficient saving length-3 is positive. No hashing/regrouping by coefficient rays, forest, endpoint test, bound propagation or old proof replay.',
 primal='For paired a.x+gamma_plus*t<=bplus and -a.x+gamma_minus*t<=bminus introduce free z, equality a.x-z=0, and retain BOTH inequalities as z+gamma_plus*t<=bplus and -z+gamma_minus*t<=bminus. All RHS/scalar coefficients/other rows unchanged. Original E padded with zeros; definition RHS0, bounds(-inf,+inf),cost0.',
 exact_real_theorem='Every original real feasible w extends uniquely by z=a.x. Every extended feasible (w,z) projects to the identical w, since adding/subtracting the definition equality reconstructs each source row EXACTLY. Original costs/domains and objective retained. This is exact-real equality over stored binary64 coefficients without inversion,multiplication or coefficient rounding; no original variable is changed or removed.',
 finite_limit='The exact real z=a.x need not be binary64 even when w is. NO universal binary64 extension or equality of finite/tolerance-feasible sets is claimed. Any future returned original78791 coordinates must remain BITWISE unchanged; discard only appended definitions. Never reconstruct/repair original native/scalar values from z. Full extended,546,516,FULL490 primal/equality/bound/KKT gates remain mandatory.',
 dual='Let yplus,yminus be nonnegative paired inequality duals and nu the new equality dual for a.x-z=0. New-z stationarity is yplus-yminus-nu=0. When exact, nu=yplus-yminus and original native stationarity is identical after discarding nu and keeping all source inequality duals unchanged. Source equality/bound duals copied; appended free bounds have zero duals. All RHS and objective terms identical, so signed gaps agree. Finite z-stationarity errors can amplify through a; require full original stationarity and absolute gap<=1e-9, not inferred from the extended residual. No dual certificate computed here.',
 difference='This shares a repeated dense local mechanics expression through an explicit defining equality, reducing stored nonzeros while ADDING variables/equalities. It is not496 inverse residual-coordinate substitution,522 torque elimination,538 scalar elimination,552 inward-bound domain change,516 regrouping,536 forest or a backend/metric retry. Original546 unbounded-native domain is retained;552 finite bounds are not used.',
 audit='Independent559 imports no producer/solver. Check inheritedhashes, all eligible original-ID pair decisions, exact new equality coefficients, BOTH signed source-row reconstruction identities and dual column cancellation over Fractions, all untouched row/domain/cost/E/ID copies. No optimizer, old implication/endpoint/fixture/model/geometry replay.',
 future='No560 numerical algorithm/child/whole-path wrapper/budget/sourcefreeze/candidate exists. A complete method must handle actual nonzero E, appended variables and original scalar at78790, original-domain/KKT projection, actual primary-derived interior ceiling, all state branches and unchanged45s/caps before any LP. Old illustrative witness ceiling never actual. Exact interval eta>0 before immutable490 encoder; newX fresh mechanics/J and all actual/filter/runtime/72retention/PPO gates.',
 resources=dict(preparation_wall_s=240,independent_audit_wall_s=240,address_space_GiB=40,checks_between_operations=True,peak_memory_measured=False,LP_budget_unchanged_s=45),
 physical_admission=False,new_candidate=False)
def check(start):
 if time.monotonic()-start>240:raise RuntimeError('240s symbolic preparation deadline; no continuation')
def derive(role,out,start):
 src=D/'box_support_preparation_546'/role;A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));ids=np.load(src/'original_row_ids.npy');positions=np.load(src/'retained_source_positions.npy')
 n=A.shape[1];assert n==78791 and A.has_canonical_format and np.all(A.data!=0) and np.isfinite(A.data).all()
 assert len(set(map(int,ids)))==len(ids) and E.shape[1]==n and E.nnz==0
 loc={int(v):i for i,v in enumerate(ids)};pairs=[];decisions=np.zeros(68250,dtype=np.int8);scalar=[];terms=0
 for i in range(68250):
  check(start);p=loc.get(331495+i);q=loc.get(331495+68250+i)
  if p is None or q is None:continue
  ps,pe=A.indptr[p:p+2];qs,qe=A.indptr[q:q+2];pj=A.indices[ps:pe];pv=A.data[ps:pe];qj=A.indices[qs:qe];qv=A.data[qs:qe]
  pm=pj<n-1;qm=qj<n-1;assert np.array_equal(pj[pm],qj[qm]) and np.array_equal(pv[pm],-qv[qm]),'original consistency pair is not exact opposite'
  if pm.sum()<4:decisions[i]=1;continue
  decisions[i]=2;z=n+len(pairs);pairs.append((i,p,q,z,int(pm.sum())));terms+=int(pm.sum())
  scalar.append((float(pv[~pm][0]) if (~pm).any() else 0.,float(qv[~qm][0]) if (~qm).any() else 0.))
 pair=np.asarray(pairs,dtype=np.int64).reshape(-1,5);k=len(pair);assert k>0,'no positive storage-reduction certificate'
 rows=np.zeros(A.shape[0],dtype=bool);rows[pair[:,1]]=True;rows[pair[:,2]]=True;C=A.tocoo();keep=~rows[C.row]
 br=C.row[keep].tolist();bc=C.col[keep].tolist();bv=C.data[keep].tolist();er=[];ec=[];ev=[]
 for v,g in zip(pair,scalar):
  check(start);i,p,q,z,count=map(int,v);j=z-n
  for row,sgn,gamma in [(p,1.,g[0]),(q,-1.,g[1])]:
   br.append(row);bc.append(z);bv.append(sgn)
   if gamma:br.append(row);bc.append(n-1);bv.append(gamma)
  b,e=A.indptr[p:p+2]
  for col,val in zip(A.indices[b:e],A.data[b:e]):
   if col<n-1:er.append(j);ec.append(int(col));ev.append(float(val))
  er.append(j);ec.append(z);ev.append(-1.)
 B=sparse.csr_matrix((bv,(br,bc)),shape=(A.shape[0],n+k));F=sparse.csr_matrix((ev,(er,ec)),shape=(k,n+k));EE=sparse.vstack([sparse.hstack([E,sparse.csr_matrix((E.shape[0],k))]),F],format='csr')
 assert A.nnz+E.nnz-B.nnz-EE.nnz==terms-3*k and terms-3*k>0
 aa=dict(arr,cost=np.r_[arr['cost'],np.zeros(k)],bounds=np.vstack([arr['bounds'],np.tile([-np.inf,np.inf],(k,1))]),equality_rhs=np.r_[arr['equality_rhs'],np.zeros(k)])
 sparse.save_npz(out/'matrix.npz',B);sparse.save_npz(out/'equality.npz',EE);np.savez_compressed(out/'arrays.npz',**aa);np.save(out/'pairs.npy',pair);np.save(out/'decisions.npy',decisions);np.save(out/'original_row_ids.npy',ids);np.save(out/'source516_positions.npy',positions)
 return dict(source_shape=list(A.shape),source_inequality_nonzeros=int(A.nnz),source_equality_shape=list(E.shape),extended_shape=list(B.shape),extended_inequality_nonzeros=int(B.nnz),extended_equality_shape=list(EE.shape),extended_equality_nonzeros=int(EE.nnz),definition_variables=k,source_variables_retained=n,original_scalar_column=n-1,source_inequalities_retained=A.shape[0],definition_native_coefficients=terms,net_nonzeros_removed=terms-3*k,original_ID_pairs_considered=68250,missing_one_or_both_source_rows=int((decisions==0).sum()),present_low_degree_pairs_not_lifted=int((decisions==1).sum()),exact_opposite_pairs_lifted=k,real_projection_equivalence=True,finite_binary64_extension_guaranteed=False,source_coefficients_rounded=0)
