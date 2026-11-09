"""Independent zero-face data identities and analytic dual audit. No producer import."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def limit():
 if time.monotonic()-START>240:raise TimeoutError('240s independent zero-face audit cap')
def same(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 if a.dtype.kind=='f':assert np.array_equal(a.view(np.uint64),b.view(np.uint64))
 else:assert np.array_equal(a,b)
def csr_same(a,b):
 assert a.shape==b.shape;same(a.indptr,b.indptr);same(a.indices,b.indices);same(a.data,b.data)
def main():
 assert read(D/'zero_face_preparation_exit_572.json')['actual_exit']==0
 launch=read(D/'zero_face_launch_572.json')
 for p,h in launch['source_input_sha256'].items():assert digest(p)==h,p
 source=D/'coupled_support_preparation_570/primary';new=D/'zero_face_preparation_572'
 for p,h in read(new/'SHA256.json').items():assert digest(new/p)==h,p
 assert read(D/'definition_audit_559/independent_audit.json')['passed'] and read(D/'coupled_support_audit_571/independent_audit.json')['passed']
 flags=np.load(D/'definition_audit_559/primary_decisions_verified.npy');assert len(flags)==68250 and np.all(flags==1)
 # Native opposite-pair identities are inherited, not recomputed here. Inputs are bound through559's source freeze and571's retained-copy proof.
 old=read(D/'definition_launch_558.json')['source_input_sha256']
 for p,h in old.items():
  if '/box_support_preparation_546/' in p:assert launch['source_input_sha256'][p]==h
 A=sparse.load_npz(source/'matrix.npz').tocsr();E=sparse.load_npz(source/'equality.npz').tocsr();a=dict(np.load(source/'arrays.npz'));ids=np.load(source/'original_row_ids.npy')
 B=sparse.load_npz(new/'matrix.npz').tocsr();H=sparse.load_npz(new/'equality.npz').tocsr();b=dict(np.load(new/'arrays.npz'));layout=read(new/'layout.json')
 n=78791;t=78790;k=331495;m=68250;old_e=1366
 assert A.shape[1]==n and E.shape==(old_e,n) and E.nnz==0 and H.shape==(old_e+m+1,n)
 # Reference resolution only; no558 degree selection or native a/-a proof is repeated.
 order=np.argsort(ids);sorted_ids=ids[order];positive_ids=np.arange(k,k+m);negative_ids=np.arange(k+m,k+2*m)
 pp=np.searchsorted(sorted_ids,positive_ids);qq=np.searchsorted(sorted_ids,negative_ids);assert np.array_equal(sorted_ids[pp],positive_ids) and np.array_equal(sorted_ids[qq],negative_ids)
 plus=order[pp];minus=order[qq];same(np.c_[plus,minus],np.load(new/'equality_pair_570_positions.npy'));same(positive_ids,np.load(new/'equality_original_positive_ids.npy'))
 positive_flags=np.zeros(m,dtype=np.uint8);rhs_flags=np.zeros(m,dtype=np.uint8)
 for j,(p,q) in enumerate(zip(plus,minus)):
  if j%128==0:limit()
  assert F(float(A[p,t]))==-1 and F(float(A[q,t]))==-1
  assert F(float(a['rhs'][p]))+F(float(a['rhs'][q]))==0
  assert F(float(b['equality_rhs'][old_e+j]))==F(float(a['rhs'][p]));positive_flags[j]=1;rhs_flags[j]=1
 csr_same(H[:old_e].tocsr(),E)
 C=A[plus,:t].tocsr();csr_same(H[old_e:old_e+m,:t].tocsr(),C);assert H[old_e:old_e+m,t:].nnz==0
 last=H.getrow(old_e+m);assert last.nnz==1 and last.indices[0]==t and F(float(last.data[0]))==1 and b['equality_rhs'][-1]==0
 same(a['equality_rhs'],b['equality_rhs'][:old_e]);same(a['rhs'][plus],b['equality_rhs'][old_e:old_e+m])
 mask=np.ones(A.shape[0],bool);mask[plus]=False;mask[minus]=False;keep=np.flatnonzero(mask)
 same(keep,np.load(new/'retained_570_positions.npy'));csr_same(B,A[keep].tocsr());same(a['rhs'][keep],b['rhs']);same(ids[keep],np.load(new/'original_row_ids.npy'))
 same(np.load(source/'retained_546_positions.npy')[keep],np.load(new/'source_546_positions.npy'));same(np.load(source/'source_516_positions.npy')[keep],np.load(new/'source_516_positions.npy'))
 same(a['cost'],b['cost']);same(a['bounds'],b['bounds']);assert set(a)==set(b)
 assert layout['original_scalar_column']==t and layout['scalar_zero_equality_row']==old_e+m and layout['exact_zero_face_equivalence'] and not layout['whole_original_LP_equivalence']
 full=D/'lifted_feasibility_preparation_490/primary';fa=dict(np.load(full/'arrays.npz'));fE=sparse.load_npz(full/'equality.npz');same(fa['cost'],a['cost']);same(fa['bounds'],a['bounds'])
 dual=dict(np.load(new/'analytic_lower_bound_dual.npz'));assert set(dual)=={'lower','upper','full490_inequality','full490_equality','face_inequality','face_equality'}
 assert dual['full490_inequality'].shape==(len(fa['rhs']),) and dual['full490_equality'].shape==(fE.shape[0],) and dual['face_inequality'].shape==(B.shape[0],) and dual['face_equality'].shape==(H.shape[0],)
 for key in ['upper','full490_inequality','full490_equality','face_inequality','face_equality']:assert np.all(dual[key]==0)
 assert dual['lower'].shape==(n,) and dual['upper'].shape==(n,) and np.isfinite(dual['lower']).all() and np.all(dual['lower']>=0)
 coefficient_flags=np.zeros(n,dtype=np.uint8)
 for j in range(n):
  if j%512==0:limit()
  expected=F(1) if j==t else F(0)
  assert F(float(a['cost'][j]))==expected and F(float(dual['lower'][j]))==expected
  assert F(float(a['cost'][j]))+F(float(dual['upper'][j]))-F(float(dual['lower'][j]))==0
  coefficient_flags[j]=1
 assert a['bounds'][t,0]==0 and np.isfinite(a['bounds'][t,1])
 assert np.isneginf(a['bounds'][:t,0]).all() and np.isposinf(a['bounds'][:t,1]).all()
 assert np.all(dual['lower'][:t]==0) and np.all(dual['upper']==0)
 # Row/equality duals are exactly zero, so their coefficient/RHS contributions are symbolically zero. Do not multiply infinity by zero.
 dual_value=F(float(a['bounds'][t,0]))*F(float(dual['lower'][t]))-F(float(a['bounds'][t,1]))*F(float(dual['upper'][t]));assert dual_value==0
 summary=dict(source_inequality_shape=[int(x) for x in A.shape],source_inequality_nonzeros=int(A.nnz),face_inequality_shape=[int(x) for x in B.shape],face_inequality_nonzeros=int(B.nnz),face_equality_shape=[int(x) for x in H.shape],face_equality_nonzeros=int(H.nnz),original_equality_rows=old_e,inherited_opposite_native_pairs=m,new_scalar_coefficient_checks=2*m,new_opposite_RHS_checks=m,new_native_equality_coefficients_copied=int(C.nnz),inequalities_replaced_by_equalities=2*m,new_affine_consistency_equalities=m,new_scalar_zero_equalities=1,net_stored_nonzeros_removed=int(A.nnz-B.nnz-H.nnz),original_variables_retained=n,variables_added=0,variables_removed=0,analytic_lower_bound_dual_vectors=1,original_cost_bounds_unchanged=True,exact_zero_face_equivalence=True,whole_original_feasible_set_equivalence=False,zero_face_feasibility_known=False,original_optimum_zero_established=False,native_opposite_pair_proofs_repeated=0,old_proofs_repeated=0)
 assert summary==read(new/'preparation.json')['primary'];limit()
 out=D/'zero_face_audit_573';out.mkdir(exist_ok=False);np.save(out/'face_scalar_coefficients_verified.npy',positive_flags);np.save(out/'face_RHS_verified.npy',rhs_flags);np.save(out/'analytic_dual_stationarity_verified.npy',coefficient_flags);np.save(out/'retained_570_positions_verified.npy',keep)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,preparation_only=True,primary=summary,new_exact_analytic_dual_coordinate_identities=n,new_finite_bound_dual_products=2,analytic_dual_value=str(dual_value),exact_copy_error=0,elapsed_s=time.monotonic()-START,producer_import=False,numerical_optimizer_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,primal_vectors_constructed_or_evaluated=0,full_primal_dual_KKT_evaluations=0,physical_admission=False,execution_recovery=False,old_proofs_replayed=False,scope='Exact zero-face identity/copy audit plus one analytic lower-bound dual certificate. No primal feasibility, optimum-zero, rank, convergence or physical claim.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
