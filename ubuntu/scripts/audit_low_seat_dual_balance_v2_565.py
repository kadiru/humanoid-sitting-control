"""Independent exact dual-balance and primal-recovery coefficient audit; no producer/optimizer import."""
from pathlib import Path
import json,hashlib,time,datetime,resource
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(start):
 if time.monotonic()-start>120:raise RuntimeError('120s remaining witness-only staged audit deadline; no continuation')
def bits(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def one(role,out,start):
 src=D/'box_support_preparation_546'/role;new=D/'dual_balance_preparation_564'/role
 A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'))
 B=sparse.load_npz(new/'equality.npz').tocsr();z=dict(np.load(new/'arrays.npz'));layout=read(new/'layout.json')
 m,n=A.shape;j=78790
 assert n==78791 and A.has_canonical_format and B.has_canonical_format
 assert B.shape==(n,m+2) and B.nnz==A.nnz+2 and np.isfinite(B.data).all() and np.all(B.data!=0)
 assert z['cost'].shape==(m+2,) and z['equality_rhs'].shape==(n,) and z['bounds'].shape==(m+2,2)
 assert E.shape==(1366,n) and E.nnz==0 and np.all(a['equality_rhs']==0)
 assert np.all(a['cost'][:j]==0) and abs(a['cost'][j])==1
 assert np.all(np.isneginf(a['bounds'][:j,0])) and np.all(np.isposinf(a['bounds'][:j,1]))
 L,U=a['bounds'][j];assert np.isfinite([L,U]).all() and L<=U
 assert np.all(z['bounds'][:,0]==0) and np.all(np.isposinf(z['bounds'][:,1]))
 bits(z['cost'][:m],a['rhs']);assert z['cost'][m]==U and z['cost'][m+1]==-L
 bits(z['equality_rhs'],-a['cost'])
 for dest,name in [('original_row_ids.npy','original_row_ids.npy'),('source516_positions.npy','retained_source_positions.npy')]:bits(np.load(new/dest),np.load(src/name))
 slots=np.load(new/'original_coordinate_slots.npy');assert np.array_equal(slots,np.arange(n,dtype=np.int64))
 assert layout['original_scalar_slot']==j and layout['scalar_upper_multiplier_column']==m and layout['scalar_lower_multiplier_column']==m+1
 # Every new balance row is exactly the original stationarity coefficient list.
 # No product/re-solve or old implication proof is used.
 C=A.tocsc();native_terms=0
 for k in range(n):
  if k%256==0:check(start)
  c0,c1=C.indptr[k:k+2];b0,b1=B.indptr[k:k+2];bi=B.indices[b0:b1];bv=B.data[b0:b1]
  retained=bi<m;assert np.array_equal(bi[retained],C.indices[c0:c1]);bits(bv[retained],C.data[c0:c1]);native_terms+=int(c1-c0)
  if k==j:
   assert np.array_equal(bi[~retained],[m,m+1]) and np.array_equal(bv[~retained],[1.,-1.])
  else:assert retained.all()
 # Independently view each multiplier column: q_i-B_i.T*w is exactly
 # the original row slack; upper/lower columns give U-t and t-L.
 T=B.tocsc();slack_terms=0
 for i in range(m):
  if i%256==0:check(start)
  a0,a1=A.indptr[i:i+2];t0,t1=T.indptr[i:i+2]
  assert np.array_equal(T.indices[t0:t1],A.indices[a0:a1]);bits(-T.data[t0:t1],-A.data[a0:a1]);slack_terms+=int(a1-a0)
 for i,sgn in [(m,1.),(m+1,-1.)]:
  t0,t1=T.indptr[i:i+2];assert np.array_equal(T.indices[t0:t1],[j]) and np.array_equal(T.data[t0:t1],[sgn])
 # Gap coefficient identities hold as exact dyadics, with nu=-w.
 assert np.array_equal(-z['equality_rhs'],a['cost'])
 assert np.array_equal(z['cost'][:m],a['rhs']) and z['cost'][m]==U and z['cost'][m+1]==-L
 assert native_terms==slack_terms==A.nnz
 np.save(out/(role+'_verified_original_coordinate_slots.npy'),slots)
 np.save(out/(role+'_verified_source_row_multiplier_slots.npy'),np.arange(m,dtype=np.int64))
 return dict(passed=True,dual_shape=list(B.shape),dual_nonzeros=B.nnz,original_stationarity_coefficient_identities=native_terms,original_slack_coefficient_identities=slack_terms,scalar_bound_column_identities=4,signed_gap_coordinate_coefficient_identities=n,signed_gap_multiplier_coefficient_identities=m+2,retained_copy_error_exact=0,original_primal_from_equality_marginals=True,optimizer_primal_is_multiplier_vector=True,original_domain_unchanged=True,candidate_evaluated=False)
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
 stage=read(D/'dual_balance_audit_stage_565.json');assert stage['original_actual_exit']==1 and stage['v2_compile_actual_exit']==0
 for p,h in stage['preserved_sha256'].items():assert sha(p)==h,p
 assert read(D/'dual_balance_preparation_exit_564.json')['actual_exit']==0
 launch=read(D/'dual_balance_launch_564.json')
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'dual_balance_preparation_564/SHA256.json').items():assert sha(D/'dual_balance_preparation_564'/p)==h,p
 out=D/'dual_balance_audit_565';assert out.is_dir() and not (out/'independent_audit.json').exists()
 # Primary one() returned successfully, including both saved verified arrays,
 # before the original print raised. Do NOT call one(primary) or replay it.
 saved=read(D/'dual_balance_preparation_564/preparation.json')['systems']['primary'];nnz=saved['source_nonzeros'];m=saved['source_shape'][0];n=saved['source_shape'][1]
 systems={'primary':dict(passed=True,dual_shape=saved['dual_equality_shape'],dual_nonzeros=saved['dual_equality_nonzeros'],original_stationarity_coefficient_identities=nnz,original_slack_coefficient_identities=nnz,scalar_bound_column_identities=4,signed_gap_coordinate_coefficient_identities=n,signed_gap_multiplier_coefficient_identities=m+2,retained_copy_error_exact=0,original_primal_from_equality_marginals=True,optimizer_primal_is_multiplier_vector=True,original_domain_unchanged=True,candidate_evaluated=False,inherited_completed_arithmetic=True,summary_reconstructed_from_frozen_control_flow_and_saved_dimensions=True,original_serialized_summary_available=False)}
 (out/'primary_completion_reconstruction.json').write_text(json.dumps(dict(scope='No primary arithmetic replay. Original failed traceback is at print AFTER one(primary) returned. Frozen source places every primary assertion before both verified-array saves. Numeric counts reconstructed from immutable source dimensions; original summary/timer were not serialized.',original_exit=read(D/'dual_balance_audit_original_exit_565.json'),primary=systems['primary']),indent=2))
 systems['witness']=one('witness',out,start);print(json.dumps(dict(role='witness',**systems['witness'])),flush=True)
 check(start);r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_pass=True,physical_pass=False,preparation_only=True,systems=systems,staged_witness_elapsed_s=time.monotonic()-start,original_monotonic_elapsed_s=None,original_log_filesystem_interval_s=stage['original_log_filesystem_interval_s'],remaining_stage_wall_limit_s=120,original_wall_limit_s=240,source_proofs_inherited=['517','547'],old_proofs_repeated=False,producer_imported=False,optimizer_imported=False,execution_recovery=True,recovery_scope='Original primary arithmetic completed once, then JSON print failed on numpy.int32. Preserve original source/log/actual1/two arrays. V2 reconstructs saved-primary summary from control-flow evidence and performs ONLY previously unexecuted witness arithmetic. No completed coefficient checks repeated.',completed_scientific_calculations_repeated=False,lp_calls=0,small_examples=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,new_candidate=False,actual_state_audit=False,limitation='Coefficient identities and original-domain provenance only. No primal-dual point or performance result. Primary original serialized summary/timer unavailable; completion evidenced by preserved arrays and return-before-print traceback. Future w must be BITWISE backend equality-marginal copy with independent full D/546/516/490 KKT and all physical gates.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(passed=True,staged_witness_elapsed_s=r['staged_witness_elapsed_s'])),flush=True)
if __name__=='__main__':main()
