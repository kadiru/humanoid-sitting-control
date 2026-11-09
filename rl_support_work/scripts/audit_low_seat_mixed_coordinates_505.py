"""Independent partial-residual map assembly and inherited subblock proof audit.
No inverse solve, repeated directed product/integer fixture, LP, map or model.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');N=78790;N0=10540
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def mx(A):return float(np.max(abs(A.data))) if A.nnz else 0.
def main():
 assert read(D/'mixed_preparation_exit_504.json')['actual_exit']==0
 for p,h in read(D/'mixed_preparation_launch_504.json')['source_input_sha256'].items():assert sha(p)==h,p
 src=D/'mixed_coordinate_preparation_504';old=D/'lifted_feasibility_preparation_490';O=D/'mixed_coordinate_audit_505';O.mkdir(exist_ok=False)
 for p,h in read(src/'SHA256.json').items():assert sha(src/p)==h,p
 C=sparse.load_npz(old/'consistency_chain.npz').tocsr();r=dict(np.load(old/'reference.npz'));z=dict(np.load(src/'coordinates.npz'))
 q=dict(np.load(D/'residual_coordinate_preparation_496/local_blocks.npz'));proof=dict(np.load(D/'residual_coordinate_audit_497/directed_inverse_certificate.npz'))
 kin=np.array(list(range(14))+[46]);rr=(np.arange(1365)[:,None]*50+np.arange(15)).ravel();kk=(np.arange(1365)[:,None]*50+kin).ravel();aa=(np.arange(1365)[:,None]*50+np.arange(15,50)).ravel()
 # Defining zero block gives (LQ-I)[kin,kin] = K*R-I exactly as real
 # arithmetic. The saved full-row bound covers every selected subrow.
 assert np.all(q['L'][:,kin,15:]==0)
 R=np.stack([q['Q'][i,:15,kin].T for i in range(1365)])
 assert np.array_equal(R,z['R'])
 inh=dict(np.load(src/'inherited_subblock_certificate.npz'))
 assert np.array_equal(inh['lower'],proof['lower'][:,kin,:][:,:,kin]) and np.array_equal(inh['upper'],proof['upper'][:,kin,:][:,:,kin])
 assert np.array_equal(inh['row_norm_upper'],proof['row_norm_upper'][:,kin]) and np.all(inh['row_norm_upper']<1)
 # Independent state-local row assembly. No production construct import.
 T=sparse.load_npz(D/'sparse_gap_residual_preparation_502/gap_incidence.npz').tocsr()[:N0,:N0]
 rootblock=sparse.block_diag(list(R),format='csr');cross=(-rootblock@C[kk,:N0])@T
 selector=sparse.csr_matrix((np.ones(len(rr)),(rr,np.arange(len(rr)))),shape=(68250,len(rr)))
 native=sparse.csr_matrix((np.ones(len(aa)),(aa,aa)),shape=(68250,68250))
 expected=sparse.vstack([sparse.hstack([T,sparse.csr_matrix((N0,68250))]),sparse.hstack([selector@cross,selector@rootblock@selector.T+native])],format='csr')
 a=np.zeros(N);a[N0+rr]=-rootblock@r['h'][kk]
 S=sparse.load_npz(src/'coordinate_transform.npz');assert mx(S-expected)==0 and np.array_equal(a,z['shift'])
 assert mx(S[N0+aa]-sparse.eye(N,format='csr')[N0+aa])==0 and np.all(a[N0+aa]==0)
 checks={}
 for name in ['feature_basis','all_rows_chain','consistency_chain','interval_matrix','normal_drop_chain']:
  M=sparse.load_npz(old/(name+'.npz')).tocsr();actual=sparse.load_npz(src/(name+'.npz'));assert mx(actual-M@expected)==0
  assert np.array_equal(M@a,np.load(src/(name+'_shift.npz'))['shift']);checks[name]=dict(error=0.,rows=M.shape[0],new_nnz=actual.nnz,no_thresholding=True)
 for mode in ['primary','witness']:
  M=sparse.load_npz(old/mode/'matrix.npz');v=dict(np.load(old/mode/'arrays.npz'));out=dict(np.load(src/mode/'arrays.npz'));actual=sparse.load_npz(src/mode/'matrix.npz');V=sparse.block_diag([expected,sparse.csr_matrix([[1.]])],format='csr')
  assert mx(actual-M@V)==0 and np.array_equal(out['rhs'],v['rhs']-M[:,:-1]@a)
  for key in ['cost','bounds','equality_rhs']:assert np.array_equal(out[key],v[key])
  assert mx(sparse.load_npz(src/mode/'equality.npz')-sparse.load_npz(old/mode/'equality.npz'))==0
  checks[mode]=dict(error=0.,rhs_error=0.,shape=list(actual.shape),new_nnz=actual.nnz,cost_bounds_equalities_exact=True)
 prep=read(src/'preparation.json');out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,local_zero_kinematic_auxiliary_block_exact=True,inherited_R_norm_upper=float(inh['row_norm_upper'].max()),inherited_proof='K*R-I is the kin/kin principal submatrix of certified LQ-I because L[kin,aux]=0 exactly; selected full-row outward bounds imply norm<1. No old products or inverse repeated.',coordinate_transform_independent_rebuild_error=0.,native_auxiliary_identity_exact=True,new_shift_exact=True,checks=checks,literal_kinematic_identity_defect_retained=prep['literal_kinematic_identity_defect'],literal_auxiliary_consistency_shift_retained=prep['literal_auxiliary_shift_max'],all_original_rows_and_both_consistency_signs_retained=True,original_witness_and_exact_interval_checks_still_required=True,solver_speedup_demonstrated=False,actual_model_calls=0,local_inverse_calls=0,directed_inverse_product_proof_calls=0,integer_incidence_proof_repeats=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
