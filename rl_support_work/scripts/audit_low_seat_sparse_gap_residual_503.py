"""Independent NEW integer incidence and whole-row sparse-basis audit.
No repeated local inverse, directed inverse proof, old vector fixture, finite
map, LP, mechanics, derivative, root or virtual-work evaluation.
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
 assert read(D/'sparse_gap_preparation_exit_502.json')['actual_exit']==0
 for p,h in read(D/'sparse_gap_preparation_launch_502.json')['source_input_sha256'].items():assert sha(p)==h,p
 src=D/'sparse_gap_residual_preparation_502';old=D/'lifted_feasibility_preparation_490';prior=D/'residual_coordinate_preparation_496';O=D/'sparse_gap_residual_audit_503';O.mkdir(exist_ok=False)
 for p,h in read(src/'SHA256.json').items():assert sha(src/p)==h,p
 layout=read(D/'gap_coordinate_preparation_v3_466/layout.json');col={c['edge']:j for j,c in enumerate(layout['columns']) if c['kind']=='normal_drop'};slots=np.array(list(col.values()));pivot=layout['pivot'];assert pivot==203 and len(col)==205
 # Independent oriented edge/vertex incidence and inverse prefix sums.
 edge=sparse.diags([np.ones(206),-np.ones(206)],[0,1],shape=(206,207),format='csr')
 H=edge[:,1:206];Tlocal=H[np.array(list(col))]
 prefix=-sparse.tril(sparse.csr_matrix(np.ones((205,206))),k=0,format='csr')
 dependent=sparse.lil_matrix((206,205))
 for j,e in enumerate(col):dependent[e,j]=1.;dependent[pivot,j]=-1.
 inverse=prefix@dependent.tocsr()
 I=sparse.eye(205,format='csr');assert mx(Tlocal@inverse-I)==mx(inverse@Tlocal-I)==0
 T=sparse.block_diag([sparse.eye(int(slots[0])),Tlocal,sparse.eye(N-N0)],format='csr')
 Ti=sparse.block_diag([sparse.eye(int(slots[0])),inverse,sparse.eye(N-N0)],format='csr')
 assert np.array_equal(slots,np.arange(10335,10540)) and mx(T-sparse.load_npz(src/'gap_incidence.npz'))==0 and mx(Ti-sparse.load_npz(src/'gap_prefix_inverse.npz'))==0
 Sold=sparse.load_npz(prior/'coordinate_transform.npz').tocsc();zero=sparse.csc_matrix((N,1));newcols=[]
 for i in range(1,206):newcols.append((Sold[:,col[i]] if i in col else zero)-(Sold[:,col[i-1]] if i-1 in col else zero))
 S=sparse.hstack([Sold[:,:10335]]+newcols+[Sold[:,N0:]],format='csr');savedS=sparse.load_npz(src/'coordinate_transform.npz');assert mx(S-savedS)==0
 assert S[:N0,N0:].nnz==0 and mx(S[N0:,N0:]-Sold[N0:,N0:])==0
 a=np.load(src/'coordinates.npz')['shift'];assert np.array_equal(a,np.load(prior/'shift.npz')['a'])
 proof=read(D/'residual_coordinate_audit_497/independent_audit.json');assert proof['passed'] and proof['directed_inverse_norm_upper_max']<1
 # No inverse solve/proof repeat. New block triangular S is invertible by
 # NEW exact integer Tlocal and unchanged previously certified Q blocks.
 checks={}
 for name in ['feature_basis','all_rows_chain','consistency_chain','interval_matrix','normal_drop_chain']:
  M=sparse.load_npz(old/(name+'.npz')).tocsr();actual=sparse.load_npz(src/(name+'.npz'));expected=M@S
  assert mx(actual-expected)==0 and np.array_equal(M@a,np.load(src/(name+'_shift.npz'))['shift'])
  checks[name]=dict(error=0.,rows=M.shape[0],new_nnz=actual.nnz,no_thresholding=True)
 P=sparse.load_npz(src/'feature_basis.npz');normal=np.array(layout['normal_states'])*75+71;G=P[normal][:,slots];wanted=sparse.vstack([sparse.csr_matrix((1,205)),sparse.eye(205),sparse.csr_matrix((1,205))],format='csr')
 assert mx(G-wanted)==0 and mx(sparse.load_npz(src/'normal_drop_chain.npz')[:,slots]-H)==0
 assert mx(H.sum(axis=0) if sparse.issparse(H.sum(axis=0)) else sparse.csr_matrix(H.sum(axis=0)))==0
 for mode in ['primary','witness']:
  M=sparse.load_npz(old/mode/'matrix.npz');r=dict(np.load(old/mode/'arrays.npz'));actual=sparse.load_npz(src/mode/'matrix.npz');out=dict(np.load(src/mode/'arrays.npz'))
  V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr');assert mx(actual-M@V)==0
  assert np.array_equal(out['rhs'],r['rhs']-M[:,:-1]@a)
  for key in ['cost','bounds','equality_rhs']:assert np.array_equal(out[key],r[key])
  assert mx(sparse.load_npz(src/mode/'equality.npz')-sparse.load_npz(old/mode/'equality.npz'))==0
  checks[mode]=dict(error=0.,rhs_error=0.,shape=list(actual.shape),new_nnz=actual.nnz,cost_bounds_equalities_exact=True)
 prep=read(src/'preparation.json');out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,new_integer_incidence_inverse_error=0.,new_gap_feature_identity_error=0.,all206_drop_differences_and_endpoint_sum_exact=True,coordinate_transform_independent_column_rebuild_error=0.,unchanged_Q_and_shift=True,inherited_Q_nonsingularity_bound=proof['directed_inverse_norm_upper_max'],invertibility='Block triangular: NEW exact integer invertible205gap incidence, identity other joint directions, unchanged prior certifiedQ; rounded lower-left does not affect invertibility.',checks=checks,literal_closure_identity_defect_retained=prep['literal_closure_identity_defect'],all_original_rows_and_both_consistency_signs_retained=True,original_witness_and_exact_interval_checks_still_required=True,solver_speedup_demonstrated=False,actual_model_calls=0,local_inverse_calls=0,directed_inverse_proof_calls=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
