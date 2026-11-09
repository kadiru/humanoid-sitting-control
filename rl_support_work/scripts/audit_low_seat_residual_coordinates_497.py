"""Independent saved-array substitution and directed-rounding inverse audit.
No second local solve, mechanics, LP, finite decoder or old fixture replay.
"""
import datetime
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');N0=10540
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def maxsp(x):return float(abs(x.data).max()) if x.nnz else 0.
def main():
 assert read(D/'residual_preparation_exit_496.json')['actual_exit']==0
 src=D/'residual_coordinate_preparation_496';old=D/'lifted_feasibility_preparation_490';O=D/'residual_coordinate_audit_497';O.mkdir(exist_ok=False)
 for p,h in read(src/'SHA256.json').items():assert sha(src/p)==h,p
 r=dict(np.load(old/'reference.npz'));C=sparse.load_npz(old/'consistency_chain.npz').tocsr();blocks=np.load(src/'local_blocks.npz');L=blocks['L'];Q=blocks['Q'];m=len(Q)
 orig=C[:,N0:].tocoo();rebuilt=np.zeros_like(L);assert np.all(orig.row//50==orig.col//50)
 rebuilt[orig.row//50,orig.row%50,orig.col%50]=orig.data;assert np.array_equal(L,rebuilt)
 # IEEE754 outward intervals for EVERY exact binary-rational L*Q entry.
 # Each product and each partial sum is expanded one representable number
 # each way. Finite operands/no overflow are explicit requirements.
 lo=np.zeros_like(Q);hi=np.zeros_like(Q)
 for k in range(50):
  p=L[:,:,k,None]*Q[:,None,k,:];assert np.isfinite(p).all()
  lo=np.nextafter(lo+np.nextafter(p,-np.inf),-np.inf)
  hi=np.nextafter(hi+np.nextafter(p,np.inf),np.inf)
 eye=np.eye(50);lo=np.nextafter(lo-eye,-np.inf);hi=np.nextafter(hi-eye,np.inf)
 absbound=np.maximum(abs(lo),abs(hi));rowbound=np.zeros((m,50))
 for k in range(50):rowbound=np.nextafter(rowbound+absbound[:,:,k],np.inf)
 normbound=rowbound.max(axis=1);assert np.all(normbound<1)
 np.savez_compressed(O/'directed_inverse_certificate.npz',lower=lo,upper=hi,row_norm_upper=rowbound,block_norm_upper=normbound)
 # Rebuild the affine coordinate change independently from its definition.
 Qs=sparse.block_diag(list(Q),format='csr');C0=C[:,:N0]
 U=-Qs@C0;S=sparse.bmat([[sparse.eye(N0,format='csr'),None],[U,Qs]],format='csr');a=np.r_[np.zeros(N0),-Qs@r['h']]
 savedS=sparse.load_npz(src/'coordinate_transform.npz');saveda=np.load(src/'shift.npz')['a'];assert maxsp(S-savedS)==0 and np.array_equal(a,saveda)
 matrices={};different_association={}
 for name in ['feature_basis','all_rows_chain','consistency_chain','interval_matrix','normal_drop_chain']:
  A=sparse.load_npz(old/(name+'.npz')).tocsr();actual=sparse.load_npz(src/(name+'.npz'));expected=A@S;shift=A@a
  assert maxsp(actual-expected)==0 and np.array_equal(shift,np.load(src/(name+'_shift.npz'))['shift'])
  matrices[name]=dict(error=0,rows=A.shape[0],no_thresholding=True)
 for mode in ['primary','witness']:
  A=sparse.load_npz(old/mode/'matrix.npz').tocsr();arr=dict(np.load(old/mode/'arrays.npz'));T=sparse.load_npz(src/mode/'matrix.npz');out=dict(np.load(src/mode/'arrays.npz'))
  # Whole-row rebuild preserves original coefficient association exactly.
  V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr');assert maxsp(T-A@V)==0
  assert np.array_equal(out['rhs'],arr['rhs']-A[:,:-1]@a)
  for k in ['cost','bounds','equality_rhs']:assert np.array_equal(out[k],arr[k])
  assert maxsp(sparse.load_npz(src/mode/'equality.npz')-sparse.load_npz(old/mode/'equality.npz'))==0
  # Independent block-elimination association. Differences are reported, not
  # used as a physical/linear feasibility tolerance or a row replacement.
  az=A[:,N0:-1]@Qs;alt=sparse.hstack([A[:,:N0]-az@C0,az,A[:,-1]],format='csr')
  delta=T-alt;scale=np.asarray(abs(T).max(axis=1).toarray()).ravel()+np.asarray(abs(alt).max(axis=1).toarray()).ravel()+1.
  rel=delta.multiply((1/scale)[:,None]);different_association[mode]=dict(absolute=maxsp(delta),row_scaled=maxsp(rel))
  assert maxsp(rel)<2e-10 # separately scoped algebra comparison only
  matrices[mode]=dict(error=0,shape=list(T.shape),rhs_error=0,cost_bounds_equalities_exact=True)
 v=np.load(src/'affine_vectors.npz');vector_errors=[]
 for new,oldv in zip(v['new'],v['old']):
  # Independent state-local reconstruction, not production decoder.
  local=np.einsum('nij,nj->ni',Q,(new[N0:]-r['h']-C0@new[:N0]).reshape(m,50))
  direct=np.r_[new[:N0],local.ravel()];err=float(np.max(abs(direct-oldv)));vector_errors.append(err);assert err<2e-10
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,local_blocks=m,directed_inverse_norm_upper_max=float(normbound.max()),bijective_exact_real_affine_map=True,proof='Every exact binary-rational L*Q has ||I-LQ||infinity<1 by outward-rounded operations, so each Q is nonsingular. S is block triangular with identity and Q blocks.',matrix_checks=matrices,independent_association=different_association,vector_errors=vector_errors,all_original_rows_and_both_consistency_signs_retained=True,closure_defects_retained=True,original_witness_and_exact_interval_checks_still_required=True,actual_model_calls=0,local_solve_calls=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
