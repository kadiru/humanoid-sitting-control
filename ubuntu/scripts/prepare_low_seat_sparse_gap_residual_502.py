"""One new whole-row normal-gap-incidence preparation; no optimizer/model."""
import datetime
from low_seat_sparse_gap_residual_502 import *
O=D/'sparse_gap_residual_preparation_502'
def main():
 launch=read(D/'sparse_gap_preparation_launch_502.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);old=D/'lifted_feasibility_preparation_490';prior=D/'residual_coordinate_preparation_496'
 layout=read(D/'gap_coordinate_preparation_v3_466/layout.json');T,Ti,cols,edges=incidence(layout)
 E=sparse.eye(N,format='csr');assert maximum(T@Ti-E)==maximum(Ti@T-E)==0
 Sold=sparse.load_npz(prior/'coordinate_transform.npz').tocsr();S=(Sold@T).tocsr();a=np.load(prior/'shift.npz')['a'];r=dict(np.load(old/'reference.npz'))
 assert maximum(S[N0:,N0:]-Sold[N0:,N0:])==0 and S[:N0,N0:].nnz==0
 for name,M in [('gap_incidence',T),('gap_prefix_inverse',Ti),('coordinate_transform',S)]:sparse.save_npz(O/(name+'.npz'),M)
 np.savez_compressed(O/'coordinates.npz',slots=cols,old_edges=edges,new_gap_indices=np.arange(1,206),shift=a)
 chains={}
 for name in ['feature_basis','all_rows_chain','consistency_chain','interval_matrix','normal_drop_chain']:
  M=sparse.load_npz(old/(name+'.npz')).tocsr();A=(M@S).tocsr();shift=M@a
  sparse.save_npz(O/(name+'.npz'),A);np.savez_compressed(O/(name+'_shift.npz'),shift=shift)
  priorM=sparse.load_npz(prior/(name+'.npz'))
  chains[name]=dict(shape=list(A.shape),original_nnz=M.nnz,prior_residual_nnz=priorM.nnz,new_nnz=A.nnz)
 P=sparse.load_npz(O/'feature_basis.npz');normalrows=np.array(layout['normal_states'])*75+71
 G=P[normalrows][:,cols];wanted=sparse.vstack([sparse.csr_matrix((1,205)),sparse.eye(205),sparse.csr_matrix((1,205))],format='csr')
 assert maximum(G-wanted)==0
 drop=sparse.load_npz(O/'normal_drop_chain.npz')[:,cols];assert maximum(drop-(wanted[:-1]-wanted[1:]))==0 and np.array_equal(np.asarray(drop.sum(axis=0)),np.zeros((1,205)))
 for mode in ['primary','witness']:
  out=O/mode;out.mkdir();A=sparse.load_npz(old/mode/'matrix.npz').tocsr();z=dict(np.load(old/mode/'arrays.npz'))
  V,b=substitute(A,z['rhs'],S,a);sparse.save_npz(out/'matrix.npz',V)
  eq=sparse.load_npz(old/mode/'equality.npz');assert eq.nnz==0;sparse.save_npz(out/'equality.npz',eq)
  np.savez_compressed(out/'arrays.npz',rhs=b,cost=z['cost'],bounds=z['bounds'],equality_rhs=z['equality_rhs'])
  priorA=sparse.load_npz(prior/mode/'matrix.npz');chains[mode]=dict(shape=list(V.shape),original_nnz=A.nnz,prior_residual_nnz=priorA.nnz,new_nnz=V.nnz,rows_preserved=True)
 C=sparse.load_npz(O/'consistency_chain.npz');ideal=sparse.hstack([sparse.csr_matrix((C.shape[0],N0)),sparse.eye(C.shape[0])],format='csr')
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,coordinates=N,new_normal_gap_coordinates=205,normal_anchor_increments_exact_zero=True,normal_drop_incidence_exact=True,integer_inverse_error=0.,stored_Q_unchanged=True,prior_nonsingular_Q_bound=read(D/'residual_coordinate_audit_497/independent_audit.json')['directed_inverse_norm_upper_max'],source_transform_nnz=Sold.nnz,new_transform_nnz=S.nnz,literal_closure_identity_defect=maximum(C-ideal),literal_closure_shift_defect=float(np.max(abs(r['h']+sparse.load_npz(old/'consistency_chain.npz')@a))),chains=chains,original_reconstructed_witness_still_required=True,actual_X_Y_J_changed=False,reference_auxiliary_changed=False,new_solver_exists=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
