"""One saved-matrix partial-residual/native-auxiliary preparation."""
import datetime
from low_seat_mixed_coordinates_504 import *
O=D/'mixed_coordinate_preparation_504'
def main():
 launch=read(D/'mixed_preparation_launch_504.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);old=D/'lifted_feasibility_preparation_490';prior=D/'residual_coordinate_preparation_496';gap=D/'sparse_gap_residual_preparation_502'
 r=dict(np.load(old/'reference.npz'));C=sparse.load_npz(old/'consistency_chain.npz').tocsr();T=sparse.load_npz(gap/'gap_incidence.npz').tocsr()
 q=dict(np.load(prior/'local_blocks.npz'));L,Q=q['L'],q['Q'];assert np.all(L[:,KIN,15:]==0)
 cert=dict(np.load(D/'residual_coordinate_audit_497/directed_inverse_certificate.npz'))
 inherited_rows=cert['row_norm_upper'][:,KIN];assert np.all(inherited_rows<1)
 S,a,R,kinrows,roots,aux=construct(C,r['h'],Q,T)
 sparse.save_npz(O/'coordinate_transform.npz',S)
 np.savez_compressed(O/'coordinates.npz',shift=a,R=R,kin=KIN,aux_outputs=AUX,kinrows=kinrows,root_slots=roots,auxiliary_slots=aux)
 # Principal intervals inherit an existing rigorous calculation. No LQ or
 # directed products are recomputed; full selected-row bounds are conservative.
 np.savez_compressed(O/'inherited_subblock_certificate.npz',lower=cert['lower'][:,KIN,:][:,:,KIN],upper=cert['upper'][:,KIN,:][:,:,KIN],row_norm_upper=inherited_rows)
 native=N0+aux;identity=sparse.eye(N,format='csr')[native]
 assert maximum(S[native]-identity)==0 and np.all(a[native]==0) and S[:N0,N0:].nnz==0
 chains={}
 for name in ['feature_basis','all_rows_chain','consistency_chain','interval_matrix','normal_drop_chain']:
  M=sparse.load_npz(old/(name+'.npz')).tocsr();A=(M@S).tocsr();shift=M@a
  sparse.save_npz(O/(name+'.npz'),A);np.savez_compressed(O/(name+'_shift.npz'),shift=shift)
  previous=read(gap/'preparation.json')['chains'][name]
  chains[name]=dict(shape=list(A.shape),original_nnz=M.nnz,prior_502_nnz=previous['new_nnz'],new_nnz=A.nnz)
 for mode in ['primary','witness']:
  out=O/mode;out.mkdir();A=sparse.load_npz(old/mode/'matrix.npz').tocsr();z=dict(np.load(old/mode/'arrays.npz'))
  V,b=substitute(A,z['rhs'],S,a);sparse.save_npz(out/'matrix.npz',V)
  eq=sparse.load_npz(old/mode/'equality.npz');assert eq.nnz==0;sparse.save_npz(out/'equality.npz',eq)
  np.savez_compressed(out/'arrays.npz',rhs=b,cost=z['cost'],bounds=z['bounds'],equality_rhs=z['equality_rhs'])
  chains[mode]=dict(shape=list(V.shape),original_nnz=A.nnz,prior_502_nnz=read(gap/'preparation.json')['chains'][mode]['new_nnz'],new_nnz=V.nnz,rows_preserved=True)
 closure=sparse.load_npz(O/'consistency_chain.npz');closure_shift=r['h']+sparse.load_npz(old/'consistency_chain.npz')@a
 kin_ideal=sparse.csr_matrix((np.ones(len(roots)),(np.arange(len(roots)),N0+roots)),shape=(len(roots),N))
 auxrows=(np.arange(STATES)[:,None]*50+AUX).ravel()
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,coordinates=N,kinematic_residual_coordinates=len(roots),native_auxiliary_coordinates=len(aux),local_zero_kinematic_auxiliary_block_exact=True,inherited_R_nonsingularity_bound=float(inherited_rows.max()),prior_directed_proof_inherited_not_repeated=True,prior_integer_incidence_inherited_not_repeated=True,new_transform_nnz=S.nnz,native_auxiliary_identity_exact=True,new_shift_differs_from_496=not np.array_equal(a,np.load(prior/'shift.npz')['a']),literal_kinematic_identity_defect=maximum(closure[kinrows]-kin_ideal),literal_kinematic_shift_defect=float(abs(closure_shift[kinrows]).max()),literal_auxiliary_shift_max=float(abs(closure_shift[auxrows]).max()),all_consistency_rows_retained=closure.shape[0],chains=chains,original_reconstructed_witness_still_required=True,actual_X_Y_J_changed=False,reference_auxiliary_changed=False,new_solver_exists=False,model_calls=0,root_calls=0,local_inverse_solves=0,directed_product_proofs=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
