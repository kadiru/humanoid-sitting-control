"""One new residual-coordinate saved-matrix preparation; no solver/model."""
import datetime
from low_seat_residual_coordinates_496 import *
O=D/'residual_coordinate_preparation_496'
def main():
 launch=read(D/'residual_preparation_launch_496.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False)
 old,r,C,L=load();S,a,Q,Qs,U=transform(C,L,r['h'])
 local_defect=L@Q-np.eye(BLOCK);norm=np.sum(abs(local_defect),axis=2).max(axis=1)
 np.savez_compressed(O/'local_blocks.npz',L=L,Q=Q,local_defect=local_defect,ordinary_residual_norm=norm)
 sparse.save_npz(O/'coordinate_transform.npz',S)
 np.savez_compressed(O/'shift.npz',a=a,h=r['h'])
 chains={}
 for name in ['feature_basis','all_rows_chain','consistency_chain','interval_matrix','normal_drop_chain']:
  M=sparse.load_npz(old/(name+'.npz')).tocsr();T=(M@S).tocsr();shift=M@a
  sparse.save_npz(O/(name+'.npz'),T);np.savez_compressed(O/(name+'_shift.npz'),shift=shift)
  chains[name]=dict(shape=list(T.shape),source_nnz=M.nnz,transformed_nnz=T.nnz)
 for mode in ['primary','witness']:
  out=O/mode;out.mkdir();A=sparse.load_npz(old/mode/'matrix.npz').tocsr();z=dict(np.load(old/mode/'arrays.npz'))
  T,b=substitute(A,z['rhs'],S,a);sparse.save_npz(out/'matrix.npz',T)
  E=sparse.load_npz(old/mode/'equality.npz');assert E.nnz==0
  sparse.save_npz(out/'equality.npz',E)
  np.savez_compressed(out/'arrays.npz',rhs=b,cost=z['cost'],bounds=z['bounds'],equality_rhs=z['equality_rhs'])
  chains[mode]=dict(shape=list(T.shape),source_nnz=A.nnz,transformed_nnz=T.nnz,rows_preserved=True)
 # Two NEW affine reconstruction comparisons on all78790 coordinates. These
 # are algebraic vectors, never finite maps, states, primal witnesses or seeds.
 vectors=np.stack([np.sin(np.arange(S.shape[1])*.017)*1e-7,np.cos(np.arange(S.shape[1])*.023)*1e-7])
 oldvectors=np.asarray([S@v+a for v in vectors]);outputs=[]
 for i,v in enumerate(vectors):
  oldc=r['h']+C@oldvectors[i];newc=r['h']+C@a+(C@S)@v
  outputs.append(float(np.max(abs(oldc-newc))))
 np.savez_compressed(O/'affine_vectors.npz',new=vectors,old=oldvectors)
 target=sparse.hstack([sparse.csr_matrix((C.shape[0],N0)),sparse.eye(C.shape[0],format='csr')],format='csr')
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),specification=SPEC,passed=True,coordinates=S.shape[0],local_blocks=len(Q),new_local_residual_right_hand_sides=int(len(Q)*BLOCK),ordinary_inverse_residual_norm_max=float(norm.max()),literal_closure_identity_defect=maximum(C@S-target),literal_closure_shift_defect=float(np.max(abs(r['h']+C@a))),algebra_vector_errors=outputs,chains=chains,actual_X_Y_J_changed=False,reference_auxiliary_changed=False,new_solver_exists=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,new_complete_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2))
 (O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2))
 assert norm.max()<1e-8
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

