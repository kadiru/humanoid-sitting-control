"""Prepare ONE new full-feature collocation transcription from sealed saved arrays."""
import datetime,json
import numpy as np
from scipy import sparse
from low_seat_feature_collocation_488 import *
O=D/'feature_collocation_preparation_488'

def main():
    launch=read(D/'feature_preparation_launch_488.json');assert launch['compile_actual_exit']==0
    for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
    O.mkdir(exist_ok=False);d=load_saved();s=d['seed'];F=np.c_[s['X'],s['Y']];K=len(F)
    P,active,extra_rows=lift_basis(d);assert P.shape==(105750,99265) and len(active)==1365
    assert P[np.repeat(d['mesh']['fixed'],75)].nnz==0
    inactive_rows=(np.arange(K)[:,None]*75+np.arange(16,24))[d['mesh']['inactive']]
    assert P[inactive_rows].nnz==0
    R=(d['M']@P).multiply((1/d['rows']['scale'])[:,None]).tocsr()
    E=(d['E']@P).tocsr();C=(d['C']@P).tocsr()
    r=(d['M']@F.ravel()+d['rows']['constant'])/d['rows']['scale'];e=d['E']@F.ravel()+d['eq']['constant']
    assert np.array_equal(r,s['rows']) and np.array_equal(e,s['equalities'])
    L=closure_blocks(s['J'][active]);local=sparse.block_diag(list(L),format='csr')
    state_rows=(active[:,None]*75+np.arange(75)).ravel();Q=(local@P[state_rows]).tocsr()
    closure=np.c_[np.zeros((len(active),51)),s['Y'][active,:14]].ravel()
    assert Q.shape==(88725,99265)
    T,G,A,rhs,raw=reduced_null(s['J']);GB=(sparse.block_diag(list(G),format='csr')@d['B']).tocsr()
    N=sparse.vstack([sparse.eye(d['B'].shape[1],format='csr'),GB[extra_rows]],format='csr')
    reconstruct=P@N-GB;reconstruct.eliminate_zeros()
    qnull=Q@N;qnull.eliminate_zeros();enull=E@N;enull.eliminate_zeros()
    oldA=sparse.load_npz(D/'free_gap_path_472/linear_00/A.npz');oldC=sparse.load_npz(D/'free_gap_path_472/linear_00/C.npz')
    rowdiff=R@N-oldA;rowdiff.eliminate_zeros();boxdiff=C@N-oldC;boxdiff.eliminate_zeros()
    normal=np.array([i for i,c in enumerate(d['gap_layout']['columns']) if c['kind']=='normal_drop'])
    rr=[];cc=[];vv=[]
    for j in normal:
        c=d['gap_layout']['columns'][j];rr.extend([c['edge'],c['pivot']]);cc.extend([int(j),int(j)]);vv.extend([1.,-1.])
    drop=sparse.csr_matrix((vv,(rr,cc)),shape=(206,P.shape[1]))
    cm=d['cm'];free=cm['free'];ref=cm['reference'].ravel()[free]
    lower=np.maximum(cm['lower'].ravel()[free],ref-.02);upper=np.minimum(cm['upper'].ravel()[free],ref+.02)
    checks=dict(reference_rows_error=0.,reference_equalities_error=0.,basis_null_reconstruction=max_sparse(reconstruct),closure_null_error=max_sparse(qnull),original_equalities_null_error=max_sparse(enull),old331888_row_chain_error=max_sparse(rowdiff),old11906_box_chain_error=max_sparse(boxdiff),implicit_schur_error=float(abs(np.einsum('nij,njk->nik',A,T[:,DEP])-rhs).max()),schur_condition_max=float(np.linalg.cond(A).max()),source_consistency_error=0.,source_anchor_error=float(abs(closure).max()),original_equality_lifted_nonzeros=int(E.nnz),original_equality_reduced_nonzeros=int(enull.nnz))
    # Save detailed arrays before aggregate assertions; no model work is hidden.
    for name,m in [('feature_basis',P),('all_rows_lifted_chain',R),('original_equalities_lifted_chain',E),('original_control_box_chain',C),('normal_drop_chain',drop),('closure_jacobian',Q),('null_reduction',N)]:sparse.save_npz(O/(name+'.npz'),m)
    np.savez_compressed(O/'reference.npz',X=s['X'],Y=s['Y'],J=s['J'],features=F,target=s['target'],rows=r,equalities=e,closure=closure,active=active,extra_rows=extra_rows,old_control_lower=lower,old_control_upper=upper,label_to_state=d['mesh']['label_to_state'],fixed=d['mesh']['fixed'],inactive=d['mesh']['inactive'])
    np.savez_compressed(O/'elimination.npz',T=T,G=G,schur=A,rhs=rhs,raw_output_chain=raw)
    layout=dict(specification=SPEC,local_controls=LOCAL_CONTROLS.tolist(),extra_features=EXTRA.tolist(),active_states=active.tolist(),old_reduced_coordinates=int(d['B'].shape[1]),lifted_coordinates=P.shape[1],closure_order='For each active state:51(Y-Mechanics(X)),then14auxiliaryfeet/moments.',normal_drop_columns=normal.tolist(),all1411_original_equalities_retained=True,normal_equalities_not_zero_in_lifted_space=True,exact_drop_state_source=str(D/'free_gap_path_472/linear_00/metadata.json'),no_finite_map_or_optimizer=True)
    (O/'layout.json').write_text(json.dumps(layout,indent=2));(O/'checks.json').write_text(json.dumps(checks,indent=2))
    for key in ['basis_null_reconstruction','closure_null_error','original_equalities_null_error','old331888_row_chain_error','old11906_box_chain_error','implicit_schur_error']:assert checks[key]<=ALGEBRA_BOUND,(key,checks[key])
    assert checks['original_equality_lifted_nonzeros']>0 and checks['original_equality_reduced_nonzeros']==0
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,specification=SPEC,checks=checks,states=K,labels=1467,exact_times=1460,fixed_states=45,active_states=1365,lifted_coordinates=P.shape[1],nonlinear_consistency_rows=1365*51,affine_foot_moment_rows=1365*14,original_final_inequalities=331888,original_final_equalities=1411,model_calls=0,root_calls=0,ad_calls=0,finite_work_columns=0,lp_calls=0,new_complete_path=False,physical_admission=False)
    (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
