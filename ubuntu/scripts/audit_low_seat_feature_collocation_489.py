"""Independent saved-array audit of lifted features488; no mechanics/root/LP calls."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'feature_collocation_preparation_488';O=D/'feature_collocation_audit_489'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def error(a,b):
    x=(a-b).tocsr();x.eliminate_zeros();return float(abs(x.data).max()) if x.nnz else 0.
def maximum(a):return float(abs(a.data).max()) if a.nnz else 0.
def main():
    assert read(D/'feature_preparation_exit_488.json')['actual_exit']==0
    for n,h in read(P/'SHA256.json').items():assert sha(P/n)==h,n
    for n,h in read(D/'feature_preparation_launch_488.json')['source_input_sha256'].items():assert sha(n)==h,n
    O.mkdir(exist_ok=False);s=dict(np.load(D/'free_gap_path_472/initial.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));a=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));ea=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz'));cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'))
    ref=dict(np.load(P/'reference.npz'));el=dict(np.load(P/'elimination.npz'));layout=read(P/'layout.json');B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsr()
    active=np.flatnonzero(~mesh['fixed']);extra=[0,1,2]+list(range(4,16))+[24+i for i in range(51) if i!=47];control=[71,3]+list(range(16,24));n0=B.shape[1];n=n0+65*len(active);K=len(s['X'])
    rr=[];cc=[];vv=[]
    # Independent row-wise incidence construction, not production lift_basis.
    for i in range(B.shape[0]):
        for p in range(B.indptr[i],B.indptr[i+1]):rr.append((i//10)*75+control[i%10]);cc.append(int(B.indices[p]));vv.append(float(B.data[p]))
    for k,b in enumerate(active):
        for j,f in enumerate(extra):rr.append(int(b)*75+f);cc.append(n0+65*k+j);vv.append(1.)
    S=sparse.csr_matrix((vv,(rr,cc)),shape=(K*75,n));assert error(S,sparse.load_npz(P/'feature_basis.npz'))==0
    assert layout['extra_features']==extra and layout['local_controls']==control and layout['active_states']==active.tolist()
    assert S[np.repeat(mesh['fixed'],75)].nnz==0
    ids=(np.arange(K)[:,None]*75+np.arange(16,24))[mesh['inactive']];assert S[ids].nnz==0
    for key in ['X','Y','J','target']:assert np.array_equal(ref[key],s[key]),key
    assert np.array_equal(ref['label_to_state'],mesh['label_to_state']) and np.array_equal(ref['fixed'],mesh['fixed']) and np.array_equal(ref['inactive'],mesh['inactive'])
    M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz');E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz');C0=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/original_control_box_matrix.npz')
    R=(M@S).multiply((1/a['scale'])[:,None]).tocsr();EC=E@S;BC=C0@S
    matrix_errors={name:error(m,sparse.load_npz(P/(name+'.npz'))) for name,m in [('all_rows_lifted_chain',R),('original_equalities_lifted_chain',EC),('original_control_box_chain',BC)]}
    assert all(v==0 for v in matrix_errors.values())
    F=np.c_[s['X'],s['Y']];assert np.array_equal((M@F.ravel()+a['constant'])/a['scale'],ref['rows']);assert np.array_equal(E@F.ravel()+ea['constant'],ref['equalities'])
    assert np.array_equal(ref['rows'],s['rows']) and np.array_equal(ref['equalities'],s['equalities'])
    free=cm['free'];base=cm['reference'].ravel()[free]
    assert np.array_equal(ref['old_control_lower'],np.maximum(cm['lower'].ravel()[free],base-.02));assert np.array_equal(ref['old_control_upper'],np.minimum(cm['upper'].ravel()[free],base+.02))
    # Independently assemble51 consistency rows and14 actual anchor equations.
    lr=[];lc=[];lv=[]
    for k,b in enumerate(active):
        J=s['J'][b]
        for y in range(51):
            for x in np.flatnonzero(J[y]):lr.append(k*65+y);lc.append(int(b)*75+int(x));lv.append(-float(J[y,x]))
            lr.append(k*65+y);lc.append(int(b)*75+24+y);lv.append(1.)
        for y in range(14):lr.append(k*65+51+y);lc.append(int(b)*75+24+y);lv.append(1.)
    L=sparse.csr_matrix((lv,(lr,lc)),shape=(65*len(active),K*75));Q=L@S;assert error(Q,sparse.load_npz(P/'closure_jacobian.npz'))==0
    assert np.array_equal(ref['closure'],np.c_[np.zeros((len(active),51)),s['Y'][active,:14]].ravel())
    # Verify saved elimination by its defining equations; no second solve/fixture.
    T=el['T'];dep=np.array([0,1,2]+list(range(4,16)));ind=np.array([3]+list(range(16,24)));eq=np.r_[np.arange(14),47]
    JA=s['J'][:,eq][:,:,dep];rhs=np.concatenate([np.tile(np.eye(15)[-1],(K,1))[:,:,None],-s['J'][:,eq][:,:,ind]],axis=2)
    assert np.array_equal(JA,el['schur']) and np.array_equal(rhs,el['rhs']);assert np.array_equal(T[:,ind,1:],np.tile(np.eye(9),(K,1,1))) and np.all(T[:,ind,0]==0)
    schur_error=float(abs(np.einsum('nij,njk->nik',JA,T[:,dep])-rhs).max());raw=np.einsum('nij,njk->nik',s['J'],T);G=np.concatenate([T,raw],axis=1);G[:,24:38,:]=0;G[:,71,:]=0;G[:,71,0]=1
    assert np.array_equal(raw,el['raw_output_chain']) and np.array_equal(G,el['G'])
    GB=sparse.block_diag(list(G),format='csr')@B;extra_rows=(active[:,None]*75+np.array(extra)).ravel();N=sparse.vstack([sparse.eye(n0,format='csr'),GB[extra_rows]],format='csr')
    assert error(N,sparse.load_npz(P/'null_reduction.npz'))==0
    closure_null=maximum(Q@N);eq_null=EC@N;eq_null.eliminate_zeros();assert eq_null.nnz==0
    oldA=sparse.load_npz(D/'free_gap_path_472/linear_00/A.npz');oldC=sparse.load_npz(D/'free_gap_path_472/linear_00/C.npz')
    equivalence=dict(basis=error(S@N,GB),all331888rows=error(R@N,oldA),all11906boxes=error(BC@N,oldC),closure=closure_null,schur=schur_error)
    assert all(v<=2e-10 for v in equivalence.values())
    gl=read(D/'gap_coordinate_preparation_v3_466/layout.json');dr=[];dc=[];dv=[]
    for j,c in enumerate(gl['columns']):
        if c['kind']=='normal_drop':dr.extend([c['edge'],c['pivot']]);dc.extend([j,j]);dv.extend([1.,-1.])
    DS=sparse.csr_matrix((dv,(dr,dc)),shape=(206,n));assert error(DS,sparse.load_npz(P/'normal_drop_chain.npz'))==0
    # An auxiliary command change with X held fixed must remain a nonzero
    # consistency defect, even though the command rows themselves are affine.
    state_index=0;ycommand=15;j=n0+65*state_index+extra.index(24+ycommand);delta=2.**-20
    defect=Q[:,j].toarray().ravel()*delta;assert defect[state_index*65+ycommand]==delta and np.count_nonzero(defect)==1
    changed_rows=R[:,j].nnz;assert changed_rows>0
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,lifted_coordinates=n,closure_equations=Q.shape[0],nonlinear_consistency_rows=51*len(active),affine_anchor_rows=14*len(active),all_row_matrix_errors=matrix_errors,closure_matrix_error=0.,null_equivalence_errors=equivalence,original_lifted_equality_nonzeros=int(EC.nnz),original_reduced_equality_nonzeros=0,auxiliary_only_command_counterexample=dict(command_change_rad=delta,consistency_defect_rad=delta,affine_rows_affected=changed_rows,rejected_as_actual_path=True),all_original_buffers_and_boxes_unchanged=True,all_labels_fixed_aliases_inactive_preserved=True,reference_is_exact472_initial=True,no_new_candidate=True,model_calls=0,root_calls=0,ad_calls=0,finite_work_columns=0,lp_calls=0,physical_admission=False)
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
