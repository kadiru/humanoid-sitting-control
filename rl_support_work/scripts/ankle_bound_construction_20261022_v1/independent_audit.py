"""Independent integer-dyadic row checker; imports no producer mathematics."""
from common import *
from fractions import Fraction

def dyad(x):
    n,d=float(x).as_integer_ratio();return n,d.bit_length()-1

def add(s,e,n,k):
    if k>e:return (s<<(k-e))+n,k
    return s+(n<<(e-k)),e

def exact_row(M,i,vector):
    s=e=0
    for j in range(int(M.indptr[i]),int(M.indptr[i+1])):
        n,k=dyad(M.data[j]);v,q=vector[int(M.indices[j])]
        s,e=add(s,e,n*v,k+q)
    assert abs(s).bit_length()<=4352 and e<=2148
    return s,e

def less_equal(s,e,t,k):return (s<<max(0,k-e))<=(t<<max(0,e-k))

def main():
    origin=admission('audit');spec=contract()
    for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():
        assert sha(path)==h,path
        check(origin)
    candidate=read(O/'candidate.json')
    if not candidate['diagnostic_passed']:
        write(O/'audit.json',dict(passed=True,diagnostic_passed=False,
            scope='Producer failure accounting only; no primal, improvement or optimality certificate',
            reason=candidate['reason'],physical_success=False));return
    import numpy as np
    from scipy import sparse
    M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr();C=sparse.load_npz(PRIOR/'C.npz').tocsr()
    E=sparse.load_npz(PRIOR/'full_equality.npz').tocsr();a=dict(np.load(PRIOR/'full_arrays.npz'))
    h=np.load(PRIOR/'state.npz')['h'];point=np.load(O/'constructed_point.npy');raw=np.load(O/'raw_direction.npy')
    assert M.shape==(652529,78791) and M.nnz==2614306 and M.has_canonical_format
    assert C.shape==(68250,78790) and C.nnz==749570 and C.has_canonical_format
    assert np.diff(M.indptr).max()<=256 and np.diff(C.indptr).max()<=256
    assert E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
    assert point.shape==(78791,) and raw.shape==(78790,)
    assert point.dtype==np.dtype('float64') and raw.dtype==np.dtype('float64')
    assert np.isfinite(point).all() and np.isfinite(raw).all()
    active=np.load(BASE/'reference.npz')['active'];layout=read(D/'gap_coordinate_preparation_v3_466/layout.json')['columns']
    pcols=[i for i,c in enumerate(layout) if c['kind']=='orientation']
    assert len(pcols)==1365 and {layout[i]['state'] for i in pcols}==set(map(int,active))
    B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsc()
    assert B.shape==(14100,10540)
    for j in pcols:
        lo,hi=map(int,B.indptr[j:j+2]);assert hi-lo==1 and B.indices[lo]==10*layout[j]['state']+1 and B.data[lo]==1.
    mask=np.ones(10540,dtype=bool);mask[pcols]=False
    assert np.all(raw[:10540][mask]==0) and np.all(raw[10540:].reshape(1365,50)[:,14]==0)
    assert raw[42104]==0 and point[42104]==0
    assert np.isneginf(a['bounds'][:-1,0]).all() and np.isposinf(a['bounds'][:-1,1]).all()
    assert np.array_equal(a['bounds'][-1],[0.,1.]) and 0<=point[-1]<=1
    mu=read(ACCEPTED/'decision.json')['summary']['phi']
    sm=read(ACCEPTED/'decision.json')['summary']
    assert mu==sm['consistency_scaled_max'] and sm['auxiliary_soft_merit']+2.**-20<mu
    assert sm['auxiliary_hard_pass'] and read(ACCEPTED/'audit.json')['passed']
    assert read(PRIOR/'input_proof.json')['new_state_full_input_proved']
    k=candidate['dyadic_exponent'];assert isinstance(k,int) and 1<=k<=1022
    pv=list(map(dyad,point));rv=list(map(dyad,raw))
    for i,(n,e) in enumerate(rv):
        pn,pe=pv[i]
        assert less_equal(pn,pe,n,e+k) and less_equal(n,e+k,pn,pe)
    # Verify t is on/above the prescribed decreasing epigraph line exactly.
    mn,me=dyad(mu);tn,te=pv[-1]
    lower,le=add(mn,me,-mn,me+k+1)
    assert less_equal(lower,le,tn,te)
    reduction,re=add(mn,me,-tn,te);tol,ee=dyad(1e-9)
    assert not less_equal(reduction,re,tol,ee)
    raw_rows=0;point_rows=0;products=0;theta_n=theta_e=0
    meta=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');kept=np.flatnonzero(meta['keep']);assert len(kept)==331495
    current_r=np.load(PRIOR/'state.npz')['r']
    for i in range(C.shape[0]):
        if i%512==0:check(origin)
        s,e=exact_row(C,i,rv);n,k0=dyad(h[i])
        rs,re0=add(s,e,n,k0)
        assert less_equal(abs(rs),re0,tol,ee),('raw consistency residual',i)
        predicted,pe=add(s,e+k,n,k0)
        if not less_equal(abs(predicted),pe,theta_n,theta_e):theta_n,theta_e=abs(predicted),pe
        raw_rows+=1;products+=int(C.indptr[i+1]-C.indptr[i])
    for i in range(M.shape[0]):
        if i%512==0:check(origin)
        s,e=exact_row(M,i,pv);n,k0=dyad(a['rhs'][i])
        assert less_equal(s,e,n,k0),('original exact primal row',i)
        if i<331495 and meta['soft'][kept[i]]:
            violation,ve=add(s,e,tn,te)
            rn,rk=dyad(current_r[kept[i]]);violation,ve=add(violation,ve,-rn,rk)
            if not less_equal(violation,ve,theta_n,theta_e):theta_n,theta_e=violation,ve
        point_rows+=1;products+=int(M.indptr[i+1]-M.indptr[i])
    exact_theta=Fraction(theta_n,1<<theta_e);exact_mu=Fraction.from_float(mu)
    assert exact_mu-exact_theta>=exact_mu/100 and exact_mu-exact_theta>Fraction.from_float(1e-9)
    assert abs(float(exact_theta)-candidate['modeled_theta'])<=1e-9
    check(origin)
    write(O/'audit.json',dict(passed=True,diagnostic_passed=True,
        exact_stored_binary64_primal_feasible=True,full_original_inequality_rows=point_rows,
        original_equality_rows=1366,original_bounds_checked=True,
        raw_consistency_rows=raw_rows,exact_products=products,
        exact_modeled_theta=[str(exact_theta.numerator),str(exact_theta.denominator)],
        exact_predicted_fractional_reduction=float((exact_mu-exact_theta)/exact_mu),minimum_one_percent_predicted_improvement=True,
        right_ankle_roll_held=True,only_existing_orientation_base_coordinates_freed=True,
        exact_epigraph_reduction=[str(reduction),re],
        raw_sha256=sha(O/'raw_direction.npy'),point_sha256=sha(O/'constructed_point.npy'),
        full_KKT_passed=False,stationarity_checked=False,dual_gap_checked=False,
        primary_optimum_claimed=False,interior_clearance_checked=False,finite_map=False,
        nonlinear_model_evaluation=False,physical_success=False,
        scope='Approved cycle stage B: exact original primal and useful modeled-descent certificate; conditional stage C still requires its own freeze. No physical admission.'))
    event('independent_audit_completed',origin)

if __name__=='__main__':main()
