"""One proposed constructive primal diagnostic. Never a model/LP/IK execution."""
from common import *

def enclosure(M, x, origin):
    """Outward binary64 product/sum intervals, grouped by row length."""
    import numpy as np
    sizes=np.diff(M.indptr)
    assert sizes.max(initial=0)<=256
    lo=np.zeros(M.shape[0]);hi=lo.copy()
    for width in np.unique(sizes):
        check(origin)
        if width==0:continue
        rows=np.flatnonzero(sizes==width);starts=M.indptr[rows]
        a=np.zeros(len(rows));b=a.copy()
        for j in range(int(width)):
            ids=starts+j;v=M.data[ids];xx=x[M.indices[ids]]
            product=v*xx
            assert np.isfinite(product).all()
            zero=(v==0)|(xx==0)
            pl=np.where(zero,0.,np.nextafter(product,-np.inf))
            ph=np.where(zero,0.,np.nextafter(product,np.inf))
            # Sum with an exact zero needs no rounding reserve.
            al=a+pl;bh=b+ph
            a=np.where(pl==0,a,np.where(a==0,pl,np.nextafter(al,-np.inf)))
            b=np.where(ph==0,b,np.where(b==0,ph,np.nextafter(bh,np.inf)))
        assert np.isfinite(a).all() and np.isfinite(b).all()
        lo[rows]=a;hi[rows]=b
    return lo,hi

def fail(reason, **kw):
    write(O/'candidate.json',dict(diagnostic_passed=False,reason=reason,no_retry=True,**kw))

def main():
    origin=admission('producer');spec=contract()
    for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():
        assert sha(path)==h,path
        check(origin)
    import numpy as np
    from scipy import sparse
    proof=read(PRIOR/'input_proof.json')
    assert proof['passed'] and proof['new_state_full_input_proved']
    assert read(ACCEPTED/'audit.json')['passed']
    reference=read(ACCEPTED/'decision.json')['summary']
    mu=float(reference['phi']);soft=float(reference['auxiliary_soft_merit'])
    assert mu==reference['consistency_scaled_max'] and soft+2.**-20<mu
    assert reference['auxiliary_hard_pass']
    assert read(ACCEPTED/'decision.json')['trial_merit_accepted']
    C=sparse.load_npz(PRIOR/'C.npz').tocsr();M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr()
    E=sparse.load_npz(PRIOR/'full_equality.npz').tocsr();a=dict(np.load(PRIOR/'full_arrays.npz'))
    state=dict(np.load(PRIOR/'state.npz'));h=state['h']
    assert C.shape==(68250,78790) and C.nnz==749570
    assert M.shape==(652529,78791) and M.nnz==2614306
    assert E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
    assert np.all(a['cost'][:-1]==0) and a['cost'][-1]==1
    assert np.array_equal(a['bounds'][-1],[0.,1.])
    assert np.isneginf(a['bounds'][:-1,0]).all() and np.isposinf(a['bounds'][:-1,1]).all()
    assert C.has_canonical_format and M.has_canonical_format
    assert all(np.isfinite(v).all() for v in [M.data,C.data,a['rhs'],h])
    # Direct association with the fixed 50-row/50-extra-coordinate local layout.
    cr=np.repeat(np.arange(68250),np.diff(C.indptr));ext=C.indices>=10540
    col=C.indices[ext]-10540;row=cr[ext]
    assert np.array_equal(col//50,row//50)
    local=np.zeros((1365,50,50));local[row//50,row%50,col%50]=C.data[ext]
    physical_rows=np.r_[np.arange(14),46]
    auxiliary_rows=np.r_[np.arange(14,46),np.arange(47,50)]
    auxiliary_columns=np.arange(15,50)
    expected=np.zeros((50,35),dtype=bool);expected[auxiliary_rows,np.arange(35)]=True
    assert np.all(local[:,:,15:][:,~expected]==0)
    diagonal=local[:,auxiliary_rows,auxiliary_columns]
    assert np.isfinite(diagonal).all() and np.all(diagonal>0)
    check(origin)
    write(O/'linear_blocks_started.json',dict(blocks=1365,shape=[15,15],right_hand_sides=1,
        source='current sealed C and h',nonlinear_root_calls=0,regularization=False,fallback=False))
    # No condition-based selection, second solve, alternative block or regularization.
    physical=np.linalg.solve(local[:,physical_rows,:15],-h.reshape(1365,50)[:,physical_rows,None]).squeeze(2)
    raw=np.zeros(78790);extra=raw[10540:].reshape(1365,50);extra[:,:15]=physical
    residual=(C@raw+h).reshape(1365,50)
    extra[:,15:]=-residual[:,auxiliary_rows]/diagonal
    assert np.isfinite(raw).all() and np.all(raw[:10540]==0)
    rl,rh=enclosure(C,raw,origin)
    residual_lo=np.nextafter(rl+h,-np.inf);residual_hi=np.nextafter(rh+h,np.inf)
    correction_error=float(np.maximum(abs(residual_lo),abs(residual_hi)).max())
    np.save(O/'raw_direction.npy',raw)
    if correction_error>1e-9:
        fail('Linear consistency correction did not meet the fixed residual test',correction_error=correction_error);return
    # Epigraph path t(alpha)=mu-alpha*mu/2. Base origin is [0,mu].
    tcol=M.getcol(78790).tocoo();assert np.all(tcol.data==-1.)
    shift=np.zeros(M.shape[0]);shift[tcol.row]=mu
    base=a['rhs']+shift
    z=base-a['rhs'];error=(a['rhs']-(base-z))+(shift-z)
    slack_lower=np.where(error<0,np.nextafter(base,-np.inf),base)
    assert np.isfinite(base).all() and np.isfinite(error).all()
    if np.any(slack_lower<0):fail('The epigraph origin lacks certified affine feasibility');return
    path_direction=np.r_[raw,-mu/2]
    _,slope_upper=enclosure(M,path_direction,origin)
    outward=slope_upper>0
    if np.any(slack_lower[outward]<=0):fail('An active original row blocks this fixed correction');return
    limits=np.nextafter(slack_lower[outward]/slope_upper[outward],-np.inf)
    limit=min(1.,float(limits.min(initial=1.)))
    limit=float(np.nextafter(limit/2,0.))
    if not np.isfinite(limit) or limit<=0:fail('No positive finite step bound');return
    import math
    _,exponent=math.frexp(limit);k=max(1,1-exponent)
    if k>1022:fail('No normal binary64 dyadic step');return
    alpha=math.ldexp(1.,-k);direction=np.ldexp(raw,-k)
    assert np.array_equal(np.ldexp(direction,k),raw),'Underflow changes the prescribed ray'
    epigraph=float(np.nextafter(mu-alpha*(mu/2),np.inf))
    if not mu-epigraph>1e-9:
        fail('The certified step gives no improvement larger than the numerical test',alpha=alpha,
            predicted_epigraph_reduction=mu-epigraph,correction_error=correction_error);return
    point=np.r_[direction,epigraph];pl,ph=enclosure(M,point,origin)
    upper_violation=float((ph-a['rhs']).max())
    # Final exact primal audit decides; this floating screen never clips or retries.
    if upper_violation>1e-9:
        fail('Full-original finite-point numerical screen failed',upper_violation=upper_violation);return
    A=sparse.load_npz(PRIOR/'A.npz');rows=state['r']+A@direction;closure=h+C@direction
    ra=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');soft_mask=ra['keep']&ra['soft']
    theta=max(0.,float(-rows[soft_mask].min()),float(abs(closure).max()))
    if not np.isfinite(theta) or not mu-theta>1e-9:fail('Full modeled merit did not improve by the numerical test');return
    np.save(O/'constructed_point.npy',point)
    write(O/'candidate.json',dict(diagnostic_passed=True,alpha=alpha,dyadic_exponent=k,mu=mu,
        modeled_theta=theta,modeled_reduction=mu-theta,epigraph=epigraph,
        correction_error=correction_error,upper_violation=upper_violation,
        primary_optimality_checked=False,full_KKT_passed=False,controller_admitted=False,
        interior_solve=False,finite_encoding=False,actual_model_evaluation=False,no_retry=True))
    event('producer_completed',origin)

if __name__=='__main__':main()
