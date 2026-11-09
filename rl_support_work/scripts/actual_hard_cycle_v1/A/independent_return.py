"""Separate strengthened KKT and full original primal audit; no producer import."""
from common import *
from fractions import Fraction

def main():
    origin=admission('audit');spec=contract();verify_static(spec,origin,check)
    assert read(O/'producer_exit.json')['completed_within_deadline']
    marker=read(O/'scipy_started.json');backend=read(O/'backend_result.json')
    decision=read(O/'candidate.json')
    assert marker['cold'] and not marker['x0_or_basis'] and marker['method']=='highs-ipm'
    assert marker['options']==solver_options(marker['options']['time_limit'])
    if not (backend['success'] and backend['status']==0 and backend['candidate_returned']):
        assert not decision['accepted'] and not (O/'witness.npy').exists()
        write(O/'audit.json',dict(passed=True,candidate_accepted=False,scope='Failure accounting only',physical_admission=False));return
    import numpy as np
    from scipy import sparse
    from audit_numerical import audit_kkt
    x=np.load(O/'backend_primal.npy',allow_pickle=False)
    assert x.shape==(78791,) and x.dtype==np.dtype('float64')
    m=dict(np.load(O/'backend_marginals.npz',allow_pickle=False))
    import common as c
    from reserve_bridge import audit_extension,point_certificate,sparse_same
    proof=read(O/'input_audit.json');assert proof['passed'] and proof['original_prefix_preserved']
    for name,h in proof['files_sha256'].items():assert sha(O/name)==h
    assert proof['hard_input_audit_sha256']==sha(O/'hard_input_audit.json')
    M=sparse.load_npz(O/'matrix.npz');E=sparse.load_npz(O/'equality.npz')
    a=dict(np.load(O/'arrays.npz',allow_pickle=False))
    base=sparse.load_npz(CURRENT/'augmented_matrix.npz')
    base_a=dict(np.load(CURRENT/'augmented_arrays.npz',allow_pickle=False))
    audit_extension(c,origin,M,a,base,base_a)
    sparse_same(E,sparse.load_npz(CURRENT/'full_equality.npz'))
    numerical=audit_kkt(M,E,a,x,m)
    # A second implementation evaluates the original primal directly.
    original=sparse.load_npz(CURRENT/'full_matrix.npz')
    oa=dict(np.load(CURRENT/'full_arrays.npz',allow_pickle=False))
    v=original.dot(x)-oa['rhs']; eq=E.dot(x)-oa['equality_rhs']
    primal=max(0.,float(np.max(v,initial=0)),float(np.max(np.abs(eq),initial=0)),
        float(np.max(oa['bounds'][:,0]-x)),float(np.max(x-oa['bounds'][:,1])))
    G=sparse.load_npz(CURRENT/'target_matrix.npz');g=np.load(CURRENT/'target_rhs.npy',allow_pickle=False)
    slacks=g-G.dot(x)
    state=dict(np.load(CURRENT/'state.npz',allow_pickle=False))
    row=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz',allow_pickle=False)
    soft=np.logical_and(row['keep'],row['soft'])
    r=np.add(state['r'],sparse.load_npz(CURRENT/'A.npz').dot(x[:-1]))
    h=np.add(state['h'],sparse.load_npz(CURRENT/'C.npz').dot(x[:-1]))
    theta=max(0.,float(np.max(-r[soft])),float(np.max(np.abs(h))))
    mu=read(ACCEPTED/'decision.json')['summary']['phi']
    progress=bool(np.isfinite(theta) and Fraction(theta)<=Fraction(mu)*Fraction(99,100))
    hard=point_certificate(c,origin,x,independent=True) if numerical['passed'] else None
    accepted=bool(numerical['passed'] and primal<=1e-9 and np.all(slacks>=0) and progress
                  and hard and hard['passed'])
    assert decision==dict(accepted=accepted,mu=mu,theta=theta,original_primal=primal,
        target_slacks=slacks.tolist(),progress=progress,optimum_scope='strengthened_problem_only',
        hard_exact_pass=bool(hard and hard['passed']),physical_success=False)
    if accepted:
        witness=np.load(O/'witness.npy',allow_pickle=False)
        assert witness.dtype==x.dtype and witness.shape==x.shape and witness.tobytes()==x.tobytes()
        assert read(O/'ceiling.json')==dict(eligible=True,mu=mu,theta=theta,ceiling=(mu+theta)/2)
    else:
        assert not (O/'witness.npy').exists() and not (O/'ceiling.json').exists()
    event('audit_completed',origin)
    write(O/'audit.json',dict(passed=True,candidate_accepted=accepted,strengthened_numerical=numerical,
        original_primal=primal,target_slacks=slacks.tolist(),mu=mu,theta=theta,
        unchanged_primal_sha256=sha(O/'backend_primal.npy'),hard_exact_certificate=hard,physical_admission=False))

if __name__=='__main__':main()
