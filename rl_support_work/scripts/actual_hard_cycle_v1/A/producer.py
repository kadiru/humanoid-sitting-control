"""One approved cold primary solve; inherit already audited input, never rebuild it."""
from common import *
from fractions import Fraction as F

def main():
    origin = admission('producer'); spec = contract()
    verify_static(spec, origin, check); inherited_input(origin, check)
    import numpy as np
    from scipy import sparse
    from scipy.optimize import linprog
    from numerical_checks import check_point
    M = sparse.load_npz(CURRENT/'augmented_matrix.npz').tocsr()
    E = sparse.load_npz(CURRENT/'full_equality.npz').tocsr()
    a = dict(np.load(CURRENT/'augmented_arrays.npz', allow_pickle=False))
    assert M.shape == (652531, 78791) and M.nnz == 2614318
    assert E.shape == (1366, 78791) and E.nnz == 0
    import common as c
    from reserve_bridge import append,audit_extension,point_certificate
    base_M,base_a=M,a
    original_bounds=a['bounds'].copy()
    M,a=append(c,origin,base_M,base_a)
    audit_extension(c,origin,M,a,base_M,base_a)
    sparse.save_npz(O/'matrix.npz',M);sparse.save_npz(O/'equality.npz',E)
    np.savez(O/'arrays.npz',**a)
    write(O/'input_audit.json',dict(passed=True,original_prefix_preserved=True,
        hard_input_audit_sha256=sha(O/'hard_input_audit.json'),
        files_sha256={n:sha(O/n) for n in ('matrix.npz','equality.npz','arrays.npz')}))
    del base_M,base_a
    options = solver_options(SOLVE_END-(time.monotonic()-origin))
    write(O/'scipy_started.json', dict(method='highs-ipm', cold=True,
        x0_or_basis=False, options=options, elapsed_s=time.monotonic()-origin))
    r = linprog(a['cost'], A_ub=M, b_ub=a['rhs'], A_eq=E,
        b_eq=a['equality_rhs'], bounds=a['bounds'], method='highs-ipm', options=options)
    write(O/'backend_result.json', dict(status=int(r.status), success=bool(r.success),
        candidate_returned=r.x is not None, message=str(r.message), iterations=int(r.nit),
        crossover_iterations=int(getattr(r, 'crossover_nit', 0)), elapsed_s=time.monotonic()-origin))
    check(origin)
    if not (r.success and r.status == 0 and r.x is not None):
        write(O/'candidate.json', dict(accepted=False, reason='No successful complete return')); return
    x = np.asarray(r.x)
    m = dict(inequality=r.ineqlin.marginals, equality=r.eqlin.marginals,
        lower=r.lower.marginals, upper=r.upper.marginals)
    assert x.dtype == np.dtype('float64') and x.shape == (78791,)
    for key, count in [('inequality',809178),('equality',1366),('lower',78791),('upper',78791)]:
        assert m[key].dtype == x.dtype and m[key].shape == (count,)
    np.save(O/'backend_primal.npy', x); np.savez(O/'backend_marginals.npz', **m)
    numerical, residuals = check_point(M, E, a, x, m)
    np.savez(O/'residuals.npz', **residuals); write(O/'numerical_checks.json', numerical)
    # The original rows are the immutable prefix of this audited augmentation.
    primal = max(0., float(residuals['inequality'][:652529].max(initial=0)),
        float(abs(residuals['equality']).max(initial=0)),
        float((original_bounds[:,0]-x).max()), float((x-original_bounds[:,1]).max()))
    slacks = -residuals['inequality'][652529:652531]
    state = np.load(CURRENT/'state.npz', allow_pickle=False)
    rows = np.load(D/'clock_row_preparation_v2_446/row_arrays.npz', allow_pickle=False)
    values = state['r'] + sparse.load_npz(CURRENT/'A.npz') @ x[:-1]
    h = state['h'] + sparse.load_npz(CURRENT/'C.npz') @ x[:-1]
    theta = max(0., float(-values[rows['keep'] & rows['soft']].min()), float(abs(h).max()))
    mu = read(ACCEPTED/'decision.json')['summary']['phi']
    assert mu == 0.08605330623059242
    progress = bool(np.isfinite(theta) and 100*(F(mu)-F(theta)) >= F(mu))
    hard=point_certificate(c,origin,x) if numerical['passed'] else None
    accepted = bool(numerical['passed'] and primal <= 1e-9 and np.all(slacks >= 0) and progress
                    and hard and hard['passed'])
    write(O/'candidate.json', dict(accepted=accepted, mu=mu, theta=theta,
        original_primal=primal, target_slacks=slacks.tolist(), progress=progress,
        optimum_scope='strengthened_problem_only',hard_exact_pass=bool(hard and hard['passed']), physical_success=False))
    if accepted:
        np.save(O/'witness.npy', x)
        write(O/'ceiling.json', dict(eligible=True, mu=mu, theta=theta, ceiling=(mu+theta)/2))
    event('producer_completed', origin)

if __name__ == '__main__': main()
