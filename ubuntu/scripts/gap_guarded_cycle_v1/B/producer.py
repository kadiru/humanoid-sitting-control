"""One new-state interior solve; strengthened numerical and exact margin gates."""
from common import *
from fractions import Fraction
def numerical_check(M, E, a, x, marginal):
    import numpy as np
    y = -marginal["inequality"]
    nu = -marginal["equality"]
    upper = -marginal["upper"]
    lower = marginal["lower"]
    assert y.shape == (M.shape[0],) and nu.shape == (E.shape[0],)
    assert upper.shape == lower.shape == x.shape == (78791,)
    row = M @ x - a["rhs"]
    equality = E @ x - a["equality_rhs"]
    stationarity = a["cost"] + M.T @ y + E.T @ nu + upper - lower
    lo = np.isfinite(a["bounds"][:, 0])
    hi = np.isfinite(a["bounds"][:, 1])
    primal = max(0., float(row.max(initial=0)),
                 float(abs(equality).max(initial=0)),
                 float((a["bounds"][:, 0]-x).max()),
                 float((x-a["bounds"][:, 1]).max()))
    dual = float(abs(stationarity).max(initial=0))
    gap = abs(float(a["cost"] @ x + a["rhs"] @ y
                    + a["equality_rhs"] @ nu
                    + a["bounds"][hi, 1] @ upper[hi]
                    - a["bounds"][lo, 0] @ lower[lo]))
    finite = all(np.isfinite(z).all() for z in
                 (x, y, nu, upper, lower, row, equality, stationarity,
                  primal, dual, gap))
    signs = bool((y >= 0).all() and (upper >= 0).all() and (lower >= 0).all()
                 and (upper[~hi] == 0).all() and (lower[~lo] == 0).all())
    metrics = dict(primal=primal, dual=dual, absolute_gap=gap,
                   finite=bool(finite), dual_signs=signs,
                   passed=bool(finite and signs and max(primal, dual, gap) <= 1e-9))
    # Nonfinite metrics terminate rather than being serialized as JSON NaN.
    assert finite, "Nonfinite full-original return"
    np.savez(O / "residuals.npz", row=row, equality=equality,
             stationarity=stationarity)
    write(O / "numerical_checks.json", metrics)
    return metrics

def exact_margin(H, metadata, direction, origin):
    resources = read(HERE / "resources.json")
    f = lambda value: Fraction.from_float(float(value))
    coordinates = list(map(f, direction))
    eta = Fraction(1, 2)
    limiting = None
    bad = []
    max_bits = 0
    with (O / "exact_rows.jsonl").open("x") as output:
        for row in range(H.shape[0]):
            if row % 256 == 0:
                check(origin)
            start, end = H.indptr[row:row+2]
            value = sum((f(a) * coordinates[j]
                         for a, j in zip(H.data[start:end], H.indices[start:end])),
                        Fraction(0))
            assert abs(value.numerator).bit_length() <= 4213 and value.denominator.bit_length() <= 2149
            low = Fraction(*metadata["lower"][row])
            high = Fraction(*metadata["upper"][row])
            width = Fraction(*metadata["widths"][row])
            assert width > 0
            clearance = min(value-low, high-value)/width
            if clearance < eta:
                eta, limiting = clearance, row
            if clearance <= 0:
                bad.append(row)
            max_bits = max(max_bits, *(max(abs(q.numerator).bit_length(),
                                          q.denominator.bit_length())
                                      for q in (value, clearance)))
            output.write(f"[{row},{value.numerator},{value.denominator}]\n")
    result = dict(passed=eta > 0, exact_eta=[eta.numerator, eta.denominator],
                  nonpositive=bad, limiting_row=limiting, rows=H.shape[0],
                  coefficient_products=H.nnz, observed_max_rational_bits=max_bits,
                  direction_scaled=False, finite_map_performed=False,
                  rows_sha256=sha(O / "exact_rows.jsonl"))
    assert max_bits <= max(resources["clearance_numerator_bits_upper_bound"],
                           resources["clearance_denominator_bits_upper_bound"])
    assert (O / "exact_rows.jsonl").stat().st_size <= resources["exact_trace_total_bytes_upper_bound"]
    write(O / "exact_margin.json", result)
    return result

def main():
 origin=admission('producer');spec=contract()
 verify_static(spec,origin,check); inherited_input(origin,check); bind_predecessor(PRIMARY,origin,check)
 from builder import main_input
 from independent_input import prove
 M,E,a=main_input(origin);prove(origin);event('new_input_independently_audited',origin)
 import numpy as np
 from scipy import sparse
 from scipy.optimize import linprog
 remaining=SOLVE_END-(time.monotonic()-origin);assert remaining>0
 options=dict(time_limit=remaining,maxiter=4096,presolve=True,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
 write(O/'scipy_started.json',dict(method='highs-ipm',cold=True,x0_or_basis=False,options=options,input_audit_sha256=sha(O/'input_audit.json'),elapsed_s=time.monotonic()-origin))
 started=time.monotonic()
 r=linprog(a['cost'],A_ub=M,b_ub=a['rhs'],A_eq=E,b_eq=a['equality_rhs'],bounds=a['bounds'],method='highs-ipm',options=options)
 write(O/'backend_result.json',dict(status=int(r.status),success=bool(r.success),candidate_returned=r.x is not None,message=str(r.message),iterations=int(r.nit),crossover_iterations=int(getattr(r,'crossover_nit',0)),seconds=time.monotonic()-started,total_elapsed_s=time.monotonic()-origin))
 check(origin)
 if not(r.success and r.status==0 and r.x is not None):
  write(O/'candidate.json',dict(accepted=False,reason='No successful complete backend point',no_retry=True));return
 x=np.asarray(r.x);np.save(O/'backend_primal.npy',x)
 assert x.shape==(78791,) and x.dtype==np.dtype('float64') and np.isfinite(x).all()
 marginal=dict(inequality=r.ineqlin.marginals,equality=r.eqlin.marginals,lower=r.lower.marginals,upper=r.upper.marginals)
 np.savez(O/'backend_marginals.npz',**marginal)
 numerical=numerical_check(M,E,a,x,marginal)
 if not numerical['passed']:
  write(O/'candidate.json',dict(accepted=False,reason='Strengthened numerical gate failed',no_retry=True));return
 H=sparse.load_npz(PRIOR/'H.npz').tocsr();assert H.shape==(92267,78790) and H.nnz==125867
 slacks=np.load(O/'arrays.npz')['rhs'][-2:] - M[-2:] @ x
 if not np.all(slacks>=0):
  write(O/'candidate.json',dict(accepted=False,reason='Negative buffered gap slack',no_retry=True));return
 certificate=exact_margin(H,read(PRIOR/'intervals.json'),x[:-1],origin)
 accepted=bool(certificate['passed'])
 if accepted:np.save(O/'witness.npy',x)
 write(O/'candidate.json',dict(accepted=accepted,reason='Exact unscaled interior margin' if accepted else 'Nonpositive exact margin',no_retry=True,numerical_eta=float(x[-1])))
 event('producer_completed',origin)
if __name__=='__main__':main()
