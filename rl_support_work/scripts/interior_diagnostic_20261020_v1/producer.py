"""One cold full-original interior solve; no decoder, mechanics or old proof replay."""
import sys
import time
import resource
from fractions import Fraction
from common import (BASE, D, HERE, LIMIT, O, PRIOR, PRIMARY_SHA, ROWS, STATE,
                    check_time, event, inventory, phase, read, sha, write)


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
                check_time(origin, 600)
            start, end = H.indptr[row:row+2]
            value = sum((f(a) * coordinates[j]
                         for a, j in zip(H.data[start:end], H.indices[start:end])),
                        Fraction(0))
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


def main(origin):
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    spec = phase(origin, "numerical")
    inventory(spec, origin, 600)
    # All scientific imports, input loads, copies and new input auditing are timed.
    import numpy as np
    from scipy import sparse
    from scipy.optimize import linprog
    from independent import audit_new_input

    primary_path = PRIOR / "primary/lp/witness.npy"
    assert sha(primary_path) == PRIMARY_SHA
    primary = np.load(primary_path, allow_pickle=False)
    assert primary.dtype == np.dtype("float64") and primary.shape == (78791,)
    assert np.isfinite(primary).all() and primary[-1] == 0
    state = dict(np.load(STATE / "state.npz", allow_pickle=False))
    rows = dict(np.load(ROWS, allow_pickle=False))
    A = sparse.load_npz(STATE / "A.npz").tocsr()
    C = sparse.load_npz(STATE / "C.npz").tocsr()
    predicted_rows = state["r"] + A @ primary[:-1]
    predicted_consistency = state["h"] + C @ primary[:-1]
    soft = rows["keep"] & rows["soft"]
    assert np.isfinite(predicted_rows).all() and np.isfinite(predicted_consistency).all()
    theta = max(0., float(-predicted_rows[soft].min()),
                float(abs(predicted_consistency).max()))
    mu = read(BASE / "preparation.json")["initial_phi"]
    eligible = theta < mu
    ceiling = (mu + theta)/2 if eligible else None
    np.savez(O / "primary_prediction.npz", rows=predicted_rows,
             consistency=predicted_consistency)
    write(O / "ceiling.json", dict(mu=mu, theta=theta, ceiling=ceiling,
                                   eligible=eligible, primary_sha256=PRIMARY_SHA))
    if not eligible:
        # Independently check this new eligibility result even without an LP.
        audit_new_input(origin, assembled=False)
        write(O / "worker_result.json", dict(outcome="primary_not_eligible",
              accepted=False, cold_solves=0, finite_maps=0, model_calls=0))
        return
    M = sparse.load_npz(BASE / "witness/matrix.npz").tocsr()
    E = sparse.load_npz(BASE / "witness/equality.npz").tocsr()
    arrays = dict(np.load(BASE / "witness/arrays.npz", allow_pickle=False))
    keep = rows["keep"]
    k, m = int(keep.sum()), C.shape[0]
    assert k == 331495 and m == 68250
    rhs = arrays["rhs"].copy()
    # Fresh construction preserves the original operation order; no old-ceiling subtraction.
    rhs[:k] = state["r"][keep] - state["guard"][keep] + rows["soft"][keep]*ceiling
    rhs[k:k+m] = -state["h"] + ceiling
    rhs[k+m:k+2*m] = state["h"] + ceiling
    arrays["rhs"] = rhs
    np.savez(O / "input_arrays.npz", **arrays)
    audit_new_input(origin, assembled=True)
    event("new_input_audit_complete", origin)
    # Cold means no x0, initial basis, retained internal state, or prior solution.
    configured_elapsed = time.monotonic()-origin
    remaining = 300-configured_elapsed
    assert remaining > 0
    options = dict(time_limit=remaining, maxiter=4096, presolve=True,
                   primal_feasibility_tolerance=1e-9,
                   dual_feasibility_tolerance=1e-9, ipm_optimality_tolerance=1e-9)
    write(O / "scipy_started.json", dict(method="highs-ipm", cold=True,
          options=options, elapsed_s=time.monotonic()-origin,
          configured_elapsed_s=configured_elapsed,
          input_audit_sha256=sha(O / "input_audit.json")))
    check_time(origin, 300)
    backend_start = time.monotonic()
    result = linprog(arrays["cost"], A_ub=M, b_ub=rhs, A_eq=E,
                     b_eq=arrays["equality_rhs"], bounds=arrays["bounds"],
                     method="highs-ipm", options=options)
    returned = event("backend_returned", origin,
                     backend_elapsed_s=time.monotonic()-backend_start)
    write(O / "backend_result.json", dict(status=int(result.status),
          success=bool(result.success), message=str(result.message),
          nit=int(result.nit), crossover_nit=int(result.crossover_nit),
          objective=None if result.fun is None else float(result.fun),
          elapsed_s=returned["elapsed_s"]))
    if result.x is not None:
        np.save(O / "backend_primal.npy", result.x)
    if not result.success or result.status != 0 or returned["elapsed_s"] >= 300:
        write(O / "worker_result.json", dict(outcome="backend_not_accepted",
              accepted=False, cold_solves=1, finite_maps=0, model_calls=0))
        return
    x = result.x
    assert x.dtype == np.dtype("float64") and x.shape == (78791,)
    assert np.isfinite(x).all()
    marginals = dict(inequality=result.ineqlin.marginals,
                     equality=result.eqlin.marginals,
                     lower=result.lower.marginals, upper=result.upper.marginals)
    np.savez(O / "backend_marginals.npz", **marginals)
    numerical = numerical_check(M, E, arrays, x, marginals)
    if not numerical["passed"]:
        write(O / "worker_result.json", dict(outcome="numerical_check_failed",
              accepted=False, cold_solves=1, finite_maps=0, model_calls=0))
        return
    H = sparse.load_npz(STATE / "H.npz").tocsr()
    assert H.shape == (92267, 78790) and H.nnz == 125867
    certificate = exact_margin(H, read(STATE / "intervals.json"), x[:-1], origin)
    check_time(origin, 600)
    write(O / "worker_result.json", dict(
        outcome="finite_margin_passed" if certificate["passed"] else "exact_margin_failed",
        accepted=certificate["passed"], numerical_eta=float(x[-1]),
        cold_solves=1, finite_maps=0, model_calls=0, physical_success=False))


if __name__ == "__main__":
    main(float(sys.argv[1]))
