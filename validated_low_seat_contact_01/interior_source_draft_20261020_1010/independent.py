"""Independent new-input and return checks; imports no producer or optimizer."""
import sys
import time
import resource
from fractions import Fraction
from common import (BASE, D, LIMIT, O, PRIOR, PRIMARY_SHA, ROWS, STATE,
                    check_time, inventory, phase, read, sha, write)


def identical(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def sparse_identical(a, b):
    assert a.shape == b.shape
    identical(a.indptr, b.indptr)
    identical(a.indices, b.indices)
    identical(a.data, b.data)


def audit_new_input(origin, assembled):
    import numpy as np
    from scipy import sparse

    check_time(origin, 300)
    for proof in ("lifted_feasibility_audit_491/independent_audit.json",
                  "torque_path_audit_525/independent_audit.json"):
        assert read(D / proof)["passed"] is True
    reference = dict(np.load(BASE / "reference.npz", allow_pickle=False))
    state = dict(np.load(STATE / "state.npz", allow_pickle=False))
    for key in ("features", "actualY", "J", "r", "e", "h", "guard"):
        identical(state[key], reference[key])
    A = sparse.load_npz(STATE / "A.npz").tocsr()
    C = sparse.load_npz(STATE / "C.npz").tocsr()
    H = sparse.load_npz(STATE / "H.npz").tocsr()
    for matrix, name in ((A, "all_rows_chain"), (C, "consistency_chain"),
                         (H, "interval_matrix")):
        sparse_identical(matrix, sparse.load_npz(BASE / (name+".npz")).tocsr())
    old_intervals = read(BASE / "intervals.json")
    saved_intervals = read(STATE / "intervals.json")
    assert saved_intervals["trust"] == 1
    assert set(saved_intervals) == set(old_intervals) | {"trust"}
    assert all(saved_intervals[key] == value for key, value in old_intervals.items())
    assert H.shape == (92267, 78790)
    point_path = PRIOR / "primary/lp/witness.npy"
    assert sha(point_path) == PRIMARY_SHA
    point = np.load(point_path, allow_pickle=False)
    assert point.dtype == np.dtype("float64") and point.shape == (78791,)
    assert np.isfinite(point).all() and point[-1] == 0
    row = dict(np.load(ROWS, allow_pickle=False))
    keep, soft = row["keep"], row["soft"]
    assert keep.dtype == soft.dtype == np.dtype("bool")
    predicted_rate = reference["r"] + A.dot(point[:78790])
    predicted_closure = reference["h"] + C.dot(point[:78790])
    assert np.isfinite(predicted_rate).all() and np.isfinite(predicted_closure).all()
    predicted = dict(np.load(O / "primary_prediction.npz", allow_pickle=False))
    identical(predicted_rate, predicted["rows"])
    identical(predicted_closure, predicted["consistency"])
    theta = max(0., -float(np.min(predicted_rate[np.logical_and(keep, soft)])),
                float(np.max(np.abs(predicted_closure))))
    mu = read(BASE / "preparation.json")["initial_phi"]
    assert mu == 0.19791129319207476
    eligible = bool(theta < mu)
    ceiling = (mu + theta)/2 if eligible else None
    assert read(O / "ceiling.json") == dict(mu=mu, theta=theta, ceiling=ceiling,
                                          eligible=eligible, primary_sha256=PRIMARY_SHA)
    assert assembled == eligible
    files = [O / "ceiling.json", O / "primary_prediction.npz"]
    if assembled:
        inherited = dict(np.load(BASE / "witness/arrays.npz", allow_pickle=False))
        proposed = dict(np.load(O / "input_arrays.npz", allow_pickle=False))
        assert set(proposed) == set(inherited) == {"rhs", "cost", "bounds", "equality_rhs"}
        for key in ("cost", "bounds", "equality_rhs"):
            identical(proposed[key], inherited[key])
        k, m = int(np.count_nonzero(keep)), C.shape[0]
        assert k == 331495 and m == 68250
        # Independently construct the three changed blocks, not a rounded RHS adjustment.
        first = np.subtract(reference["r"][keep], reference["guard"][keep])
        first = np.add(first, np.multiply(soft[keep].astype(float), ceiling))
        expected = np.concatenate((first, np.add(-reference["h"], ceiling),
                                   np.add(reference["h"], ceiling)))
        identical(expected, proposed["rhs"][:k+2*m])
        identical(proposed["rhs"][k+2*m:], inherited["rhs"][k+2*m:])
        M = sparse.load_npz(BASE / "witness/matrix.npz")
        E = sparse.load_npz(BASE / "witness/equality.npz")
        assert M.shape == (837063, 78791) and M.nnz == 2738418
        assert E.shape == (1366, 78791) and E.nnz == 0
        assert proposed["rhs"].shape == (837063,)
        assert proposed["cost"].shape == (78791,)
        assert proposed["bounds"].shape == (78791, 2)
        assert proposed["equality_rhs"].shape == (1366,)
        assert np.all(proposed["cost"][:-1] == 0) and proposed["cost"][-1] == -1
        assert np.all(np.isneginf(proposed["bounds"][:-1, 0]))
        assert np.all(np.isposinf(proposed["bounds"][:-1, 1]))
        assert np.array_equal(proposed["bounds"][-1], [0., .5])
        assert np.all(proposed["equality_rhs"] == 0)
        assert np.isfinite(M.data).all() and np.isfinite(proposed["rhs"]).all()
        files.append(O / "input_arrays.npz")
    result = dict(passed=True, assembled=assembled, eligible=eligible,
                  new_point_evaluated_once_by_auditor=True,
                  previous_coefficient_and_interval_proofs_inherited=True,
                  previous_proofs_replayed=False, initial_state_bitwise_identical=True,
                  actual_ceiling=ceiling, theta=theta, changed_rhs_entries=467995 if assembled else 0,
                  all_original_rows_retained=True,
                  files_sha256={str(p): sha(p) for p in files},
                  elapsed_s=time.monotonic()-origin)
    check_time(origin, 300)
    write(O / "input_audit.json", result)
    return result


def audit_kkt(M, E, arrays, point, marginal):
    import numpy as np

    for name, shape in (("inequality", (837063,)), ("equality", (1366,)),
                        ("lower", (78791,)), ("upper", (78791,))):
        assert marginal[name].shape == shape
        assert marginal[name].dtype == np.dtype("float64")
        assert np.isfinite(marginal[name]).all()
    row_dual = np.negative(marginal["inequality"])
    eq_dual = np.negative(marginal["equality"])
    up_dual = np.negative(marginal["upper"])
    low_dual = marginal["lower"].copy()
    residual = M.dot(point) - arrays["rhs"]
    eq_residual = E.dot(point) - arrays["equality_rhs"]
    gradient = arrays["cost"] + M.transpose().dot(row_dual)
    gradient = gradient + E.transpose().dot(eq_dual) + up_dual - low_dual
    lower_finite = np.isfinite(arrays["bounds"][:, 0])
    upper_finite = np.isfinite(arrays["bounds"][:, 1])
    primal = max(0., float(np.max(residual, initial=0)),
                 float(np.max(np.abs(eq_residual), initial=0)),
                 float(np.max(arrays["bounds"][:, 0]-point)),
                 float(np.max(point-arrays["bounds"][:, 1])))
    dual = float(np.max(np.abs(gradient), initial=0))
    gap = abs(float(arrays["cost"].dot(point) + arrays["rhs"].dot(row_dual)
                    + arrays["equality_rhs"].dot(eq_dual)
                    + arrays["bounds"][upper_finite, 1].dot(up_dual[upper_finite])
                    - arrays["bounds"][lower_finite, 0].dot(low_dual[lower_finite])))
    finite = all(np.isfinite(a).all() for a in
                 (point, row_dual, eq_dual, up_dual, low_dual,
                  residual, eq_residual, gradient, primal, dual, gap))
    signs = bool(np.all(row_dual >= 0) and np.all(up_dual >= 0)
                 and np.all(low_dual >= 0) and np.all(up_dual[~upper_finite] == 0)
                 and np.all(low_dual[~lower_finite] == 0))
    metrics = dict(primal=primal, dual=dual, absolute_gap=gap, finite=bool(finite),
                   dual_signs=signs,
                   passed=bool(finite and signs and max(primal, dual, gap) <= 1e-9))
    saved = dict(np.load(O / "residuals.npz", allow_pickle=False))
    for key, value in (("row", residual), ("equality", eq_residual),
                       ("stationarity", gradient)):
        identical(value, saved[key])
    assert metrics == read(O / "numerical_checks.json")
    return metrics


def audit_exact(H, intervals, point, origin):
    # Independent dyadic integer accumulation, without Fraction dot products.
    pairs = [float(x).as_integer_ratio() for x in point[:-1]]
    eta = Fraction(1, 2)
    limiting = None
    nonpositive = []
    maximum_bits = 0
    with (O / "exact_rows.jsonl").open() as rows:
        for row in range(H.shape[0]):
            if row % 256 == 0:
                check_time(origin, 600)
            accumulator, exponent = 0, 0
            for offset in range(H.indptr[row], H.indptr[row+1]):
                a, b = float(H.data[offset]).as_integer_ratio()
                c, d = pairs[H.indices[offset]]
                power = b.bit_length() + d.bit_length() - 2
                term = a*c
                if power > exponent:
                    accumulator <<= power-exponent
                    exponent = power
                accumulator += term << (exponent-power)
            value = Fraction(accumulator, 1 << exponent)
            assert read_line(rows) == [row, value.numerator, value.denominator]
            low = Fraction(*intervals["lower"][row])
            high = Fraction(*intervals["upper"][row])
            width = Fraction(*intervals["widths"][row])
            assert width > 0
            clearance = min(value-low, high-value)/width
            if clearance < eta:
                eta, limiting = clearance, row
            if clearance <= 0:
                nonpositive.append(row)
            for q in (value, clearance):
                maximum_bits = max(maximum_bits, abs(q.numerator).bit_length(),
                                   q.denominator.bit_length())
        assert rows.read() == ""
    result = dict(passed=eta > 0, exact_eta=[eta.numerator, eta.denominator],
                  nonpositive=nonpositive, limiting_row=limiting, rows=H.shape[0],
                  coefficient_products=H.nnz, observed_max_rational_bits=maximum_bits,
                  direction_scaled=False, finite_map_performed=False,
                  rows_sha256=sha(O / "exact_rows.jsonl"))
    assert result == read(O / "exact_margin.json")
    return result


def read_line(stream):
    import json
    return json.loads(next(stream))


def main(origin):
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    spec = phase(origin, "audit")
    inventory(spec, origin, 600)
    termination = read(O / "numerical_exit.json")
    # A killed, late or failed worker has no admissible candidate, even with partial files.
    if not termination["completed_within_deadline"]:
        write(O / "audit_result.json", dict(passed=True, candidate_accepted=False,
              reason="Numerical phase incomplete; no partial certificate promoted"))
        return
    outcome = read(O / "worker_result.json")
    proof = read(O / "input_audit.json")
    assert proof["passed"] and proof["elapsed_s"] < 300
    for path, digest in proof["files_sha256"].items():
        assert sha(path) == digest
    # New point/RHS association was independently checked before the backend;
    # inherit that certificate here instead of evaluating the same point a third time.
    if not proof["eligible"]:
        assert outcome["cold_solves"] == 0 and not outcome["accepted"]
        assert not (O / "scipy_started.json").exists()
        write(O / "audit_result.json", dict(passed=True, candidate_accepted=False,
              reason="Independently checked primary has no modeled improvement"))
        return
    started = read(O / "scipy_started.json")
    assert started["input_audit_sha256"] == sha(O / "input_audit.json")
    assert started["cold"] and started["method"] == "highs-ipm"
    options = started["options"]
    assert 0 <= started["configured_elapsed_s"] <= started["elapsed_s"] < 300
    assert options["time_limit"] == 300-started["configured_elapsed_s"]
    assert options["maxiter"] == 4096 and options["presolve"] is True
    assert all(options[key] == 1e-9 for key in (
        "primal_feasibility_tolerance", "dual_feasibility_tolerance",
        "ipm_optimality_tolerance"))
    backend = read(O / "backend_result.json")
    assert outcome["cold_solves"] == 1
    if not backend["success"] or backend["status"] != 0 or backend["elapsed_s"] >= 300:
        assert not outcome["accepted"]
        write(O / "audit_result.json", dict(passed=True, candidate_accepted=False,
              reason="Backend did not return an acceptable full candidate within its cap"))
        return
    import numpy as np
    from scipy import sparse

    point = np.load(O / "backend_primal.npy", allow_pickle=False)
    assert point.shape == (78791,) and point.dtype == np.dtype("float64")
    assert np.isfinite(point).all()
    M = sparse.load_npz(BASE / "witness/matrix.npz").tocsr()
    E = sparse.load_npz(BASE / "witness/equality.npz").tocsr()
    arrays = dict(np.load(O / "input_arrays.npz", allow_pickle=False))
    marginal = dict(np.load(O / "backend_marginals.npz", allow_pickle=False))
    numerical = audit_kkt(M, E, arrays, point, marginal)
    finite = None
    if numerical["passed"]:
        finite = audit_exact(sparse.load_npz(STATE / "H.npz").tocsr(),
                             read(STATE / "intervals.json"), point, origin)
    else:
        assert not (O / "exact_margin.json").exists()
    accepted = bool(numerical["passed"] and finite and finite["passed"])
    assert outcome["accepted"] == accepted
    assert outcome["finite_maps"] == outcome["model_calls"] == 0
    check_time(origin, 600)
    write(O / "audit_result.json", dict(passed=True, candidate_accepted=accepted,
          full_original_numerical=numerical, exact_margin=finite,
          unchanged_backend_point_sha256=sha(O / "backend_primal.npy"),
          independent_integer_ratio_sum=True, primary_point_evaluations=0,
          physical_success=False, finite_maps=0, model_calls=0))


if __name__ == "__main__":
    main(float(sys.argv[1]))
