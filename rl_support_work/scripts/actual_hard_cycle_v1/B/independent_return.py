"""Independent strengthened KKT and integer-accumulation exact margin audit."""
from common import *
from fractions import Fraction
def identical(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()
def audit_kkt(M, E, arrays, point, marginal):
    import numpy as np

    for name, shape in (("inequality", (993712,)), ("equality", (1366,)),
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
def read_line(stream):
    import json
    return json.loads(next(stream))
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
                check(origin)
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
            assert abs(value.numerator).bit_length() <= 4213 and value.denominator.bit_length() <= 2149
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
    resources = read(HERE / "resources.json")
    assert maximum_bits <= max(resources["clearance_numerator_bits_upper_bound"],
                               resources["clearance_denominator_bits_upper_bound"])
    assert (O / "exact_rows.jsonl").stat().st_size <= resources["exact_trace_total_bytes_upper_bound"]
    assert result == read(O / "exact_margin.json")
    return result

def main():
 origin=admission('audit');spec=contract()
 verify_static(spec,origin,check); bind_predecessor(PRIMARY,origin,check)
 assert read(O/'producer_exit.json')['completed_within_deadline']
 proof=read(O/'input_audit.json');assert proof['passed'] and proof['new_state']
 for n,h in proof['files_sha256'].items():assert sha(O/n)==h,n
 marker=read(O/'scipy_started.json');backend=read(O/'backend_result.json');decision=read(O/'candidate.json')
 assert marker['input_audit_sha256']==sha(O/'input_audit.json')
 assert marker['cold'] and marker['method']=='highs-ipm' and not marker['x0_or_basis']
 opts=marker['options'];assert 0<opts['time_limit']<=SOLVE_END and opts['presolve'] and opts['maxiter']==4096
 assert all(opts[k]==1e-9 for k in ('primal_feasibility_tolerance','dual_feasibility_tolerance','ipm_optimality_tolerance'))
 if not (O/'numerical_checks.json').exists():
  assert not(backend['status']==0 and backend['success'] and backend['candidate_returned'])
  assert not decision['accepted'] and not (O/'witness.npy').exists()
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,scope='Failure accounting only',physical_admission=False));return
 import numpy as np
 from scipy import sparse
 x=np.load(O/'backend_primal.npy',allow_pickle=False);m=dict(np.load(O/'backend_marginals.npz',allow_pickle=False))
 assert backend['success'] and backend['status']==0 and x.dtype==np.dtype('float64') and x.shape==(78791,)
 M=sparse.load_npz(O/'matrix.npz');E=sparse.load_npz(O/'equality.npz');a=dict(np.load(O/'arrays.npz',allow_pickle=False))
 numerical=audit_kkt(M,E,a,x,m);certificate=None
 slacks=a['rhs'][837063:837065] - M[837063:837065] @ x
 target_pass=bool(np.isfinite(slacks).all() and np.all(slacks>=0))
 if numerical['passed'] and target_pass:
  H=sparse.load_npz(PRIOR/'H.npz').tocsr();assert H.shape==(92267,78790) and H.nnz==125867
  certificate=audit_exact(H,read(PRIOR/'intervals.json'),x,origin)
 else:assert not (O/'exact_margin.json').exists()
 from reserve_bridge import point_certificate
 import common as c
 hard=point_certificate(c,origin,x,interior=True,independent=True) if certificate and certificate['passed'] else None
 accepted=bool(numerical['passed'] and target_pass and certificate and certificate['passed'] and hard and hard['passed']);assert decision['accepted']==accepted
 if accepted:assert np.load(O/'witness.npy',allow_pickle=False).tobytes()==x.tobytes()
 else:assert not (O/'witness.npy').exists()
 event('audit_completed',origin)
 write(O/'audit.json',dict(passed=True,candidate_accepted=accepted,strengthened_numerical=numerical,original_primal_pass=numerical['primal']<=1e-9,target_slacks=slacks.tolist(),exact_margin=certificate,hard_exact_certificate=hard,unchanged_backend_point_sha256=sha(O/'backend_primal.npy'),unchanged_primal_sha256=sha(O/'backend_primal.npy'),independent_integer_ratio_sum=True,finite_maps=0,model_calls=0,physical_admission=False))
if __name__=='__main__':main()
