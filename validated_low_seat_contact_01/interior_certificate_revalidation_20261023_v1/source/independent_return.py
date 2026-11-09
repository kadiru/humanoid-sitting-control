"""Independent correction identity, unchanged full-space KKT and exact margin audit."""
from common import *
from fractions import Fraction

def check_change(original,corrected,row,expected_hex):
 import numpy as np
 assert set(original)==set(corrected)=={'inequality','equality','lower','upper'}
 for name in original:
  a,b=original[name],corrected[name];assert a.shape==b.shape and a.dtype==b.dtype==np.float64
  if name!='inequality':identical(a,b)
  else:
   assert float(a[row]).hex()==expected_hex
   assert int(b.view(np.uint64)[row])==0
   identical(a[:row],b[:row]);identical(a[row+1:],b[row+1:])
 return True

def identical(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()

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
 for p,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():assert sha(p)==h,p;check(origin)
 assert read(O/'producer_exit.json')['completed_within_deadline']
 assert sha(REJECTED/'backend_primal.npy')==POINT_SHA
 change=read(O/'certificate_change.json');assert change['row']==ROW and change['source_hex']==RAW_HEX and change['replacement_hex']=='0x0.0p+0'
 assert change['changed_dual_values']==1 and change['changed_primal_values']==0 and change['source_primal_sha256']==POINT_SHA
 import numpy as np
 from scipy import sparse
 original=dict(np.load(REJECTED/'backend_marginals.npz',allow_pickle=False));corrected=dict(np.load(O/'corrected_marginals.npz',allow_pickle=False))
 assert check_change(original,corrected,ROW,RAW_HEX)
 x=np.load(REJECTED/'backend_primal.npy',allow_pickle=False);assert x.shape==(78791,) and x.dtype==np.float64 and np.isfinite(x).all()
 M=sparse.load_npz(REJECTED/'matrix.npz');E=sparse.load_npz(REJECTED/'equality.npz');a=dict(np.load(REJECTED/'arrays.npz',allow_pickle=False))
 assert M.shape==(837063,78791) and M.nnz<=3050574 and E.shape==(1366,78791) and E.nnz==0
 numerical=audit_kkt(M,E,a,x,corrected);exact=None
 if numerical['passed']:
  H=sparse.load_npz(PRIOR/'H.npz').tocsr();assert H.shape==(92267,78790) and H.nnz==125867
  exact=audit_exact(H,read(PRIOR/'intervals.json'),x,origin)
 else:assert not (O/'exact_margin.json').exists()
 accepted=bool(numerical['passed'] and exact and exact['passed']);assert read(O/'candidate.json')['accepted']==accepted
 if accepted:assert sha(O/'witness.npy')==POINT_SHA
 else:assert not (O/'witness.npy').exists()
 write(O/'audit.json',dict(passed=True,candidate_accepted=accepted,correction_identity_passed=True,changed_dual_values=1,unchanged_primal_sha256=POINT_SHA,full_original_numerical=numerical,exact_margin=exact,independent_integer_ratio_sum_executed=exact is not None,solver_calls=0,finite_maps=0,model_calls=0,physical_admission=False))
 check(origin)
if __name__=='__main__':main()
