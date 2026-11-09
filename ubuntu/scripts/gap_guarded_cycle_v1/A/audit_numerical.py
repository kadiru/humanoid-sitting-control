"""Independent full original KKT and integer-accumulation exact margin audit."""
from common import *
from fractions import Fraction
def identical(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()
def audit_kkt(M, E, arrays, point, marginal):
    import numpy as np

    for name, shape in (("inequality", (M.shape[0],)), ("equality", (E.shape[0],)),
                        ("lower", (M.shape[1],)), ("upper", (M.shape[1],))):
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
    for key, value in (("inequality", residual), ("equality", eq_residual),
                       ("stationarity", gradient)):
        identical(value, saved[key])
    assert metrics == read(O / "numerical_checks.json")
    return metrics
