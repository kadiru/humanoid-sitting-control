"""Independent integer-ratio reference audit. No producer import or Fraction.

Computes interval endpoints feature by feature, rather than using the
producer's absolute-weight row-margin formula. Pure synthetic-use component.
"""
from math import gcd, isfinite


def q(x):
    if isinstance(x, bool):
        raise TypeError('boolean')
    if isinstance(x, int):
        return (x, 1)
    if isinstance(x, float):
        if not isfinite(x):
            raise ValueError('nonfinite')
        return x.as_integer_ratio()
    return normalize(x.numerator, x.denominator)


def normalize(n, d):
    if d == 0:
        raise ValueError('zero denominator')
    if d < 0:
        n, d = -n, -d
    k = gcd(n, d)
    return (n // k, d // k)


def add(a, b):
    return normalize(a[0]*b[1]+b[0]*a[1], a[1]*b[1])


def mul(a, b):
    return normalize(a[0]*b[0], a[1]*b[1])


def neg(a):
    return (-a[0], a[1])


def le(a, b):
    return a[0]*b[1] <= b[0]*a[1]


def audit(terms, constant, scale, base, transform, delta, actual, profile, eta, reserve, result):
    zero = (0, 1)
    scale = q(scale)
    if le(scale, zero) or not le(zero, q(eta)) or not le(zero, q(reserve)):
        raise ValueError('invalid scale/margin')
    reciprocal = (scale[1], scale[0])
    coefficients = {}
    for j, v in terms:
        coefficients[j] = add(coefficients.get(j, zero), q(v))
    coefficients = {j: mul(a, reciprocal) for j, a in coefficients.items() if a[0] != 0}
    lower = predicted = realized = mul(q(constant), reciprocal)
    errors = {}
    valid_error = True
    for j, a in coefficients.items():
        center = q(base[j])
        for k, v in transform[j].items():
            center = add(center, mul(q(v), q(delta[k])))
        radius = mul(q(eta), q(profile[j]))
        if not le(zero, q(profile[j])):
            raise ValueError('negative profile')
        # Choose each adversarial interval endpoint by the original signed term.
        endpoint = add(center, neg(radius) if a[0] > 0 else radius)
        lower = add(lower, mul(a, endpoint))
        predicted = add(predicted, mul(a, center))
        realized = add(realized, mul(a, q(actual[j])))
        error = add(q(actual[j]), neg(center))
        errors[j] = error
        valid_error &= le(neg(radius), error) and le(error, radius)
    valid_margin = le(q(reserve), lower)
    accepted = bool(valid_error and valid_margin)
    assert result['passed'] == accepted
    assert result['error_pass'] == valid_error and result['margin_pass'] == valid_margin
    assert q(result['lower']) == lower and q(result['predicted']) == predicted
    assert q(result['actual']) == realized and q(result['reserve']) == q(reserve)
    assert {j:q(x) for j,x in result['remainders'].items()} == errors
    if accepted:
        assert le(q(reserve), lower) and le(lower, realized)
    return accepted
