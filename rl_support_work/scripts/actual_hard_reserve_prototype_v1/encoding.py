"""Reference inward binary64 encoding for NEW actual-row constraints only.

Inputs must be an exact rational row A*x<=b and proven finite parameter boxes.
No prior study encoding/proof is read or repeated by this pure component.
The output is sufficient, not equivalent; it may exclude every feasible point.
"""
from fractions import Fraction as Q
import math


def encode(coefficients, rhs, boxes):
    coefficients = {k:Q(v) for k,v in coefficients.items() if v != 0}
    rhs = Q(rhs)
    encoded, errors = {}, {}
    minimum_error = Q(0)
    for k, a in sorted(coefficients.items()):
        lo, hi = map(Q, boxes[k])
        if lo > hi:
            raise ValueError('Reversed box')
        value = float(a)
        if not math.isfinite(value):
            raise ValueError('Nonfinite coefficient encoding')
        encoded[k] = value
        error = Q.from_float(value) - a
        errors[k] = error
        minimum_error += min(error*lo,error*hi)
    compensated = rhs + minimum_error
    bound = float(compensated)
    if not math.isfinite(bound):
        raise ValueError('Nonfinite RHS encoding')
    if Q.from_float(bound) > compensated:
        bound = math.nextafter(bound,-math.inf)
    if not math.isfinite(bound):
        raise ValueError('Nonfinite inward RHS')
    assert Q.from_float(bound) <= compensated
    return dict(coefficients=encoded, rhs=bound, errors=errors,
                minimum_error=minimum_error, compensated_rhs=compensated,
                original_rhs=rhs, sufficient_only=True)
