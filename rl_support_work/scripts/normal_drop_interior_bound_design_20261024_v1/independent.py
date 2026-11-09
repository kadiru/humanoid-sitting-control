"""Independent exact row-combination checker; imports no producer.

This validates only a supplied bound for the supplied rows. It does not
associate these rows to a scientific input or check full primal feasibility.
"""
from fractions import Fraction as F
from math import isfinite


def check(rows, rhs, weights, bound, eta_column):
    if len(rows) != len(rhs) or len(rows) != len(weights):
        return False
    if any(type(q) is not F or q < 0 for q in weights) or type(bound) is not F:
        return False
    coefficients = {}
    value = F(0)
    for row, b, weight in zip(rows, rhs, weights):
        if type(b) is not float or not isfinite(b):
            return False
        value += weight * F(*b.as_integer_ratio())
        for j, a in row.items():
            if type(a) is not float or not isfinite(a):
                return False
            coefficients[j] = coefficients.get(j, F(0)) + weight * F(*a.as_integer_ratio())
    return (value == bound and coefficients.get(eta_column) == 1
            and all(q == 0 for j, q in coefficients.items() if j != eta_column))
