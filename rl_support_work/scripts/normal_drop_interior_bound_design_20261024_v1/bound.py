"""Synthetic-only prototype of a conditional interior dual bound.

No study loader, optimizer, primal constructor, or scientific entry point.
Rows must be the actual stored upper reserved-drop rows in any future use.
The output is a dual bound, not a primal witness or admission certificate.
"""
from fractions import Fraction as F
from math import isfinite


def exact_float(value):
    if type(value) is not float or not isfinite(value):
        raise ValueError("Require a finite stored binary64 coefficient")
    return F.from_float(value)


def upper_drop_bound(rows, rhs, independent_columns, pivot, eta_column):
    """Check a fixed conserving star and return exact nonnegative multipliers.

    Each nonpivot row i has a_i at its named independent column; the pivot
    has -a_p at every independent column. Every row has eta coefficient 1.
    No coefficient threshold, row search, or rational-width substitution.
    """
    count = len(rows)
    if count < 2 or len(rhs) != count or not 0 <= pivot < count:
        raise ValueError("Invalid fixed row family")
    if set(independent_columns) != set(range(count)) - {pivot}:
        raise ValueError("Incomplete row-to-column association")
    columns = list(independent_columns.values())
    if len(set(columns)) != count - 1 or eta_column in columns:
        raise ValueError("Invalid independent coordinate association")
    reciprocal = []
    for i, row in enumerate(rows):
        expected = set(columns) if i == pivot else {independent_columns[i]}
        if set(row) != expected | {eta_column}:
            raise ValueError("Unexpected or missing stored coefficient")
        if exact_float(row[eta_column]) != 1:
            raise ValueError("Unexpected eta coefficient")
        direction = {j: exact_float(row[j]) for j in expected}
        values = set(direction.values())
        if len(values) != 1:
            raise ValueError("Pivot coefficients are not identical")
        coefficient = next(iter(values))
        magnitude = -coefficient if i == pivot else coefficient
        if magnitude <= 0:
            raise ValueError("Unexpected incidence sign")
        reciprocal.append(1 / magnitude)
    normalizer = sum(reciprocal, F(0))
    weights = tuple(q / normalizer for q in reciprocal)
    bound = sum((q * exact_float(b) for q, b in zip(weights, rhs)), F(0))
    return weights, bound
