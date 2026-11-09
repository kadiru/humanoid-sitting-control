"""Exact, data-free prototype for an actual-feature error-reserve certificate.

This module has no file, model, solver, SSH or scientific-launch entry point.
It is a reference specification, not a production study implementation.
The supplied tangent need not have a globally valid Hessian: the remainder is
measured on the eventual finite-encoded, freshly evaluated feature vector.
"""
from fractions import Fraction as Q
import math


def rational(value):
    if isinstance(value, bool):
        raise TypeError('Boolean is not a scientific coefficient')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite input')
        return Q.from_float(value)
    if isinstance(value, (int, Q)):
        return Q(value)
    raise TypeError('Expected integer, Fraction, or finite binary64')


def canonical(terms):
    """Combine aliases before taking absolute values, without zero tolerance."""
    out = {}
    for feature, coefficient in terms:
        if not isinstance(feature, int) or isinstance(feature, bool) or feature < 0:
            raise ValueError('Invalid feature identifier')
        out[feature] = out.get(feature, Q(0)) + rational(coefficient)
    return {j: a for j, a in sorted(out.items()) if a != 0}


def compose(terms, constant, scale, base_actual, transform, profile):
    """r(z)=(constant+sum a_j*z_j)/scale, zhat=base_actual+T*delta.

    transform maps actual feature IDs to sparse physical-coordinate tangents;
    X features have identity rows and Y features use the accepted saved J.
    profile is fixed independently of the returned direction, in feature units.
    Zero-profile features must have exactly zero observed remainder at return.
    """
    scale = rational(scale)
    if scale <= 0:
        raise ValueError('Nonpositive row scale')
    coefficients = {j: a / scale for j, a in canonical(terms).items()}
    offset = rational(constant) / scale
    intercept = offset
    tangent = {}
    sensitivity = Q(0)
    for j, a in coefficients.items():
        width = rational(profile[j])
        if width < 0:
            raise ValueError('Negative error profile')
        intercept += a * rational(base_actual[j])
        sensitivity += abs(a) * width
        for k, t in canonical(transform[j].items()).items():
            tangent[k] = tangent.get(k, Q(0)) + a * t
    return dict(coefficients=coefficients, offset=offset, intercept=intercept,
                tangent={k: t for k, t in sorted(tangent.items()) if t != 0},
                sensitivity=sensitivity)


def certify(row, base_actual, transform, delta, actual, profile, eta, reserve=0):
    """Conditional exact return certificate; never a pre-evaluation guarantee.

    Error includes all nonlinear, min-branch, finite-encoding and evaluation
    differences relative to the stored exact affine predictor. It is not a
    derivative error estimate or a proof about an entire trust box.
    """
    eta, reserve = rational(eta), rational(reserve)
    if eta < 0 or reserve < 0:
        raise ValueError('Negative margin/reserve')
    expected_intercept = row['offset'] + sum(
        (a * rational(base_actual[j]) for j, a in row['coefficients'].items()), Q(0))
    if expected_intercept != row['intercept']:
        raise ValueError('Wrong tangent intercept')
    predicted = row['intercept'] + sum(
        (a * rational(delta[k]) for k, a in row['tangent'].items()), Q(0))
    sensitivity = sum((abs(a) * rational(profile[j])
                       for j, a in row['coefficients'].items()), Q(0))
    if sensitivity != row['sensitivity']:
        raise ValueError('Wrong absolute-value reserve weight')
    remainder = {}
    feature_prediction = {}
    for j in row['coefficients']:
        if rational(profile[j]) < 0:
            raise ValueError('Negative error profile')
        prediction = rational(base_actual[j]) + sum(
            (rational(t) * rational(delta[k]) for k, t in transform[j].items()), Q(0))
        feature_prediction[j] = prediction
        remainder[j] = rational(actual[j]) - prediction
    predicted_from_features = row['offset'] + sum(
        (a * feature_prediction[j] for j, a in row['coefficients'].items()), Q(0))
    if predicted_from_features != predicted:
        raise ValueError('Wrong tangent composition')
    error_pass = all(abs(e) <= eta * rational(profile[j]) for j, e in remainder.items())
    lower = predicted - eta * sensitivity
    actual_row = row['offset'] + sum(
        (a * rational(actual[j]) for j, a in row['coefficients'].items()), Q(0))
    margin_pass = lower >= reserve
    passed = bool(error_pass and margin_pass)
    if passed:
        assert actual_row >= lower >= reserve
    return dict(passed=passed, error_pass=error_pass, margin_pass=margin_pass,
                predicted=predicted, sensitivity=sensitivity, lower=lower,
                actual=actual_row, reserve=reserve, remainders=remainder)


def require_coverage(expected_ids, results):
    """Every expected row appears exactly once, including constant rows."""
    expected_ids = list(expected_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Duplicate expected row')
    seen = [i for i, _ in results]
    if len(set(seen)) != len(seen) or set(seen) != set(expected_ids):
        raise ValueError('Missing, duplicate or unexpected row certificate')
    return all(r['passed'] for _, r in results)
