"""Invented fixtures only; never reads scientific inputs or saved points."""
from copy import deepcopy
from fractions import Fraction as F
import json
import math
from bound import upper_drop_bound
from independent import check


def feasible(rows, rhs, point):
    return all(sum((F.from_float(a) * point[j] for j, a in row.items()), F(0))
               <= F.from_float(b) for row, b in zip(rows, rhs))


def main():
    passed = []
    rows = [{0: 2., 1: 1.}, {0: -4., 1: 1.}]
    rhs = [.25, .5]
    weights, bound = upper_drop_bound(rows, rhs, {0: 0}, 1, 1)
    assert weights == (F(2, 3), F(1, 3)) and bound == F(1, 3)
    assert check(rows, rhs, weights, bound, 1)
    assert feasible(rows, rhs, {0: -F(1, 24), 1: bound})
    passed.append("unequal stored row scales cancel exactly")

    three = [{2: 3., 9: 1.}, {5: 5., 9: 1.}, {2: -7., 5: -7., 9: 1.}]
    values = [.125, .25, .375]
    w3, b3 = upper_drop_bound(three, values, {0: 2, 1: 5}, 2, 9)
    assert check(three, values, w3, b3, 9)
    passed.append("three rows with nonbinary reciprocal weights")

    tampered = deepcopy(three)
    tampered[2][5] = math.nextafter(-7., -math.inf)
    try:
        upper_drop_bound(tampered, values, {0: 2, 1: 5}, 2, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal pivot coefficients were accepted")
    assert not check(tampered, values, w3, b3, 9)
    passed.append("one-bit pivot change rejected independently")

    assert not check(rows, rhs, (F(1, 2), F(1, 2)), bound, 1)
    passed.append("incorrect dual normalization rejected")

    guarded = rows + [{0: -1.}]
    guarded_rhs = rhs + [0.]
    assert not feasible(guarded, guarded_rhs, {0: -F(1, 24), 1: bound})
    assert feasible(guarded, guarded_rhs, {0: F(0), 1: F(1, 4)})
    # z >= 0 and 2*z+eta <= 1/4 imply eta <= 1/4 < bound.
    assert F(1, 4) < bound
    passed.append("conservation upper bound need not be the full optimum")

    hard = [{0: -1.}]
    assert feasible(hard, [-1.], {0: F(1)})
    assert not feasible(hard, [-1.], {0: F(1, 2)})
    passed.append("half a feasible primary direction can violate a hard guard")
    print(json.dumps(dict(classification="synthetic arithmetic only",
                         study_inputs_read=0, scientific_launches=0,
                         physical_success=False, passed=passed), indent=2))


if __name__ == "__main__":
    main()
