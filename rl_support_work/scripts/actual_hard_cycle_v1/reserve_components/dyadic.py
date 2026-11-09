"""Exact finite-binary arithmetic for new actual-hard rows only. No study IO."""
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class D:
    n: int
    e: int = 0

    def __post_init__(self):
        if type(self.n) is not int or type(self.e) is not int:
            raise TypeError('integer mantissa/exponent required')
        n, e = self.n, self.e
        if n:
            shift = (abs(n) & -abs(n)).bit_length() - 1
            n >>= shift
            e += shift
        else:
            e = 0
        if n.bit_length() > 32768 or abs(e) > 16384:
            raise OverflowError('exact arithmetic resource cap')
        object.__setattr__(self, 'n', n)
        object.__setattr__(self, 'e', e)

    @classmethod
    def of(cls, x):
        if isinstance(x, cls):
            return x
        if type(x) is int:
            return cls(x)
        if type(x) is not float or not math.isfinite(x):
            raise ValueError('finite binary64 or integer required')
        n, q = x.as_integer_ratio()
        return cls(n, -(q.bit_length()-1))

    @classmethod
    def pair(cls, n, q):
        if type(n) is not int or type(q) is not int or q <= 0 or q & (q-1):
            raise ValueError('dyadic ratio required')
        return cls(n, -(q.bit_length()-1))

    def ratio(self):
        return (self.n << self.e, 1) if self.e >= 0 else (self.n, 1 << -self.e)

    def __add__(self, other):
        other = D.of(other)
        e = min(self.e, other.e)
        return D((self.n << (self.e-e)) + (other.n << (other.e-e)), e)

    __radd__ = __add__

    def __neg__(self):
        return D(-self.n, self.e)

    def __sub__(self, other):
        return self + -D.of(other)

    def __mul__(self, other):
        other = D.of(other)
        return D(self.n*other.n, self.e+other.e)

    __rmul__ = __mul__

    def __abs__(self):
        return D(abs(self.n), self.e)

    def __le__(self, other):
        return (self-D.of(other)).n <= 0

    def __lt__(self, other):
        return (self-D.of(other)).n < 0

    def __bool__(self):
        return self.n != 0


ZERO = D(0)
ONE = D(1)


def ratio_float(a, positive_scale):
    """Nearest binary64 of a/scale, with overflow rejected; no intermediate float."""
    a, s = D.of(a), D.of(positive_scale)
    if s.n <= 0:
        raise ValueError('positive scale required')
    shift = a.e-s.e
    n, q = a.n, s.n
    if shift >= 0:
        n <<= shift
    else:
        q <<= -shift
    value = n/q
    if not math.isfinite(value):
        raise OverflowError('nonfinite binary64 encoding')
    return value


def directed(a, positive_scale=ONE, upward=False):
    value = ratio_float(a, positive_scale)
    difference = D.of(value)*positive_scale-D.of(a)
    if (upward and difference.n < 0) or (not upward and difference.n > 0):
        value = math.nextafter(value, math.inf if upward else -math.inf)
    if not math.isfinite(value):
        raise OverflowError('nonfinite directed bound')
    return value
