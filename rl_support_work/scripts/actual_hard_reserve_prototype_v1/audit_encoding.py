"""Separate integer-ratio encoding audit; no producer or Fraction import."""
import math
from independent import q, add, mul, neg, le


def audit(coefficients,rhs,boxes,encoded):
    keys={k for k,a in coefficients.items() if a != 0}
    assert set(encoded['coefficients'])==keys==set(encoded['errors'])
    support=(0,1)
    for k in keys:
        a=q(coefficients[k]);value=encoded['coefficients'][k]
        assert isinstance(value,float) and math.isfinite(value)
        # Nearest-even cell checked with exact integer-ratio midpoints.
        predecessor=math.nextafter(value,-math.inf)
        successor=math.nextafter(value,math.inf)
        assert math.isfinite(predecessor) and math.isfinite(successor)
        low=mul(add(q(predecessor),q(value)),(1,2))
        high=mul(add(q(value),q(successor)),(1,2))
        assert le(low,a) and le(a,high)
        if a==low or a==high:
            import struct
            bits=struct.unpack('>Q',struct.pack('>d',value))[0]
            assert bits & 1 == 0
        error=add(q(value),neg(a))
        assert q(encoded['errors'][k])==error
        lo,hi=map(q,boxes[k]);assert le(lo,hi)
        endpoint=lo if error[0]>=0 else hi
        support=add(support,mul(error,endpoint))
    assert q(encoded['minimum_error'])==support
    compensated=add(q(rhs),support)
    assert q(encoded['compensated_rhs'])==compensated
    bound=encoded['rhs'];assert isinstance(bound,float) and math.isfinite(bound)
    assert le(q(bound),compensated)
    nxt=math.nextafter(bound,math.inf)
    assert math.isfinite(nxt) and not le(q(nxt),compensated)
    assert encoded['sufficient_only'] is True and q(encoded['original_rhs'])==q(rhs)
    return True
