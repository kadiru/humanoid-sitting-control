"""Streaming production component; accepts explicit arrays, never reads a study.

All original rows/prefixes are outside this component and must be preserved by
the future runner. Only NEW actual-hard rows are assembled here. A fixed-sized
state tangent cache is shared; rows, exact encodings and audits stream one at
a time. No solver, mechanics, parameter search or scientific entry point.
"""
from dataclasses import dataclass
from dyadic import D, ZERO, ONE, ratio_float, directed


def canonical(terms):
    out = {}
    for j, a in terms:
        if type(j) is not int or j < 0:
            raise ValueError('nonnegative feature/column index required')
        out[j] = out.get(j, ZERO) + D.of(a)
    return {j:a for j,a in sorted(out.items()) if a}


@dataclass(frozen=True)
class State:
    base: tuple
    tangent: tuple
    profile: tuple


def state_block(x, y, J, basis, x_profile, y_profile, fixed=False):
    if not (len(x)==len(basis)==len(x_profile)==24 and len(y)==len(J)==len(y_profile)==51
            and all(len(j)==24 for j in J)):
        raise ValueError('X24/Y51/J51x24 layout required')
    p = [canonical(row) for row in basis]
    columns = set().union(*(set(row) for row in p))
    if len(columns)>23 or any(len(row)>7 for row in p):
        raise ValueError('physical basis support cap')
    if any(a not in (ONE, -ONE) for row in p for a in row.values()):
        raise ValueError('physical basis must use exact signed units')
    if fixed and columns:
        raise ValueError('fixed state has physical parameters')
    tangent = list(p)
    for output in range(51):
        t = {}
        for k in range(24):
            j = D.of(J[output][k])
            if j:
                for col,a in p[k].items():
                    t[col] = t.get(col,ZERO)+j*a
        tangent.append({c:a for c,a in sorted(t.items()) if a})
    profiles = tuple(D.of(v) for v in tuple(x_profile)+tuple(y_profile))
    if any(v.n<0 for v in profiles):
        raise ValueError('negative profile')
    return State(tuple(D.of(v) for v in tuple(x)+tuple(y)), tuple(tangent),
                 (ZERO,)*75 if fixed else profiles)


def validate_state_locality(states, n_parameters):
    owner = {}
    for k, state in enumerate(states):
        for row in state.tangent[:24]:
            for c in row:
                if not 0 <= c < n_parameters:
                    raise ValueError('physical parameter outside original dimension')
                if c in owner and owner[c] != k:
                    raise ValueError('physical parameter spans states')
                owner[c] = k
    return owner


@dataclass(frozen=True)
class ExactRow:
    row_id: int
    terms: dict
    intercept: D
    tangent: dict
    kappa: D
    scale: D


def compose(row_id, terms, constant, scale, states):
    terms = canonical(terms)
    if len(terms)>10 or len({f//75 for f in terms})>2:
        raise ValueError('hard-row support premise failed')
    scale = D.of(scale)
    if scale.n <= 0:
        raise ValueError('nonpositive scale')
    intercept, kappa, tangent = D.of(constant), ZERO, {}
    for f,a in terms.items():
        state, local = states[f//75], f%75
        intercept += a*state.base[local]
        kappa += abs(a)*state.profile[local]
        for col,t in state.tangent[local].items():
            tangent[col] = tangent.get(col,ZERO)+a*t
    tangent = {c:t for c,t in sorted(tangent.items()) if t}
    if len(tangent)>46:
        raise ValueError('composed row support cap')
    return ExactRow(row_id, terms, intercept, tangent, kappa, scale)


@dataclass(frozen=True)
class BoxWitness:
    column: int
    feature: int
    interval: int
    lower: float
    upper: float


def parameter_boxes(states, varying, H_row, owners, closed_lower, closed_upper,
                    n_parameters):
    """Bind each used parameter to an existing unit feature-trust interval.

    H_row(index) returns stored sparse terms. Bounds are outward binary64, so
    every point in the exact closed interval is contained. The future LP must
    set these *explicit variable bounds*, and its returned values must pass
    exact box checks, as prerequisites for the inward encoding implication.
    No norm bound is inferred; no prior interval arithmetic is replayed.
    """
    needed = validate_state_locality(states,n_parameters)
    index = {}
    for i, owner in enumerate(owners):
        if owner[0]=='feature_trust':
            feature = int(owner[1])
            if feature in index:
                raise ValueError('duplicate interval owner')
            index[feature] = i
    varying = set(map(int,varying))
    boxes, witnesses = {}, {}
    for sid,state in enumerate(states):
        for j,row in enumerate(state.tangent[:24]):
            if len(row)!=1:
                continue
            col,a = next(iter(row.items()))
            if a != ONE or col in boxes:
                continue
            feature = 75*sid+j
            if feature not in varying or feature not in index:
                raise ValueError('missing direct feature interval')
            i = index[feature]
            if canonical(H_row(i)) != {col:ONE}:
                raise ValueError('unit-row witness disagrees with stored H')
            lo, hi = D.of(closed_lower[i]), D.of(closed_upper[i])
            if hi < lo:
                raise ValueError('reversed closed interval')
            low, high = directed(lo), directed(hi, upward=True)
            if not D.of(low)<=lo<=hi<=D.of(high):
                raise ValueError('outward box failed')
            boxes[col] = (D.of(low),D.of(high))
            witnesses[col] = BoxWitness(col,feature,i,low,high)
    if set(boxes)!=set(needed):
        raise ValueError('physical parameter lacks a direct +1 interval witness')
    return boxes,witnesses


@dataclass(frozen=True)
class EncodedRow:
    row_id: int
    columns: tuple
    values: tuple
    rhs: float


def encode(row, boxes, eta_column=None):
    """Common-denominator inward encoding; all compensation stays dyadic."""
    coefficients = {c:-a for c,a in row.tangent.items()}
    if eta_column is not None:
        if eta_column in coefficients:
            raise ValueError('eta aliases a physical parameter')
        if boxes.get(eta_column)!=(ZERO,D(1,-1)):
            raise ValueError('eta box must be [0,1/2]')
        if row.kappa:
            coefficients[eta_column] = row.kappa
    values, compensation = [], ZERO
    columns = tuple(sorted(coefficients))
    for col in columns:
        lo,hi = boxes[col]
        if hi < lo:
            raise ValueError('reversed box')
        a = coefficients[col]
        value = ratio_float(a,row.scale)
        error_numerator = D.of(value)*row.scale-a
        compensation += error_numerator*(lo if error_numerator.n>=0 else hi)
        values.append(value)
    bound = directed(row.intercept+compensation,row.scale)
    return EncodedRow(row.row_id,columns,tuple(values),bound)


def stream(keep,soft,M_row,constants,scales,states,boxes,expected_hard_rows,
           eta_column=None):
    """Iterate original IDs, including constants; exactly one record per hard row."""
    if not len(keep)==len(soft)==len(constants)==len(scales):
        raise ValueError('row array shape mismatch')
    count = 0
    for i in range(len(keep)):
        if bool(keep[i]) and not bool(soft[i]):
            row = compose(i,M_row(i),constants[i],scales[i],states)
            if not row.tangent and row.intercept.n < 0:
                raise ValueError('negative constant actual hard row')
            yield row,encode(row,boxes,eta_column)
            count += 1
    if count != expected_hard_rows:
        raise ValueError('hard-row count mismatch')
