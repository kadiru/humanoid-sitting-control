"""Independent Fraction/reference audit. Imports no producer arithmetic/code.

This is an independent algorithm, not an independent person or coding agent.
Its cost is additional to assembly. A future wrapper must bind all inputs to
the frozen study identities and account for this entire pass in the deadline.
"""
from fractions import Fraction as F
import math
import struct


def q(v):
    if type(v) is float:
        if not math.isfinite(v):
            raise ValueError('nonfinite data')
        return F.from_float(v)
    if type(v) is int or isinstance(v,F):
        return F(v)
    # Producer numbers are read as data, not used to perform arithmetic.
    if type(v.n) is not int or type(v.e) is not int:
        raise ValueError('invalid dyadic record')
    return F(v.n*(1 << v.e)) if v.e>=0 else F(v.n,1 << -v.e)


def collect(terms):
    out={}
    for j,a in terms:
        if type(j) is not int or j<0:
            raise ValueError('invalid sparse index')
        out[j]=out.get(j,F(0))+q(a)
    return {j:a for j,a in out.items() if a}


def state(raw,actual):
    x,y,J,p,v,s,fixed=raw
    if not (len(x)==len(p)==len(v)==24 and len(y)==len(J)==len(s)==51
            and all(len(row)==24 for row in J)):
        raise ValueError('reference block shape')
    pp=[collect(row) for row in p]
    if len(set().union(*(set(r) for r in pp)))>23 or any(len(r)>7 for r in pp):
        raise ValueError('reference support premise')
    if any(a not in (-1,1) for row in pp for a in row.values()):
        raise ValueError('reference basis unit premise')
    if fixed and any(pp):
        raise ValueError('fixed block has parameters')
    base=tuple(map(q,tuple(x)+tuple(y)))
    profiles=tuple(map(q,tuple(v)+tuple(s)))
    if any(a<0 for a in profiles):
        raise ValueError('negative profile')
    if fixed:
        profiles=(F(0),)*75
    tt=pp[:]
    for output in range(51):
        terms=[]
        for k,row in enumerate(pp):
            terms.extend((c,q(J[output][k])*a) for c,a in row.items())
        tt.append(collect(terms))
    if tuple(map(q,actual.base))!=base or tuple(map(q,actual.profile))!=profiles:
        raise ValueError('block base/profile mismatch')
    if [dict((c,q(a)) for c,a in r.items()) for r in actual.tangent]!=tt:
        raise ValueError('block tangent mismatch')
    return base,tuple(tt),profiles


def boxes(reference_states,varying,H_row,owners,closed_lo,closed_hi,
          records,n_parameters):
    required={}
    for sid,(_,tt,_) in enumerate(reference_states):
        for row in tt[:24]:
            for c in row:
                if not 0<=c<n_parameters or c in required and required[c]!=sid:
                    raise ValueError('invalid global physical parameter')
                required[c]=sid
    if set(records)!=set(required):
        raise ValueError('box witness coverage')
    found={}
    for c,w in records.items():
        if w.column!=c or w.feature not in varying:
            raise ValueError('box index mismatch')
        sid,j=divmod(w.feature,75)
        if j>=24 or reference_states[sid][1][j]!={c:F(1)}:
            raise ValueError('box is not direct physical unit')
        if owners[w.interval]!=['feature_trust',w.feature]:
            raise ValueError('box interval owner mismatch')
        if collect(H_row(w.interval))!={c:F(1)}:
            raise ValueError('box stored H mismatch')
        lo,hi=q(closed_lo[w.interval]),q(closed_hi[w.interval])
        low,high=q(w.lower),q(w.upper)
        if not low<=lo<=hi<=high:
            raise ValueError('box containment')
        # Require the tight outward floats, not arbitrary new boxes.
        if q(math.nextafter(w.lower,math.inf))<=lo:
            raise ValueError('lower box is not tight outward rounding')
        if q(math.nextafter(w.upper,-math.inf))>=hi:
            raise ValueError('upper box is not tight outward rounding')
        found[c]=(low,high)
    return found


def nearest(value,exact):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('nonfinite encoded coefficient')
    low=math.nextafter(value,-math.inf)
    high=math.nextafter(value,math.inf)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError('unbounded nearest cell rejected')
    left,right=(q(low)+q(value))/2,(q(high)+q(value))/2
    if not left<=exact<=right:
        raise ValueError('wrong nearest coefficient')
    if exact in (left,right) and struct.unpack('>Q',struct.pack('>d',value))[0]&1:
        raise ValueError('wrong nearest tie parity')


def row(i,terms,constant,scale,reference_states,box,exact,encoded,eta_column=None):
    coeff=collect(terms)
    if len(coeff)>10 or len({f//75 for f in coeff})>2 or q(scale)<=0:
        raise ValueError('row premise')
    intercept,kappa,tt=q(constant),F(0),{}
    for f,a in coeff.items():
        base,transform,profile=reference_states[f//75]
        local=f%75
        intercept+=a*base[local]
        kappa+=abs(a)*profile[local]
        for c,t in transform[local].items():
            tt[c]=tt.get(c,F(0))+a*t
    tt={c:a for c,a in tt.items() if a}
    if (exact.row_id!=i or encoded.row_id!=i or q(exact.scale)!=q(scale)
            or q(exact.intercept)!=intercept or q(exact.kappa)!=kappa
            or {c:q(a) for c,a in exact.tangent.items()}!=tt
            or {f:q(a) for f,a in exact.terms.items()}!=coeff):
        raise ValueError('exact composed row mismatch')
    aa={c:-a/q(scale) for c,a in tt.items()}
    if eta_column is not None:
        if eta_column in aa or box[eta_column]!=(F(0),F(1,2)):
            raise ValueError('eta alias/box')
        if kappa:
            aa[eta_column]=kappa/q(scale)
    if encoded.columns!=tuple(sorted(aa)) or len(encoded.values)!=len(aa):
        raise ValueError('encoded support mismatch')
    compensated=intercept/q(scale)
    for c,v in zip(encoded.columns,encoded.values):
        nearest(v,aa[c])
        error=q(v)-aa[c]
        lo,hi=box[c]
        if hi<lo:
            raise ValueError('reversed box')
        compensated+=min(error*lo,error*hi)
    bound=q(encoded.rhs)
    successor=math.nextafter(encoded.rhs,math.inf)
    if not bound<=compensated or not math.isfinite(successor) or q(successor)<=compensated:
        raise ValueError('wrong downward compensated bound')
    return True


def stream(records,keep,soft,M_row,constants,scales,reference_states,box,
           expected_count,eta_column=None):
    """Consumer must exhaust the iterator before committing any output."""
    expected=(i for i in range(len(keep)) if keep[i] and not soft[i])
    records=iter(records)
    count=0
    for i in expected:
        try:
            exact,encoded=next(records)
        except StopIteration as exc:
            raise ValueError('missing hard row') from exc
        row(i,M_row(i),constants[i],scales[i],reference_states,box,
            exact,encoded,eta_column)
        count+=1
        yield encoded
    if next(records,None) is not None or count!=expected_count:
        raise ValueError('extra/missing hard rows')


def returned(records,raw_states,owners,H_row,trace,keep,soft,M_row,constants,
             finite_actual,eta,expected_count):
    """Independent return audit; builds predictions directly from original J.

    No producer tangent or predictor is used. The NEW witness/trace provenance
    must be checked by the future wrapper before this arithmetic is authorized.
    """
    if len(trace)!=len(owners) or not F(0)<=q(eta)<=F(1,2):
        raise ValueError('return trace shape/eta')
    positions={}
    for i,owner in enumerate(owners):
        if owner[0]=='feature_trust':
            f=int(owner[1])
            if f in positions:
                raise ValueError('duplicate trace owner')
            positions[f]=i
    pred=[]
    profiles=[]
    for sid,(x,y,J,p,v,s,fixed) in enumerate(raw_states):
        dx=[]
        for j,terms in enumerate(p):
            terms=collect(terms)
            if fixed and terms:
                raise ValueError('fixed state has physical parameters')
            if not terms:
                dx.append(F(0))
                continue
            index=positions[sid*75+j]
            if collect(H_row(index))!=terms:
                raise ValueError('wrong physical trace row')
            n,d=trace[index]
            if type(n) is not int or type(d) is not int or d<=0 or d&(d-1):
                raise ValueError('nondyadic trace')
            dx.append(F(n,d))
        pred.append([q(x[j])+dx[j] for j in range(24)]+
                    [q(y[j])+sum((q(J[j][k])*dx[k] for k in range(24)),F(0)) for j in range(51)])
        profiles.append([F(0)]*75 if fixed else list(map(q,tuple(v)+tuple(s))))
    records=iter(records)
    count=0
    for i in range(len(keep)):
        if not keep[i] or soft[i]:
            continue
        try:
            result=next(records)
        except StopIteration as exc:
            raise ValueError('missing returned row') from exc
        lower,real=q(constants[i]),q(constants[i])
        for f,a in collect(M_row(i)).items():
            sid,j=divmod(f,75)
            value=q(finite_actual[sid][j])
            if profiles[sid][j]<0 or abs(value-pred[sid][j])>q(eta)*profiles[sid][j]:
                raise ValueError('independent actual error exceeds reserve')
            # Endpoint form, independent of producer's kappa sum.
            worst=pred[sid][j]-q(eta)*profiles[sid][j] if a>=0 else pred[sid][j]+q(eta)*profiles[sid][j]
            lower+=a*worst
            real+=a*value
        if lower<0 or real<lower or result[0]!=i or q(result[1])!=lower or q(result[2])!=real:
            raise ValueError('independent returned row failure/mismatch')
        count+=1
    if next(records,None) is not None or count!=expected_count:
        raise ValueError('returned row coverage')
    return True


def point(keep,soft,M_row,constants,reference_states,box,coordinates,eta_column,
          saved,expected_count):
    """NEW exact point check, recomposed by direct feature predictions."""
    values=list(map(q,coordinates))
    for c,(lo,hi) in box.items():
        if not lo<=values[c]<=hi:
            raise ValueError('independent point outside finite box')
    eta=F(0) if eta_column is None else values[eta_column]
    if not 0<=eta<=F(1,2):
        raise ValueError('independent eta outside box')
    predicted=[]
    for base,tangent,profile in reference_states:
        predicted.append([v+sum((t*values[c] for c,t in row.items()),F(0))
                          for v,row in zip(base,tangent)])
    count=0;minimum=None
    for i in range(len(keep)):
        if not keep[i] or soft[i]:continue
        lower=q(constants[i])
        for f,c in collect(M_row(i)).items():
            sid,j=divmod(f,75)
            s=reference_states[sid][2][j]
            lower+=c*predicted[sid][j]-abs(c)*eta*s
        if lower<0:
            raise ValueError('independent exact tangent reserve negative')
        minimum=lower if minimum is None or lower<minimum else minimum
        count+=1
    if (count!=expected_count or saved['rows']!=count or q(saved['minimum_raw_margin'])!=minimum
            or q(saved['eta'])!=eta or saved['point_modified'] or not saved['passed']):
        raise ValueError('independent exact point result mismatch')
    return True
