"""Exact posterior feature/row certificate from a NEW interior H trace.

The caller must bind the trace to the admitted NEW interior witness and the
freshly evaluated finite map. This component does not perform model evaluation
or authorize one. It reuses exact physical increments in that trace, avoiding
a repeated full P*d or H*d multiplication.
"""
from dyadic import D, ZERO
from assemble import canonical


def predictions(states,jacobians,physical_delta):
    if not len(states)==len(jacobians)==len(physical_delta):
        raise ValueError('state count mismatch')
    for state,J,dx in zip(states,jacobians,physical_delta):
        if len(dx)!=24 or len(J)!=51 or any(len(row)!=24 for row in J):
            raise ValueError('physical trace shape')
        dx=tuple(D.of(v) for v in dx)
        for k,t in enumerate(state.tangent[:24]):
            if not t and dx[k]:
                raise ValueError('immutable physical feature changed')
        x=tuple(state.base[k]+dx[k] for k in range(24))
        y=tuple(state.base[24+j]+sum((D.of(J[j][k])*dx[k] for k in range(24)),ZERO)
                for j in range(51))
        yield x+y


def feature_errors(used,states,predicted,finite_actual,eta):
    eta=D.of(eta)
    if not ZERO<=eta<=D(1,-1):
        raise ValueError('eta outside [0,1/2]')
    errors={}
    for f in sorted(used):
        sid,j=divmod(f,75)
        error=D.of(finite_actual[sid][j])-predicted[sid][j]
        if abs(error)>states[sid].profile[j]*eta:
            raise ValueError('finite nonlinear feature outside reserved error interval')
        errors[f]=error
    return errors


def rows(keep,soft,M_row,constants,states,predicted,finite_actual,eta,expected_count):
    """All exact signed rows must pass; no geometric tolerance substituted."""
    eta=D.of(eta)
    if not ZERO<=eta<=D(1,-1):
        raise ValueError('eta outside [0,1/2]')
    count=0
    for i in range(len(keep)):
        if not keep[i] or soft[i]:
            continue
        pred,real,kappa=D.of(constants[i]),D.of(constants[i]),ZERO
        for f,a in canonical(M_row(i)).items():
            sid,j=divmod(f,75)
            actual=D.of(finite_actual[sid][j])
            error=actual-predicted[sid][j]
            if abs(error)>eta*states[sid].profile[j]:
                raise ValueError('row feature outside posterior bound')
            pred+=a*predicted[sid][j]
            real+=a*actual
            kappa+=abs(a)*states[sid].profile[j]
        lower=pred-eta*kappa
        if lower.n<0 or real<lower:
            raise ValueError('actual-hard reserve certificate rejected')
        yield i,lower,real
        count+=1
    if count!=expected_count:
        raise ValueError('return hard-row coverage mismatch')
