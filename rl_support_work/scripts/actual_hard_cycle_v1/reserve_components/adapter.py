"""Array adapter for the inherited 75-feature layout; no file/model IO.

Study loaders must verify source/input identities BEFORE calling these methods.
STUDY is the only permitted layout for a future scientific wrapper; smaller
Layout instances exist solely to exercise indexing with synthetic fixtures.
"""
from dataclasses import dataclass
from dyadic import D, ZERO
import assemble


@dataclass(frozen=True)
class Layout:
    states: int
    active: int
    parameters: int
    original_rows: int
    hard_rows: int
    varying: int
    intervals: int


STUDY=Layout(1410,1365,78790,331888,156647,80155,92267)


def sparse_row(csr,i):
    left,right=map(int,csr.indptr[i:i+2])
    if left<0 or right<left or right>len(csr.data):
        raise ValueError('invalid CSR pointers')
    return [(int(csr.indices[k]),float(csr.data[k])) for k in range(left,right)]


def make_states(z,fixed,P,layout=STUDY):
    k=layout.states
    for name,shape in [('features',(k,75)),('actualY',(k,51)),('J',(k,51,24)),
                       ('feature_scale',(k,75)),('sigma',(k,51))]:
        if z[name].shape!=shape or str(z[name].dtype)!='float64':
            raise ValueError('frozen float64 array layout mismatch: '+name)
    if P.shape!=(k*75,layout.parameters) or len(fixed)!=k:
        raise ValueError('basis/mesh layout mismatch')
    active=[i for i in range(k) if not bool(fixed[i])]
    if len(active)!=layout.active or list(map(int,z['active']))!=active:
        raise ValueError('active-state mapping mismatch')
    raw,states=[],[]
    for sid in range(k):
        x=list(map(float,z['features'][sid,:24]))
        y=list(map(float,z['actualY'][sid]))  # NEVER take auxiliary features[24:]
        J=[list(map(float,row)) for row in z['J'][sid]]
        p=[sparse_row(P,75*sid+j) for j in range(24)]
        xp=list(map(float,z['feature_scale'][sid,:24]))
        yp=list(map(float,z['sigma'][sid]))
        args=(x,y,J,p,xp,yp,bool(fixed[sid]))
        raw.append(args)
        states.append(assemble.state_block(*args))
    assemble.validate_state_locality(states,layout.parameters)
    return raw,states


def bind_rows(row,M,layout=STUDY):
    if M.shape!=(layout.original_rows,layout.states*75):
        raise ValueError('original signed row matrix shape mismatch')
    for key in ('constant','scale','keep','soft'):
        if row[key].shape!=(layout.original_rows,):
            raise ValueError('original signed row array mismatch')
    if sum(bool(k) and not bool(s) for k,s in zip(row['keep'],row['soft']))!=layout.hard_rows:
        raise ValueError('original hard-row mask mismatch')
    return lambda i:sparse_row(M,i)


def row_inputs(row,M,layout=STUDY):
    """Convert numpy scalar wrappers, retaining every binary64 value exactly."""
    get=bind_rows(row,M,layout)
    for key in ('constant','scale'):
        if str(row[key].dtype)!='float64':
            raise ValueError('signed row constants/scales must be binary64')
    return dict(M_row=get,keep=tuple(map(bool,row['keep'])),soft=tuple(map(bool,row['soft'])),
                constants=tuple(map(float,row['constant'])),scales=tuple(map(float,row['scale'])))


def finite_features(mapped,actual_y,layout=STUDY):
    if (mapped.shape!=(layout.states,75) or actual_y.shape!=(layout.states,51)
            or str(mapped.dtype)!='float64' or str(actual_y.dtype)!='float64'):
        raise ValueError('finite actual feature array shape/precision')
    out=tuple(tuple(map(float,mapped[i,:24]))+tuple(map(float,actual_y[i]))
              for i in range(layout.states))
    # D conversion rejects all infinities/NaNs, without model evaluation.
    for state in out:
        for v in state:D.of(v)
    return out


def interval_values(im,H,layout=STUDY):
    n=layout.intervals
    if H.shape!=(n,layout.parameters) or len(im['owners'])!=n or im['trust']!=1:
        raise ValueError('frozen interval layout/trust mismatch')
    endpoints=[]
    for key in ('closed_lower','closed_upper'):
        if len(im[key])!=n:
            raise ValueError('interval endpoint count')
        endpoints.append([D.pair(*v) for v in im[key]])
    return tuple(endpoints)


def physical_increments(states,owners,H_row,trace):
    """Extract NEW exact H*d physical rows; no multiplication or model calls.

    A future caller must require the trace's source/witness digest to match the
    new admitted B result. Length/owner/basis validation here is additional.
    """
    if len(trace)!=len(owners):
        raise ValueError('new exact H trace length mismatch')
    positions={}
    for i,owner in enumerate(owners):
        if owner[0]=='feature_trust':
            f=int(owner[1])
            if f in positions:
                raise ValueError('duplicate physical trace owner')
            positions[f]=i
    out=[]
    for sid,state in enumerate(states):
        dx=[]
        for j,p in enumerate(state.tangent[:24]):
            if not p:
                dx.append(ZERO)
                continue
            f=sid*75+j
            if f not in positions:
                raise ValueError('missing physical trace owner')
            i=positions[f]
            if assemble.canonical(H_row(i))!=p:
                raise ValueError('H trace row does not match physical basis')
            dx.append(D.pair(*trace[i]))
        out.append(tuple(dx))
    return tuple(out)
