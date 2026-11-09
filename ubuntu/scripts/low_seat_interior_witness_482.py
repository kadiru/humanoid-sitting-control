"""NEW whole-path relative-interior witness formulation, preparation only.

An ordinary epigraph predictor is never decoded. A second LP maximizes the
minimum RELATIVE clearance of its coupled direction from every finite control
and drop face, while retaining at least half the predictor's modeled soft
improvement and ALL hard rows. Exact arithmetic certifies positive clearance
before any future decoding. It never edits, clips or projects an LP witness.

This is not a tolerance/guard change or continuation of478. The existing
nonlinear acceptance gates, original domain and buffers stay unchanged.
"""
from fractions import Fraction as F
import numpy as np
from scipy import sparse
from low_seat_free_gap_solver_470 import load_saved,read,sha,D,fq,pair,unpack
from low_seat_exact_gap_chart_468 import U

SPEC = dict(
    scope=__doc__, seed='Sealed472 INITIAL only; no478 accepted/rejected state or witness.',
    predictor='Unchanged470 rate/window epigraph with all existing hard rows.',
    witness='Maximize dimensionless minimum relative interior clearance, subject to half the primary modeled improvement.',
    exact_certificate='Recompute every unscaled interval dot product as Fractions. The certified margin is the actual minimum, not the LP-reported eta.',
    finite_encoding='Unchanged476 shared ray must have alpha EXACT1. Target endpoints are one representable value inward; each drop reserves its existing468 encoding-error bound.',
    rejection='Nonpositive exact clearance, empty shrunken interval or failed LP/root/nonlinear gate rejects without witness edits or fallback.',
    zero_tangent_equalities='All1366 analytic zero tangent rows retained homogeneously; original1411 actual normal/hover equalities still checked.',
    prospective_budget=dict(max_predictors=24,max_corrections_per_predictor=4,max_actual_lps=144,max_maps=120,lp_time_limit_s=45),
    budget_rationale='24 primary plus24 interior witness LPs and at most96 interior correction LPs; at most120 mapped paths preserves the existing120-load-rounding contract. No old run is extended.',
    numerical_limits='HiGHS1e-9, independent linear1e-7, original2^-20 hard/soft step guards, trust1..1/32 and quarter correction domain unchanged.',
    nonlinear_gates='All original hard rows, actual f/p monotonicity, full11906 boxes, normal<=1e-13, hover<=1e-8, fixed/inactive exactness, positive soft improvement and ratio>=.1.',
    physical_admission=False, executable_run_wrapper=False)

def load_reference():
    """Read ONLY saved472 initial matrices; no mechanics/AD/root/LP call."""
    d=load_saved();P=D/'free_gap_path_472/linear_00'
    a=dict(np.load(P/'input.npz'));meta=read(P/'metadata.json')
    a.update(A=sparse.load_npz(P/'A.npz').tocsr(),C=sparse.load_npz(P/'C.npz').tocsr(),
             drops=list(map(unpack,meta['drops'])),mode='predictor',trust=meta['trust'])
    seed=dict(np.load(D/'free_gap_path_472/initial.npz'))
    assert np.array_equal(seed['X'],a['X']) and np.array_equal(seed['Y'],a['Y'])
    assert np.array_equal(seed['J'],a['J']) and np.array_equal(seed['target'],a['target'])
    return d,a

def interval_system(d,a):
    """Exact finite intervals, plus an explicit unmodified closed-domain copy.

    Full f restrictions are LINEAR MODEL restrictions. Actual dependent f must
    still pass its unchanged nonlinear checks after the15-equation lift.
    """
    radius=.02*a['trust'];lo=np.maximum(d['lower'],a['origin']-radius)
    hi=np.minimum(d['upper'],a['origin']+radius)
    if a['mode']=='correction':
        lo=np.maximum(lo,a['z']-.25*radius);hi=np.minimum(hi,a['z']+.25*radius)
    pos={int(v):i for i,v in enumerate(d['free'])}
    ids=[int(v) for v in d['free'] if int(v)%10!=0]
    Z=d['B'][ids].tocsr();zl=[];zu=[];ol=[];ou=[];owners=[]
    for v in ids:
        b,s=divmod(v,10);i=pos[v];c=fq(a['target'][b,s]);l=fq(lo[i]);u=fq(hi[i])
        ol.append(l-c);ou.append(u-c)
        # Rounded-once p/load coordinates cannot return to the original face.
        zl.append(fq(np.nextafter(lo[i],np.inf))-c)
        zu.append(fq(np.nextafter(hi[i],-np.inf))-c)
        owners.append(['target',b,s])
    full_l=list(map(fq,a['full_step_lower']));full_u=list(map(fq,a['full_step_upper']))
    dom=d['gap_domain'];errors=[U]*206;errors[dom['pivot']]=205*U
    nl=[l-c for l,c in zip(dom['lower'],a['drops'])]
    nu=[u-c for u,c in zip(dom['upper'],a['drops'])]
    el=[l+e for l,e in zip(nl,errors)];eu=[u-e for u,e in zip(nu,errors)]
    H=sparse.vstack([a['C'],Z,d['N']],format='csr')
    lower=full_l+zl+el;upper=full_u+zu+eu
    closed_lower=full_l+ol+nl;closed_upper=full_u+ou+nu
    owners=[['full_linear_control',i] for i in range(len(full_l))]+owners+[['drop',i] for i in range(206)]
    widths=[u-l for l,u in zip(lower,upper)]
    if any(w<=0 for w in widths):
        raise ValueError('No positive representable interior in a requested finite step interval')
    return dict(H=H,lower=lower,upper=upper,widths=widths,closed_lower=closed_lower,
                closed_upper=closed_upper,owners=owners,target_rows=len(ids),drop_encoding_errors=errors)

def assemble_witness(d,a,soft_ceiling):
    """Pure saved-matrix algebra. No LP solve or complete path evaluation."""
    s=interval_system(d,a);H=s['H'];n=H.shape[1]
    keep=d['a']['keep'];soft=d['a']['soft']&keep
    top=sparse.hstack([-a['A'][keep],sparse.csr_matrix((int(keep.sum()),1))],format='csr')
    rhs=a['r'][keep]-a['guard'][keep]+soft[keep]*soft_ceiling
    closed=sparse.hstack([H,sparse.csr_matrix((H.shape[0],1))],format='csr')
    # Relative clearance is dimensionless. Exact checking below uses ORIGINAL
    # unscaled Fraction intervals, so floating normalization cannot admit a ray.
    widths=np.array([float(x) for x in s['widths']]);HN=H.multiply((1/widths)[:,None]).tocsr()
    ones=sparse.csr_matrix(np.ones((H.shape[0],1)))
    mat=sparse.vstack([top,closed,-closed,sparse.hstack([HN,ones]),sparse.hstack([-HN,ones])],format='csr')
    rhs=np.r_[rhs,[float(x) for x in s['closed_upper']],[-float(x) for x in s['closed_lower']],
              [float(u/w) for u,w in zip(s['upper'],s['widths'])],[-float(l/w) for l,w in zip(s['lower'],s['widths'])]]
    return dict(system=s,matrix=mat,rhs=rhs,cost=np.r_[np.zeros(n),-1.],
                bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),.5]],
                equality=sparse.csr_matrix((int(d['ea']['optimizer_keep'].sum()),n+1)),
                equality_rhs=np.zeros(int(d['ea']['optimizer_keep'].sum())),soft_ceiling=float(soft_ceiling))

def certify(s,t):
    """Certify a witness UNCHANGED; never manufacture a feasible direction.

    Returned eta is an exact a-posteriori lower bound. It is allowed to differ
    from the floating LP objective, but must be strictly positive. Replacing
    the claimed margin does not replace any control or physical criterion.
    """
    tf=list(map(fq,t));H=s['H'].tocsr();eta=F(1,2);owner=None;negative=[]
    for i,(l,u,w) in enumerate(zip(s['lower'],s['upper'],s['widths'])):
        b,e=H.indptr[i:i+2]
        v=sum((fq(c)*tf[j] for c,j in zip(H.data[b:e],H.indices[b:e])),F(0))
        q=min(v-l,u-v)/w
        if q<eta:eta=q;owner=s['owners'][i]
        if q<=0:negative.append(i)
    return dict(passed=eta>0,certified_relative_clearance=pair(eta),limiting_owner=owner,
                nonpositive_intervals=negative,witness_modified=False,actual_dependent_f_guaranteed=False)

def save(w,folder):
    folder.mkdir(exist_ok=False)
    for k in ['matrix','equality']:sparse.save_npz(folder/(k+'.npz'),w[k])
    sparse.save_npz(folder/'interval_matrix.npz',w['system']['H'])
    np.savez_compressed(folder/'arrays.npz',**{k:w[k] for k in ['rhs','cost','bounds','equality_rhs']})
    s=w['system'];meta={k:[pair(x) for x in s[k]] for k in ['lower','upper','widths','closed_lower','closed_upper','drop_encoding_errors']}
    meta.update(owners=s['owners'],target_rows=s['target_rows'],soft_ceiling=w['soft_ceiling'])
    (folder/'intervals.json').write_text(__import__('json').dumps(meta))
