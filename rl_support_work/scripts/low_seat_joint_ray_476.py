"""NEW shared affine-ray chart for the complete free-gap control vector.

One exact scalar multiplies EVERY free gap, p, persistent-load and normal-drop
direction. It is the largest value in [0,1] satisfying all finite control/drop
bounds and the complete linearized f/p/load step intervals. Unlike472's separate
1/(1+rho) maps, a feasible full ray is not contracted. This is new coordinate
construction, never a modification or selection of an old rejected trial.

The dependent f restriction here is a LINEAR MODEL restriction only. Actual
15-equation roots, phase order, all hard rows and all old boxes remain decisive.
Finite binary64 encoding is quantized; only the unrounded, interior zero-ray
map has the audited B derivative. No finite-map differentiability is claimed.
"""
from fractions import Fraction as F
import numpy as np
from low_seat_free_gap_solver_470 import fq,pair,gaps
from low_seat_exact_gap_chart_468 import joint_round

def pieces(d,a,t):
    """Exact target directions with the SAME persistent load/drop pivots."""
    pos={int(v):i for i,v in enumerate(d['free'])}; groups=[]
    radius=.02*a['trust']
    lo=np.maximum(d['lower'],a['origin']-radius)
    hi=np.minimum(d['upper'],a['origin']+radius)
    if a['mode']=='correction':
        lo=np.maximum(lo,a['z']-.25*radius)
        hi=np.minimum(hi,a['z']+.25*radius)
    for g in d['map_groups']:
        b=g['state'];kind=g['kind'];v=[fq(t[j]) for j in g['columns']]
        if kind=='gap': slots=[0];pivot=-1;ids=[]
        elif kind=='orientation': slots=[1];pivot=-1;ids=[pos[b*10+1]]
        else:
            pc=g['entries'][0]['pivot'];slots=sorted([2+x['index'] for x in g['entries']]+[2+pc]);pivot=slots.index(2+pc)
            ids=[pos[b*10+s] for s in slots]
            q=[];j=0
            for k in range(len(slots)):
                if k==pivot:q.append(-sum(v,F(0)))
                else:q.append(v[j]);j+=1
            v=q;assert sum(v,F(0))==0
        c=[fq(x) for x in a['target'][b,slots]]
        groups.append(dict(state=b,kind=kind,slots=slots,pivot=pivot,ids=ids,c=c,v=v,
                           lower=[fq(lo[i]) for i in ids],upper=[fq(hi[i]) for i in ids]))
    dom=d['gap_domain'];dv=[F(0)]*206
    for i,j in zip(dom['independent'],d['normal_columns']):dv[i]=fq(t[j])
    dv[dom['pivot']]=-sum(dv,F(0));assert sum(dv,F(0))==0
    return groups,dv

def restrictions(d,a,t,groups,dv):
    """Rows l<=alpha*v<=u; exact rational dot products for the full f model."""
    out=[]
    for g in groups:
        if g['kind']=='gap':continue
        for k,(c,v,l,u) in enumerate(zip(g['c'],g['v'],g['lower'],g['upper'])):
            out.append((l-c,u-c,v,('target',g['state'],g['slots'][k])))
    dom=d['gap_domain']
    for i,(c,v,l,u) in enumerate(zip(a['drops'],dv,dom['lower'],dom['upper'])):
        out.append((l-c,u-c,v,('drop',i)))
    C=a['C'].tocsr();tf=list(map(fq,t))
    for i in range(C.shape[0]):
        begin,end=C.indptr[i:i+2]
        v=sum((fq(c)*tf[j] for c,j in zip(C.data[begin:end],C.indices[begin:end])),F(0))
        out.append((fq(a['full_step_lower'][i]),fq(a['full_step_upper'][i]),v,('full_linear_control',i)))
    return out

def ray_interval(rows,cap=F(1)):
    lower=F(0);upper=cap;low_owner=None;high_owner=None
    for l,u,v,owner in rows:
        if l>u:raise ValueError('Empty original full-control step interval: '+str(owner))
        if not v:
            if not l<=0<=u:raise ValueError('Zero ray cannot restore required bound: '+str(owner))
            continue
        x,y=(l/v,u/v) if v>0 else (u/v,l/v)
        if x>lower:lower=x;low_owner=owner
        if y<upper:upper=y;high_owner=owner
    if lower>upper or upper<0:raise ValueError('No shared feasible affine-ray interval')
    return lower,upper,low_owner,high_owner

def decode(d,a,t):
    groups,dv=pieces(d,a,t);rows=restrictions(d,a,t,groups,dv)
    lower,alpha,low_owner,high_owner=ray_interval(rows)
    if any(float(x)!=0 for x in t) and alpha==0:
        raise ValueError('Nonzero outward direction has only the zero feasible shared ray')
    target=a['target'].copy();load_error=F(0);round_error=F(0)
    for g in groups:
        q=[c+alpha*v for c,v in zip(g['c'],g['v'])]
        z=[float(x) for x in q]
        round_error=max(round_error,max((abs(fq(v)-x) for v,x in zip(z,q)),default=F(0)))
        if g['kind']!='gap':
            assert all(l<=fq(v)<=u for l,v,u in zip(g['lower'],z,g['upper']))
        if g['kind']=='load':
            assert sum(q,F(0))==sum(g['c'],F(0))
            err=abs(sum(map(fq,z),F(0))-sum(g['c'],F(0)))
            assert err<=len(z)*F(1,2**54);load_error=max(load_error,err)
        target[g['state'],g['slots']]=z
    q=[c+alpha*v for c,v in zip(a['drops'],dv)]
    if not any(dv):drops=a['drops'].copy();raised=[]
    else:drops,raised=joint_round(q,d['gap_domain'])
    target[np.asarray(d['gap_layout']['normal_states']),0]=gaps(drops,d['gap_domain'])[0]
    assert all(l<=alpha*v<=u for l,u,v,_ in rows)
    assert np.array_equal(target[d['g']['fixed']],a['target'][d['g']['fixed']])
    assert np.all(target[:,2:10][d['g']['inactive']]==0)
    return target,drops,dict(alpha=pair(alpha),alpha_lower=pair(lower),limiting_lower=low_owner,
                            limiting_upper=high_owner,restriction_count=len(rows),raised=raised,
                            load_rounding_sum_error=pair(load_error),target_rounding_error=pair(round_error),
                            drops=[pair(v) for v in drops],shared_unrounded_direction=True,
                            dependent_phase_guarantee='Linearized bound only; actual root/phase/hard checks mandatory')
