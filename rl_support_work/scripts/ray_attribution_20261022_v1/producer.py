"""One exact bound attribution for the immutable saved ray; no candidate output."""
from common import *
from fractions import Fraction as F
import math

def dyad(v):
    n,d=float(v).as_integer_ratio();return n,d.bit_length()-1
def add(s,e,n,k):
    if k>e:return (s<<(k-e))+n,k
    return s+(n<<(e-k)),e
def ratio(m,me,s,se):
    return m<<max(0,se-me),s<<max(0,me-se)
def paired(n,d):
    f=F(n,d);return [str(f.numerator),str(f.denominator)]
def row_values(M,i,vector,mu,rhs):
    s=e=0;mn,me=dyad(rhs)
    for j in range(int(M.indptr[i]),int(M.indptr[i+1])):
        col=int(M.indices[j]);n,k=dyad(M.data[j]);v,q=vector[col]
        s,e=add(s,e,n*v,k+q)
        if col==78790:mn,me=add(mn,me,-n*mu[0],k+mu[1])
    assert max(abs(s).bit_length(),abs(mn).bit_length())<=4352 and max(e,me)<=2148
    return mn,me,s,e
def feature_unit(j):
    if j in [0,1,24,25,26,30,31,32,71,72,73,74]:return 'm'
    if j in [2,3]:return 'dimensionless phase'
    if 4<=j<16 or 39<=j<55:return 'rad'
    if 16<=j<24 or j==38:return 'BW fraction'
    if j in [27,28,29,33,34,35]:return '0.1*orientation vee'
    if j in [36,37]:return 'BW*m'
    if 55<=j<71:return 'Nm'
    raise AssertionError(j)
def physical_unit(group,terms):
    if 'coordinate' in group:return 'dimensionless original coordinate combination'
    if 'ramp' in group:return 'rad*s'
    if 'phase' in group:return 'dimensionless phase'
    if any(x in group for x in ['command','rate','window','target']):return 'rad'
    if 'effort' in group:return 'Nm'
    if 'force' in group:return 'BW fraction'
    if any(x in group for x in ['gap','hover','support','approach','descent']):return 'm'
    if 'planning' in group:
        units={feature_unit(t['feature']) for t in terms};assert len(units)==1;return units.pop()
    raise AssertionError(('unmapped original group',group))

def main():
    origin=admission('producer');spec=contract()
    for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():
        assert sha(path)==h,path;check(origin)
    import numpy as np
    from scipy import sparse
    assert read(RAY/'audit.json')['exact_stored_binary64_primal_feasible']
    old=read(RAY/'candidate.json');mu=float(old['mu'])
    assert read(ACCEPTED/'decision.json')['summary']['phi']==mu
    assert read(PRIOR/'input_proof.json')['new_state_full_input_proved']
    M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr();a=dict(np.load(PRIOR/'full_arrays.npz'))
    E=sparse.load_npz(PRIOR/'full_equality.npz');raw=np.load(RAY/'raw_direction.npy')
    assert M.shape==(652529,78791) and M.nnz==2614306 and M.has_canonical_format
    assert E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
    assert raw.shape==(78790,) and raw.dtype==np.dtype('float64') and np.isfinite(raw).all()
    assert np.all(raw[:10540]==0) and 0<mu<1
    assert np.array_equal(a['bounds'][-1],[0.,1.])
    assert np.isneginf(a['bounds'][:-1,0]).all() and np.isposinf(a['bounds'][:-1,1]).all()
    assert np.diff(M.indptr).max()<=256 and np.isfinite(M.data).all() and np.isfinite(a['rhs']).all()
    v=list(map(dyad,raw));v.append(dyad(-mu/2));dm=dyad(mu)
    assert F(v[-1][0],1<<v[-1][1])==-F.from_float(mu)/2
    bestn=bestd=1;bestfloat=1.;bucket={};outward=0
    threshold=F.from_float(math.nextafter(bestfloat,math.inf))
    for i in range(M.shape[0]):
        if i%256==0:check(origin)
        m,me,s,se=row_values(M,i,v,dm,a['rhs'][i]);assert m>=0,('origin infeasible',i)
        if s<=0:continue
        outward+=1;n,d=ratio(m,me,s,se);assert n>=0 and d>0
        if n*bestd<bestn*d:
            f=float(F(n,d));assert f>0
            if f!=bestfloat:bucket={}
            bestn,bestd=n,d;bestfloat=f;threshold=F.from_float(math.nextafter(f,math.inf))
        if n*threshold.denominator<=threshold.numerator*d and float(F(n,d))==bestfloat:
            bucket[i]=(m,me,s,se,n,d)
            assert len(bucket)<=10000,'Attribution detail bound exceeded; no partial result accepted'
    assert bestn<bestd and bucket
    # Re-filter the rounded bucket after the last exact-minimum change.
    bucket={i:t for i,t in bucket.items() if float(F(t[4],t[5]))==bestfloat}
    ties=[i for i,t in bucket.items() if t[4]*bestd==bestn*t[5]];assert ties
    tie_set=set(ties)
    rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
    G=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz').tocsr()
    kept=np.flatnonzero(rows['keep']);assert len(kept)==331495 and G.shape==(331888,105750)
    ref=dict(np.load(BASE/'reference.npz'));active=ref['active'];assert len(active)==1365
    mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));times=read(D/'clock_transcription_preparation_444/exact_times.json')
    assert len(mesh['canonical'])==1410 and len(times)==1467
    intervals=read(PRIOR/'intervals.json');assert len(intervals['owners'])==92267
    OUTPUT=list(range(14))+list(range(15,51));details=[]
    def term(state,feature,value):
        label=int(mesh['canonical'][state]);assert mesh['label_to_state'][label]==state
        return dict(state=int(state),feature=int(feature),coefficient=float(value),canonical_label=label,exact_time=times[label])
    for i,t in bucket.items():
        check(origin);m,me,s,se,n,d=t
        desc=dict(full_row=i,exact_limiting=i in tie_set,exact_margin=paired(m,1<<me),exact_slope=paired(s,1<<se),exact_ratio=paired(n,d),normalized_margin=float(F(m,1<<me)),normalized_slope=float(F(s,1<<se)))
        if i<331495:
            j=int(kept[i]);group=str(rows['groups'][j]);terms=[]
            for z in range(G.indptr[j],G.indptr[j+1]):
                state,feature=divmod(int(G.indices[z]),75);terms.append(term(state,feature,G.data[z]))
            desc.update(family='original_signed',original_row=j,group=group,scale=float(rows['scale'][j]),soft=bool(rows['soft'][j]),source_terms=terms,units=physical_unit(group,terms))
        elif i<467995:
            j=(i-331495)%68250;state=int(active[j//50]);output=OUTPUT[j%50]
            desc.update(family='consistency',sign=1 if i<399745 else -1,state=state,output=output,units=feature_unit(24+output),scale=float(ref['sigma'][state,output]))
        else:
            j=(i-467995)%92267;owner=intervals['owners'][j]
            desc.update(family='closed_interval',sign=1 if i<560262 else -1,interval_row=j,owner=owner,units='normalized interval width; owner identifies underlying coordinate')
            if owner[0]=='drop':desc.update(underlying_units='m',drop_edge=int(owner[1]))
            else:
                assert owner[0] in ['original_control','feature_trust']
                row=int(ref['control_feature_rows'][owner[1]]) if owner[0]=='original_control' else int(owner[1])
                b,f=divmod(row,75);desc.update(underlying_units=feature_unit(f),coordinate=term(b,f,1.))
        desc['native_terms']=[[int(M.indices[z]),float(M.data[z])] for z in range(M.indptr[i],M.indptr[i+1])]
        details.append(desc)
    bound=F(bestn,bestd);alpha=F.from_float(old['alpha'])
    # This is a comparison with the saved step, not a new step selection.
    write(O/'attribution.json',dict(attribution_passed=True,exact_alpha_bound=paired(bestn,bestd),alpha_bound_float=bestfloat,
        limiting_rows=ties,rounded_indistinguishable_rows=list(bucket),rows=details,full_rows=M.shape[0],products=M.nnz,
        positive_slope_rows=outward,saved_alpha=old['alpha'],saved_alpha_over_exact_bound=float(alpha/bound),
        exact_bound_over_saved_alpha=float(bound/alpha),
        bound_within_expected_half_then_dyadic_band=bool(2*alpha<=bound<=4*alpha),
        scalar_and_initial_alpha_bounds_not_limiting=True,
        scope='Exact real-ray bounds on the stored binary64 coefficients; no new point, encoding or physical result. Rounded-indistinguishable means ratios round to the same binary64 as the exact minimum.',
        arithmetic_note='The producer used conservative interval slopes, a half-step reserve and dyadic rounding. The reported exact bound distinguishes the original-row restriction from that combined conservatism without replaying the old interval calculation.',
        candidate_created=False,old_direction_modified=False))
    assert (O/'attribution.json').stat().st_size<256*1024**2
    event('attribution_complete',origin)

if __name__=='__main__':main()
