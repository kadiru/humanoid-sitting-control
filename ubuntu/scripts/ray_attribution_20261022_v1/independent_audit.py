"""Independent fixed-grid integer attribution check; imports no producer code."""
from common import *
from fractions import Fraction
import math

Q=1<<1074
def grid(x):
    n,d=float(x).as_integer_ratio()
    assert Q%d==0
    return n*(Q//d)
def unit_of_feature(i):
    if i in (0,1,24,25,26,30,31,32,71,72,73,74):return 'm'
    if i in (2,3):return 'dimensionless phase'
    if i in range(4,16) or i in range(39,55):return 'rad'
    if i in range(16,24) or i==38:return 'BW fraction'
    if i in (27,28,29,33,34,35):return '0.1*orientation vee'
    if i in (36,37):return 'BW*m'
    if i in range(55,71):return 'Nm'
    raise AssertionError(i)
def group_unit(g,terms):
    mapping={'coordinate':'dimensionless original coordinate combination','ramp':'rad*s','phase':'dimensionless phase','command':'rad','rate':'rad','window':'rad','target':'rad','effort':'Nm','force':'BW fraction','gap':'m','hover':'m','support':'m','approach':'m','descent':'m'}
    for part,unit in mapping.items():
        if part in g:return unit
    assert 'planning' in g
    units=set(unit_of_feature(t['feature']) for t in terms);assert len(units)==1
    return next(iter(units))
def from_pair(x):return Fraction(int(x[0]),int(x[1]))

def main():
    origin=admission('audit');c=contract()
    for p,h in {**c['input_sha256'],**c['runtime_sha256']}.items():
        assert sha(p)==h,p;check(origin)
    import numpy as np
    from scipy import sparse
    out=read(O/'attribution.json');assert out['attribution_passed'] and not out['candidate_created']
    old=read(RAY/'candidate.json');mu=read(ACCEPTED/'decision.json')['summary']['phi'];assert old['mu']==mu
    raw=np.load(RAY/'raw_direction.npy');assert raw.shape==(78790,) and np.isfinite(raw).all() and np.all(raw[:10540]==0)
    M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr();a=dict(np.load(PRIOR/'full_arrays.npz'));E=sparse.load_npz(PRIOR/'full_equality.npz')
    assert M.shape==(652529,78791) and M.nnz==2614306 and M.has_canonical_format and np.diff(M.indptr).max()<=256
    assert E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
    assert np.array_equal(a['bounds'][-1],[0.,1.]) and np.isneginf(a['bounds'][:-1,0]).all() and np.isposinf(a['bounds'][:-1,1]).all()
    bound=from_pair(out['exact_alpha_bound']);assert 0<bound<1 and 0<mu<1
    assert out['alpha_bound_float']==float(bound)
    numerator,denominator=bound.numerator,bound.denominator
    upper=Fraction.from_float(math.nextafter(float(bound),math.inf))
    vector=[grid(v) for v in raw];vector.append(grid(-mu/2));mq=grid(mu)
    assert 2*vector[-1]==-mq
    ties=[];rounded=[];proof_values={};positive=0;products=0
    for i in range(M.shape[0]):
        if i%256==0:check(origin)
        s=0;margin=grid(a['rhs'][i])*Q
        for j in range(int(M.indptr[i]),int(M.indptr[i+1])):
            col=int(M.indices[j]);coef=grid(M.data[j]);s+=coef*vector[col];products+=1
            if col==78790:margin-=coef*mq
        assert max(abs(s).bit_length(),abs(margin).bit_length())<=4352
        assert margin>=0,('origin',i)
        if s<=0:continue
        positive+=1;difference=margin*denominator-s*numerator
        assert difference>=0,('unreported tighter row',i)
        if difference==0:ties.append(i)
        if margin*upper.denominator<=s*upper.numerator and float(Fraction(margin,s))==float(bound):
            rounded.append(i);proof_values[i]=(margin,s)
    assert ties==out['limiting_rows'] and rounded==out['rounded_indistinguishable_rows']
    assert ties and len(rounded)<=10000 and positive==out['positive_slope_rows']
    assert products==out['products']==2614306 and out['full_rows']==652529
    # Independently resolve each row to original units/state/time from saved maps.
    meta=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
    kept=list(map(int,np.flatnonzero(meta['keep'])));assert len(kept)==331495
    G=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz').tocsr()
    ref=dict(np.load(BASE/'reference.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
    times=read(D/'clock_transcription_preparation_444/exact_times.json');owners=read(PRIOR/'intervals.json')['owners']
    assert G.shape==(331888,105750) and len(owners)==92267
    byrow={d['full_row']:d for d in out['rows']};assert list(byrow)==rounded
    for i in rounded:
        check(origin);record=byrow[i];m,s=proof_values[i]
        assert from_pair(record['exact_margin'])==Fraction(m,Q*Q)
        assert from_pair(record['exact_slope'])==Fraction(s,Q*Q)
        assert from_pair(record['exact_ratio'])==Fraction(m,s)
        assert record['normalized_margin']==float(Fraction(m,Q*Q))
        assert record['normalized_slope']==float(Fraction(s,Q*Q))
        assert record['exact_limiting']==(i in ties)
        assert record['native_terms']==[[int(M.indices[z]),float(M.data[z])] for z in range(M.indptr[i],M.indptr[i+1])]
        if i<331495:
            j=kept[i];expected=[]
            for z in range(G.indptr[j],G.indptr[j+1]):
                b,f=divmod(int(G.indices[z]),75);label=int(mesh['canonical'][b]);assert mesh['label_to_state'][label]==b
                expected.append(dict(state=b,feature=f,coefficient=float(G.data[z]),canonical_label=label,exact_time=times[label]))
            assert record['family']=='original_signed' and record['original_row']==j
            assert record['source_terms']==expected and record['group']==str(meta['groups'][j])
            assert record['scale']==float(meta['scale'][j]) and record['soft']==bool(meta['soft'][j])
            assert record['units']==group_unit(str(meta['groups'][j]),expected)
        elif i<467995:
            local=(i-331495)%68250;b=int(ref['active'][local//50]);ids=list(range(14))+list(range(15,51));output=ids[local%50]
            assert record['family']=='consistency' and record['state']==b and record['output']==output
            assert record['sign']==(1 if i<399745 else -1)
            assert record['scale']==float(ref['sigma'][b,output]) and record['units']==unit_of_feature(24+output)
        else:
            row=(i-467995)%92267
            assert record['family']=='closed_interval' and record['interval_row']==row
            assert record['sign']==(1 if i<560262 else -1) and record['owner']==owners[row]
            assert record['units']=='normalized interval width; owner identifies underlying coordinate'
            kind,j=owners[row]
            if kind=='drop':assert record['underlying_units']=='m' and record['drop_edge']==j
            else:
                assert kind in ('original_control','feature_trust')
                frow=int(ref['control_feature_rows'][j]) if kind=='original_control' else int(j)
                b,f=divmod(frow,75);label=int(mesh['canonical'][b]);assert mesh['label_to_state'][label]==b
                assert record['underlying_units']==unit_of_feature(f)
                assert record['coordinate']==dict(state=b,feature=f,coefficient=1.,canonical_label=label,exact_time=times[label])
    alpha=Fraction.from_float(old['alpha'])
    assert out['saved_alpha']==old['alpha']
    assert out['saved_alpha_over_exact_bound']==float(alpha/bound)
    assert out['exact_bound_over_saved_alpha']==float(bound/alpha)
    assert out['bound_within_expected_half_then_dyadic_band']==(2*alpha<=bound<=4*alpha)
    assert out['scalar_and_initial_alpha_bounds_not_limiting']
    assert not out['old_direction_modified']
    check(origin)
    write(O/'audit.json',dict(passed=True,attribution_passed=True,full_rows_checked=M.shape[0],exact_products=products,
        exact_global_bound_verified=True,all_exact_ties_verified=True,all_rounded_indistinguishable_rows_verified=True,
        row_units_states_times_verified=True,limiting_rows=ties,
        arithmetic='Independent common2^-1074 integer grid; products/sums at2^-2148. No producer arithmetic imported.',
        scope='Saved-direction attribution only; no new candidate, optimality, encoder, nonlinear model or physical success.',
        attribution_sha256=sha(O/'attribution.json'),candidate_created=False,physical_success=False))
    event('independent_attribution_audit_complete',origin)

if __name__=='__main__':main()
