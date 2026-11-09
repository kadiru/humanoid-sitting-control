"""Independent current-state finite-output verification and599 metric replay; no producer import."""
from common import *

def main():
    origin=admission('audit'); c=contract()
    assert read(O/'producer_exit.json')['completed_within_deadline']
    assert read(O/'input_verified.json')['passed']
    for p,h in {**c['input_sha256'],**c['runtime_sha256']}.items():
        assert sha(p)==h,p
        check(origin)
    bind_predecessor(PRIMARY,origin,check);bind_predecessor(INTERIOR,origin,check)
    prior_audit=read(PRIOR/'audit.json')
    assert prior_audit['passed'] and prior_audit['candidate_accepted'] and prior_audit['exact_margin']['passed']
    assert sha(PRIOR/'witness.npy')==prior_audit['unchanged_primal_sha256']
    assert sha(PRIOR/'exact_rows.jsonl')==prior_audit['exact_margin']['rows_sha256']
    import numpy as np
    import math
    import struct
    from fractions import Fraction as F
    from audit_assessment import summary
    d=load_context();z=d['z'];f=z['features'];y=z['actualY'];J=z['J'];dom=d['dom']
    mapped=np.load(O/'mapped.npz')['features'];md=read(O/'map.json')
    assert md['passed'] and md['shared_alpha']==[1,1]
    assert md['trace_sha256']==sha(PRIOR/'exact_rows.jsonl')
    # Read the already-certified rows by their explicit owner, not producer offsets.
    features={};drops_delta={};controls={}
    with (PRIOR/'exact_rows.jsonl').open() as source:
        for i,line in enumerate(source):
            row,n,den=json.loads(line); assert row==i
            value=F(n,den)
            assert abs(value.numerator).bit_length()<=4213 and value.denominator.bit_length()<=2149
            owner,index=d['intervals']['owners'][row]
            target={'feature_trust':features,'drop':drops_delta,'original_control':controls}[owner]
            assert index not in target; target[index]=F(n,den)
    assert len(features)==80155 and len(drops_delta)==206 and len(controls)==11906
    assert i+1==92267
    for i,value in controls.items(): assert value==features.get(int(z['control_feature_rows'][i]),F(0))
    fq=lambda x:F(*float(x).as_integer_ratio())
    def nearest(actual,exact):
        assert math.isfinite(actual)
        center=fq(actual)
        left=fq(math.nextafter(actual,-math.inf))
        right=fq(math.nextafter(actual,math.inf))
        lo=(left+center)/2; hi=(right+center)/2
        assert lo<=exact<=hi
        if exact==lo or exact==hi:
            bits=struct.unpack('>Q',struct.pack('>d',actual))[0]
            assert bits&1==0
        if exact==0: assert math.copysign(1.,actual)==1.
    q=[]
    for i,value in enumerate(f.ravel()):
        n,den=float(value).as_integer_ratio();delta=features.get(i,F(0))
        q.append(F(n*delta.denominator+delta.numerator*den,den*delta.denominator))
    normal_rows=set((d['normal']*75+71).tolist())
    canonical_rows=set((z['active'][:,None]*75+np.arange(24,39)).ravel().tolist())
    for i,exact in enumerate(q):
        if i not in normal_rows and i not in canonical_rows: nearest(float(mapped.ravel()[i]),exact)
    assert np.all(mapped[z['active'],24:38]==0.)
    enc=[F(*x) for x in md['drop_state']]
    old=[F(*x) for x in d['intervals']['drops']]
    raw=[old[i]+drops_delta[i] for i in range(206)]
    total=dom['total'];pivot=dom['pivot'];ind=dom['independent'];U=F(1,2**59)
    assert sum(raw)==total and all(l<=x<=u for l,x,u in zip(dom['lower'],raw,dom['upper']))
    if any(drops_delta.values()):
        floors={}
        for i in ind:
            nearest_float=float(raw[i])
            candidate=math.nextafter(nearest_float,-math.inf) if fq(nearest_float)>raw[i] else nearest_float
            floors[i]=fq(candidate)
            assert floors[i]<=raw[i]<fq(math.nextafter(candidate,math.inf))
        mass=sum(floors.values());expected_raised=[]
        for i in ind:
            if mass>=total-dom['upper'][pivot]:break
            candidate=float(raw[i])
            if fq(candidate)<raw[i]:candidate=math.nextafter(candidate,math.inf)
            ce=fq(candidate);mass+=ce-floors[i];floors[i]=ce;expected_raised.append(i)
        assert md['raised']==expected_raised
        assert all(enc[i]==floors[i] for i in ind) and enc[pivot]==total-mass
    else:assert enc==old and md['raised']==[]
    assert sum(enc)==total and all(l<=x<=u for l,x,u in zip(dom['lower'],enc,dom['upper']))
    assert all(fq(float(enc[i]))==enc[i] and abs(enc[i]-raw[i])<=U for i in ind)
    assert abs(enc[pivot]-raw[pivot])<=205*U
    cumulative=dom['hover'];nearest(float(mapped[d['normal'][0],71]),cumulative)
    for i,drop in enumerate(enc):
        cumulative-=drop;nearest(float(mapped[d['normal'][i+1],71]),cumulative)
    assert cumulative==dom['terminal']
    worst=F(0)
    for b in z['active']:
        before=sum((fq(x) for x in f[b,16:24]),F(0));after=sum((fq(x) for x in mapped[b,16:24]),F(0))
        assert sum(q[b*75+16:b*75+24],F(0))==before
        worst=max(worst,abs(after-before));nearest(float(mapped[b,38]),after-1)
    assert worst==F(*md['load_sum_rounding']) and worst<=8*F(1,2**54)
    zz=mapped.ravel()[z['control_feature_rows']]
    assert np.all(zz>=z['lower']) and np.all(zz<=z['upper'])
    for i,row in enumerate(z['varying']):
        k=11906+i;change=fq(mapped.ravel()[row])-fq(f.ravel()[row])
        assert F(*d['intervals']['closed_lower'][k])<=change<=F(*d['intervals']['closed_upper'][k])
    assert np.array_equal(mapped[d['mesh']['fixed']],f[d['mesh']['fixed']])
    assert np.all(mapped[:,16:24][d['mesh']['inactive']]==0)
    check(origin)
    a=dict(np.load(O/'actual.npz'));ny=a['Y'];changed=np.flatnonzero(np.any(f[:,:24]!=mapped[:,:24],axis=1))
    unchanged=np.ones(1410,dtype=bool);unchanged[changed]=False
    assert np.array_equal(changed,a['changed']) and np.array_equal(ny[unchanged],y[unchanged])
    assert np.all(np.isfinite(ny)) and np.all(np.isfinite(mapped))
    import common as cmodule
    from reserve_bridge import posterior_check
    hard=posterior_check(cmodule,d,mapped,ny,origin,independent=True)
    ss,vals,_=summary(d,mapped,ny);decision=read(O/'decision.json')
    assert ss==decision['summary']
    for key,value in vals.items():assert np.array_equal(value,a[key])
    mu=read(PRIMARY/'ceiling.json')['mu'];delta=mapped-f
    modelY=y+np.einsum('nyx,nx->ny',J,delta[:,:24])
    predphi=summary(d,mapped,modelY)[0]['phi'];pred=mu-predphi;ared=mu-ss['phi'];ratio=ared/pred if pred>0 else None
    from audit_progress import evaluate
    gate=evaluate(mu,ss['phi'],pred,ratio,ss['auxiliary_acceptance_domain_pass'],d['baseline_actual_rows'],vals['actual_rows'],d['row']['keep'] & ~d['row']['soft'])
    gate['original_progress_passed']=gate['passed']
    gate['actual_hard_reserve_passed']=hard['passed']
    gate['passed']=bool(gate['passed'] and hard['passed'])
    assert gate==read(O/'progress_gate.json');take=gate['passed']
    target_gaps=read(O/'target_gaps.json')
    for left,right in [(622,1273),(1400,631)]:
        assert target_gaps[str(left)+':'+str(right)]==dict(before=[float(y[left,47]),float(y[right,47])],after=[float(ny[left,47]),float(ny[right,47])])
    assert all(decision[k]==v for k,v in dict(mu=mu,predicted_phi=predphi,predicted_reduction=pred,
        actual_reduction=ared,ratio=ratio,trial_merit_accepted=take,value_state_records=len(changed)).items())
    if take:
        aj=np.load(O/'accepted_J.npz');assert aj['J'].shape==J.shape
        assert np.array_equal(aj['changed'],changed) and np.array_equal(aj['J'][unchanged],J[unchanged])
        assert np.all(np.isfinite(aj['J'])) and decision['jacobian_state_records']==len(changed)
    else:assert not (O/'accepted_J.npz').exists() and decision['jacobian_state_records']==0
    event('audit_completed',origin)
    write(O/'audit.json',dict(passed=True,candidate_accepted=take,finite_map_verified=True,trial_merit_accepted=take,
        exact_point_certificate_inherited=True,exact_H_products_repeated=0,old_proofs_replayed=0,
        unscaled_direction=True,independent_nearest_float_checks=True,exact_drop_total=True,
        all_original_numerical_checks_retained=True,new_model_oracle_independently_recomputed=False,
        model_audit_scope='Frozen original444 mechanics, fresh-call provenance and changed-state identity; independent599 all-row/metric replay. No second oracle evaluation.',
        model_calls=0,physical_admission=False,output_sha256={p.name:sha(p) for p in O.iterdir()
            if p.name in ['mapped.npz','actual.npz','map.json','decision.json','accepted_J.npz']}))

if __name__=='__main__':main()
