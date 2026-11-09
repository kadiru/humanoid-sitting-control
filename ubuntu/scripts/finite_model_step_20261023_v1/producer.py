"""One unscaled finite map and conditional nonlinear value/J evaluation."""
from common import *

def main():
    origin=admission('producer'); c=contract()
    # All hash/loading/model/audit costs share the parent's original deadline.
    for p,h in {**c['input_sha256'],**c['runtime_sha256']}.items():
        assert sha(p)==h,p
        check(origin)
    assert read(PRIOR/'summary.json')['candidate_accepted']
    a=read(PRIOR/'audit.json')
    assert a['passed'] and a['candidate_accepted'] and a['exact_margin']['passed']
    assert a['exact_margin']['rows_sha256']==sha(PRIOR/'exact_rows.jsonl')
    assert a['unchanged_primal_sha256']==sha(PRIOR/'witness.npy')
    assert read(PRIOR/'inherited_input_identity.json')['passed']
    assert read(INTERIOR/'input_audit.json')['new_state']
    assert read(CURRENT/'input_proof.json')['passed']
    for p in [D/'lifted_feasibility_audit_491/independent_audit.json',
              D/'torque_path_audit_525/independent_audit.json']:
        assert read(p)['passed']
    write(O/'input_verified.json',dict(passed=True,contract_sha256=sha(HERE/'contract.json'),
        inherited_exact_margin=a['exact_margin'],no_certificate_or_matrix_products_repeated=True))
    event('input_verified',origin)
    import numpy as np
    from fractions import Fraction as F
    sys.path.insert(0,str(R/'scripts'))
    from low_seat_exact_gap_chart_468 import joint_round,gaps,fq,pair
    from assessment import assess,encoded_prediction
    d=load_context();z=d['z'];f=z['features'];y=z['actualY'];J=z['J']
    # H's three saved blocks are bound by491/525 plus the successful input audit.
    # Reuse certified products; never recompute H*d, P*d, N*d or eta.
    v=[]
    for i,line in enumerate((PRIOR/'exact_rows.jsonl').read_text().splitlines()):
        row,n,den=json.loads(line);assert row==i
        v.append(F(n,den))
    assert len(v)==92267
    delta=[F(0)]*105750
    for row,value in zip(z['varying'],v[11906:92061]): delta[int(row)]=value
    for row,value in zip(z['control_feature_rows'],v[:11906]): assert delta[int(row)]==value
    event('map_started',origin,shared_alpha=[1,1],finite_maps=1)
    q=[fq(x)+a for x,a in zip(f.ravel(),delta)]
    out=np.asarray(list(map(float,q))).reshape(f.shape)
    drops=[F(*x) for x in d['intervals']['drops']];dd=v[92061:]
    raw=[x+a for x,a in zip(drops,dd)]
    if any(dd): nd,raised=joint_round(raw,d['dom'])
    else: nd,raised=drops.copy(),[]
    out[d['normal'],71]=gaps(nd,d['dom'])[0]
    out[d['active'],24:38]=0.
    worst=F(0)
    for b in d['active']:
        before=sum(map(fq,f[b,16:24]),F(0))
        assert sum(q[b*75+16:b*75+24],F(0))==before
        after=sum(map(fq,out[b,16:24]),F(0))
        worst=max(worst,abs(after-before));out[b,38]=float(after-1)
    assert worst<=8*F(1,2**54)
    controls=out.ravel()[z['control_feature_rows']]
    assert np.all(controls>=z['lower']) and np.all(controls<=z['upper'])
    for i,row in enumerate(z['varying']):
        change=fq(out.ravel()[row])-fq(f.ravel()[row]); k=11906+i
        assert F(*d['intervals']['closed_lower'][k])<=change<=F(*d['intervals']['closed_upper'][k])
    assert np.array_equal(out[d['mesh']['fixed']],f[d['mesh']['fixed']])
    assert np.all(out[:,16:24][d['mesh']['inactive']]==0) and np.all(np.isfinite(out))
    np.savez(O/'mapped.npz',features=out)
    write(O/'map.json',dict(passed=True,shared_alpha=[1,1],raised=raised,
        drop_state=[pair(x) for x in nd],load_sum_rounding=pair(worst),
        inherited_certificate_sha256=sha(PRIOR/'exact_margin.json'),
        trace_sha256=sha(PRIOR/'exact_rows.jsonl'),finite_maps=1))
    event('map_completed',origin)
    # Changed states only, using the frozen original mechanics implementation.
    from low_seat_clock_transcription_444 import load_inputs,Mechanics
    changed=np.flatnonzero(np.any(f[:,:24]!=out[:,:24],axis=1));ny=y.copy()
    assert len(changed)<=1365
    if len(changed):
        oracle=Mechanics(load_inputs(),d['mesh']['canonical'][changed])
        event('model_values_started',origin,records=len(changed))
        ny[changed]=oracle.values(out[changed,:24])
        event('model_values_completed',origin)
    assert np.all(np.isfinite(ny))
    state,rows=assess(d,out,ny)
    mu=read(PRIMARY/'ceiling.json')['mu']
    predphi=encoded_prediction(d,f,out,y,J)
    pred=mu-predphi;actual=mu-state['phi'];ratio=actual/pred if pred>0 else None
    take=bool(state['auxiliary_acceptance_domain_pass'] and pred>0 and actual>0 and ratio>=.1)
    np.savez(O/'actual.npz',Y=ny,changed=changed,**rows)
    result=dict(summary=state,mu=mu,predicted_phi=predphi,predicted_reduction=pred,
        actual_reduction=actual,ratio=ratio,trial_merit_accepted=take,
        finite_maps=1,value_requests=int(bool(len(changed))),value_state_records=len(changed),
        jacobian_requests=0,jacobian_state_records=0,physical_admission=False,
        source_mechanics_sha256=sha(R/'scripts/low_seat_clock_transcription_444.py'))
    write(O/'decision.json',result)
    if take:
        nJ=J.copy()
        if len(changed):
            event('new_jacobian_started',origin,records=len(changed))
            nJ[changed]=oracle.jacobian(out[changed,:24])
            event('new_jacobian_completed',origin)
        assert np.all(np.isfinite(nJ))
        np.savez(O/'accepted_J.npz',J=nJ,changed=changed)
        result.update(jacobian_requests=int(bool(len(changed))),jacobian_state_records=len(changed))
        write(O/'decision.json',result)
    event('producer_completed',origin)

if __name__=='__main__':main()
