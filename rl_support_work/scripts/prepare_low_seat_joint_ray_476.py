"""New shared-ray arithmetic preparation; no model/root/LP/path evaluation."""
import json,datetime
from pathlib import Path
from scipy import sparse
from low_seat_free_gap_solver_470 import load_saved,read,sha,D,unpack
from low_seat_exact_gap_chart_468 import fixture_center
from low_seat_joint_ray_476 import *
O=D/'joint_ray_preparation_476'

def main():
    O.mkdir(exist_ok=False);d=load_saved();P=D/'free_gap_path_472/linear_00'
    # The sealed initial472 state and its already audited reference matrix are
    # research inputs only. No rejected472 witness/trial is read or continued.
    a=dict(np.load(P/'input.npz'));m=read(P/'metadata.json')
    a.update(C=sparse.load_npz(P/'C.npz'),drops=list(map(unpack,m['drops'])),mode=m['mode'],trust=m['trust'])
    seed=dict(np.load(D/'free_gap_path_472/initial.npz'))
    assert np.array_equal(seed['X'],a['X']) and np.array_equal(seed['target'],a['target'])
    z,zd,zi=decode(d,a,np.zeros(d['B'].shape[1]));assert np.array_equal(z,a['target']) and zd==a['drops']
    # Interior drop center is an ARITHMETIC fixture, not a path or solver seed.
    a['drops']=fixture_center(d['gap_domain']);a['target']=a['target'].copy()
    a['target'][np.asarray(d['gap_layout']['normal_states']),0]=gaps(a['drops'],d['gap_domain'])[0]
    np.savez_compressed(O/'fixture_origin.npz',target=a['target'],origin=a['origin'],z=a['z'],full_step_lower=a['full_step_lower'],full_step_upper=a['full_step_upper'])
    (O/'fixture_drop_origin.json').write_text(json.dumps([pair(x) for x in a['drops']]))
    fixtures=[];basis_error=0.;n=d['B'].shape[1]
    for pattern in range(3):
        t=np.array([((j+2*pattern)%7-3 or 1)*1e-6 for j in range(n)])
        t[d['normal_columns']]*=.001
        groups,dv=pieces(d,a,t);rows=restrictions(d,a,t,groups,dv)
        _,limit,_,_=ray_interval(rows,F(10**30));assert limit>0
        # Scalars are derived solely from the exact convex-domain ray bound.
        # These three finite amplitudes check interior and bound-limited maps.
        for scale in [.25,1.25,2.]:
            u=t*float(limit)*scale;out,drops,info=decode(d,a,u)
            k=len(fixtures);np.savez_compressed(O/f'fixture_{k:02d}.npz',tangent=u,target=out)
            (O/f'fixture_{k:02d}.json').write_text(json.dumps(info))
            fixtures.append(dict(index=k,pattern=pattern,scale=scale,alpha=float(unpack(info['alpha'])),load_sum_error=float(unpack(info['load_rounding_sum_error']))))
        # Analytic unrounded direction is exactly the old B incidence map,
        # up to the explicitly measured float conversion used in this compare.
        raw=np.zeros_like(a['target'])
        for g in groups:raw[g['state'],g['slots']]=[float(x) for x in g['v']]
        cumulative=F(0);ns=d['gap_layout']['normal_states']
        for k,v in enumerate(dv):cumulative+=v;raw[ns[k+1],0]=-float(cumulative)
        basis_error=max(basis_error,float(abs(raw.ravel()-d['B']@t).max()))
    assert basis_error<2e-6
    assert all(f['alpha']==1 for f in fixtures if f['scale']==.25)
    assert all(0<f['alpha']<1 for f in fixtures if f['scale']>1)
    # Explicit scalar endpoint/outside restoration contracts, no candidate.
    bad=[(F(-1),F(0),F(1),('endpoint',0))]
    l,h,_,_=ray_interval(bad);assert l==h==0
    try:ray_interval([(F(1),F(2),F(0),('zero',0))]);raise AssertionError('must reject')
    except ValueError:pass
    proof=read(D/'simplex_preparation_460/preparation.json')['rounding_proof']
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,independent_coordinates=n,finite_joint_fixtures=len(fixtures),all_coordinate_patterns=3,fixtures=fixtures,continuous_basis_error=basis_error,reference_zero_identity=True,shared_alpha_for_all_controls=True,full_linear_dependent_f_bounds=True,actual_dependent_f_guaranteed=False,load_rounding_proof=proof,normal_drop_sum_exact=True,finite_map_differentiable=False,reference_state='Sealed472 INITIAL, never a rejected trial or accepted optimizer continuation',fixture_center_is_not_candidate=True,model_calls=0,root_calls=0,optimizer_calls=0,new_complete_path=False,physical_admission=False)
    (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
