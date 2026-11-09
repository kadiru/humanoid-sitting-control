"""Saved-array-only preparation416; no FK, root, derivative or optimizer rerun."""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from low_seat_direct_constraints_416 import partition,coordinate_bounds,assemble,box
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'direct_preparation_416'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False)
    prior=json.loads((D/'monotone_preparation_410/preparation.json').read_text());audit=json.loads((D/'monotone_preparation_audit_411/independent_audit.json').read_text())
    assert prior['passed'] and audit['passed']
    f=dict(np.load(D/'monotone_preparation_410/fixture.npz'));base=dict(np.load(D/'window_path_400/initial.npz'))
    lo,hi=coordinate_bounds(base,f['shape']);hard=partition();keep=f['keep'];scale=f['scale'];soft=keep&~hard
    initial_merit=float(max(0.,np.max(-f['iq'][soft]/scale[soft])));slack=initial_merit+1e-6
    q,J=assemble(f['iq'],f['v'],slack,scale,keep,lo,hi,f['I'],f['G']);lower,upper=box(f['z'])
    assert np.all(f['iq'][keep&hard]>=0) and np.all(q>=0)
    assert np.array_equal(f['z'][894:],np.array([93,194,246,101,16])) and f['z'][894:].sum()==650
    # Algebra-only scalar fixture; the unchanged full map derivative was already
    # checked410/411. The eliminated five budget directions are EXACT zero.
    rng=np.random.default_rng(416);direction=rng.normal(size=895);direction/=np.linalg.norm(direction)
    expanded=np.r_[direction[:894],np.zeros(5)]
    m=int(keep.sum());separate=np.r_[(f['I']@expanded)[keep]/scale[keep]+(~hard[keep])*direction[-1],(f['G']@expanded)[:786],-(f['G']@expanded)[:786]]
    error=float(abs(J@direction-separate).max());assert error<2e-12
    np.savez_compressed(O/'fixture.npz',seed=f['z'],slack=slack,initial_merit=initial_merit,hard=hard,keep=keep,scale=scale,lo=lo,hi=hi,values=q,jacobian=J,lower=lower,upper=upper,direction=direction,expanded_direction=expanded,separate_direction=separate)
    hashes=dict(prior['source_input_sha256']);hashes.update(audit['source_sha256'])
    for p in [Path(__file__),Path('scripts/low_seat_direct_constraints_416.py'),D/'monotone_preparation_410/fixture.npz',D/'monotone_preparation_audit_411/independent_audit.json',D/'window_path_400/initial.npz',D/'monotone_saved_rejections_415.json']:
        hashes[str(p.resolve())]=sha(p)
    for path,h in hashes.items():assert sha(path)==h,path
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,continuous_free_variables=894,nonnegative_epigraph_variables=1,physical_model_window_rows=27708,additional_original_coordinate_bound_rows=1572,retained_augmented_rows=len(q),hard_rows=int((hard&keep).sum()),soft_rows=int(soft.sum()),static_command_rows_newly_hard=3648,fixed_event_budgets=f['z'][894:].tolist(),minimum_seed_hard_row=float(f['iq'][hard&keep].min()),minimum_seed_augmented_row=float(q.min()),initial_soft_merit=initial_merit,algebra_directional_error=error,original_derivative_check_410=prior['full_directional_error'],unchanged_derivative_bound=2e-6,new_FK_or_root_or_derivative_evaluations=0,optimizer_calls=0,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False,source_input_sha256=hashes)
    (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2))
if __name__=='__main__':main()
