"""Independent saved-array admission; does not import either teacher or simulator."""
import json,sys,importlib.util,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=ROOT/'experiments/full_body_contact_trainer_4090_step_v5/audit.py'
sys.path.insert(0,str(src.parent))
spec=importlib.util.spec_from_file_location('original_furniture_physical_audit',src);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
def audit(out):
    out=Path(out)
    if (out/'planning_failure.json').exists():
        r=dict(integrity_pass=True,admitted=False,classification='PLANNING_REJECTION_NOT_FULL_ROLLOUT',details=read(out/'planning_failure.json'))
    else:
        sitting=(out/'furniture_trajectory.npz').exists();name='furniture_trajectory.npz' if sitting else 'trajectory.npz'
        manifest=read(out/('furniture_manifest.json' if sitting else 'manifest.json'))
        with np.load(out/name) as z:d=dict(z)
        assert all(v.shape[:2]==(1000,1) and np.isfinite(v).all() for v in d.values())
        result=original.physical(d)
        for key in ['failure','legal','no_slip']:np.testing.assert_array_equal(result[key],d[key])
        meta=read(ROOT/'outputs/full_body_contact_imitation_v4/dataset'/manifest['task']/'metadata.json')
        delta=abs(d['command'][:,:,:23]-d['previous'][:,:,:23]);assert delta.max()<=.0030002
        assert np.array_equal(d['command'][:,:,23:],d['previous'][:,:,23:])
        assert (d['command'][:,:,:23]>=np.array(meta['lower'])+.03-2e-7).all()
        assert (d['command'][:,:,:23]<=np.array(meta['upper'])-.03+2e-7).all()
        if sitting:
            np.testing.assert_array_equal(d['tick'][:,0],np.arange(1000))
            capture=read(out/'furniture_capture.json');assert capture['samples']==1000 and capture['full_standing_start']
            startup=read(out/'furniture_startup.json');warm=startup['warm_failure'][0]
            assert startup['warm_steps']==25 and startup['physics_steps']==100 and capture['physics_steps']==4100
        else:
            assert read(out/'results.json')['episodes'][0]['full_standing_start']
            warm=read(out/'startup.json')['warm_failure'][0]
            expected=d['previous'][:,:,:23]+np.clip(.1*(np.clip(d['desired'],np.array(meta['lower'])+.03,np.array(meta['upper'])-.03)-d['previous'][:,:,:23]),-.003,.003)
            expected=np.where(d['learn'][...,None],expected,d['previous'][:,:,:23])
            assert abs(expected-d['command'][:,:,:23]).max()<=2e-7
        mapping=read(out/'contact_mapping.json');assert mapping['validated']==all(f['mapping_valid'] for f in mapping['frames'])
        body_ids=[0,1,2] if sitting else [3,4]
        load=((d['physical_force_w']*d['physical_normal_w']).sum(-1).clip(min=0)*d['physical_inside_patch'])[:,:,body_ids].sum(-1)/d['physical_bodyweight_N']
        hold=0
        for good in result['good'][:,0]:hold=hold+1 if good and not warm else 0
        admitted=bool(result['clean'][0] and not warm and mapping['validated'] and (load[-250:]>=.05).all())
        r=dict(integrity_pass=True,admitted=admitted,classification='VERIFIED_DEMONSTRATION' if admitted else 'FAILED_ATTEMPT',variant=manifest['variant'],
            full20s=True,physical_clean=bool(result['clean'][0]),final_hold_s=hold*.02,minimum_final_intended_load_bw=float(load[-250:].min()),
            slip_ticks=int((~result['no_slip']).sum()),illegal_ticks=int((~result['legal']).sum()),warm_failure=bool(warm),mesh_mapping_validated=mapping['validated'],
            first_failure_s=.02*(1+int(np.flatnonzero(result['failure'][:,0])[0])) if result['failure'].any() else None,
            max_command_increment_rad=float(delta.max()),files={name:sha(out/name)},independent_auditor_sha256=sha(src))
    (out/'independent_audit.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':print(json.dumps(audit(sys.argv[1])),flush=True)
