"""Independent saved-array audit; no controller, planner or simulator imports."""
import sys,importlib.util,json
from pathlib import Path
import numpy as np
from common import ROOT,read,write,sha
SRC=ROOT/'experiments/full_body_contact_trainer_4090_step_v5'
spec=importlib.util.spec_from_file_location('original_physical_audit',SRC/'audit.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

def check(out):
    if (out/'planning_failure.json').exists():
        result=dict(passed_integrity=True,admitted=False,planning_failure=read(out/'planning_failure.json'))
        write(out/'independent_audit.json',result);return result
    with np.load(out/'trajectory.npz') as z:d=dict(z)
    assert all(a.shape[:2]==(1000,1) and np.isfinite(a).all() for a in d.values())
    result=original.physical(d);row,=read(out/'results.json')['episodes'];manifest=read(out/'manifest.json')
    for k in ['failure','good','no_slip','alive','legal']:np.testing.assert_array_equal(result[k],d[k])
    assert row['clean']==bool(result['clean'][0]) and row['full_standing_start'] and row['steps']==1000
    assert not read(out/'startup.json')['warm_failure'][0]
    hold=0
    for x in result['good'][:,0]:hold=hold+1 if x else 0
    assert row['final_hold_s']==hold*.02
    mapping=read(out/'contact_mapping.json');assert len(mapping['frames'])==1025
    assert mapping['validated']==all(x['mapping_valid'] for x in mapping['frames'])
    assert np.max(abs(d['command'][:,:,:23]-d['previous'][:,:,:23]))<=.0030002
    assert np.array_equal(d['command'][:,:,23:],d['previous'][:,:,23:])
    meta=read(ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/metadata.json')
    assert (d['command'][:,:,:23]>=np.array(meta['lower'])+.03-2e-7).all()
    assert (d['command'][:,:,:23]<=np.array(meta['upper'])-.03+2e-7).all()
    expected=d['previous'][:,:,:23]+np.clip(.1*(np.clip(d['desired'],np.array(meta['lower'])+.03,np.array(meta['upper'])-.03)-d['previous'][:,:,:23]),-.003,.003)
    assert np.max(abs(expected-d['command'][:,:,:23]))<=2e-7
    fn=(d['physical_force_w']*d['physical_normal_w']).sum(-1).clip(min=0)
    ids=[0] if manifest['profile']=='rear_pelvis' else [3,4]
    load=(fn[:,:,ids]*d['physical_inside_patch'][:,:,ids]).sum(-1)/d['weight']
    effort=(d['physical_torque']/d['physical_torque_scale'])**2
    power=abs(d['physical_torque']*d['physical_joint_velocity']).sum(-1)
    admitted=bool(row['clean'] and mapping['validated'] and (load[-250:]>=.05).all())
    result=dict(passed_integrity=True,admitted=admitted,row=row,profile=manifest['profile'],variant=manifest['variant'],
        mapping_validated=mapping['validated'],minimum_final_intended_load_bw=float(load[-250:].min()),
        final_normalized_torque_squared=float(effort[-250:].mean()),final_absolute_mechanical_power_W=float(power[-250:].mean()),
        final_mean_knee_angle_rad=float(d['q'][-250:,:,[11,12]].mean()),
        first_failure_s=float(.02*(1+np.flatnonzero(d['failure'][:,0])[0])) if d['failure'].any() else None,
        source_hashes={f:sha(out/f) for f in ['trajectory.npz','results.json','manifest.json','startup.json','plan.json']},original_auditor_sha256=sha(SRC/'audit.py'))
    write(out/'independent_audit.json',result);return result
if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]))),flush=True)
