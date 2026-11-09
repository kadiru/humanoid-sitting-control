"""Post-process an entire trial without importing the controller or simulator."""
import json,hashlib,sys,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting')
SRC=ROOT/'experiments/full_body_contact_trainer_4090_step_v5'
sys.path.insert(0,str(SRC))
from common import read,write,sha
spec=importlib.util.spec_from_file_location('original_physical_checks',SRC/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def check(out):
    with np.load(out/'trajectory.npz') as z:d=dict(z)
    assert all(v.shape[:2]==(1000,1) and np.isfinite(v).all() for v in d.values())
    assert not read(out/'startup.json')['warm_failure'][0]
    row,=read(out/'results.json')['episodes']
    assert row['full_standing_start'] and row['steps']==1000 and row['duration_s']==20
    res=audit.physical(d)
    for k in ['failure','good','no_slip','alive','legal']:np.testing.assert_array_equal(res[k],d[k])
    assert row['clean']==bool(res['clean'][0])
    assert row['slip_ticks']==int((~res['no_slip']).sum()) and row['illegal_ticks']==int((~res['legal']).sum())
    hold=0
    for good in res['good'][:,0]:hold=hold+1 if good else 0
    assert row['final_hold_s']==hold*.02
    mapping=read(out/'contact_mapping.json')
    assert len(mapping['frames'])==1025 and mapping['validated']==all(f['mapping_valid'] for f in mapping['frames'])
    force=d['physical_force_w'];fn=(force*d['physical_normal_w']).sum(-1)
    inside=d['physical_inside_patch']
    torso=(fn[:,:,3:5].clip(min=0)*inside[:,:,3:5]).sum(-1)/d['weight']
    ok=bool(row['clean'] and mapping['validated'] and (torso[-250:]>=.05).all())
    assert np.max(np.abs(np.diff(d['command'][:,:,:23],axis=0)))<=.0030002
    assert (d['command'][:,:,23:]==d['command'][0:1,:,23:]).all()
    info=dict(qualifying_lean=ok,row=row,mapping_validated=mapping['validated'],minimum_final_torso_bw=float(torso[-250:].min()),
        maximum_torso_bw=float(torso.max()),first_failure_s=float(.02*(1+np.flatnonzero(res['failure'])[0])) if res['failure'].any() else None,
        original_auditor_sha256=sha(SRC/'audit.py'),source_hashes={f:sha(out/f) for f in ['trajectory.npz','results.json','contact_mapping.json','startup.json','sensor_control.npy','plan.json']})
    write(out/'independent_audit.json',info);print(json.dumps(info),flush=True);return ok
if __name__=='__main__':check(Path(sys.argv[1]))
