"""Final runtime contract, sensor isolation, live replay, and failure checks."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
read=lambda name:json.loads((root/name).read_text())
names=['shared_frame_quality_clean_04','shared_frame_quality_noisy_04','shared_frame_quality_occlusion_03',
       'shared_frame_stress_04','shared_frame_holdout_02']
source_hash=hashlib.sha256((root/'sitting_task/src/humanoid_sitting/shared_frame.py').read_bytes()).hexdigest()
reports={}
for name in names:
    rows=read('runs/'+name+'/estimates.json')
    manifest=read('runs/'+name+'/manifest.json')
    assert manifest['source_sha256']['shared_frame.py']==source_hash, name+' has an obsolete runtime'
    assert all(r['valid']==r['quality']['alignment_valid'] and 'legacy_valid' in r for r in rows)
    for r in rows:
        assert r['quality']['quantities']['clearance']['valid']==(r['direct_clearance_m'] is not None)
        if not r['registration']['accepted']:
            assert not r['valid']
        if r.get('seat_age_s',0)>=1.:
            assert not r['valid']
    reports[name]=dict(samples=len(rows), source_hash_matches=True,
        alignment_valid=sum(r['valid'] for r in rows),
        clearance_measured=sum(r['quality']['quantities']['clearance']['valid'] for r in rows))

clean=read('runs/shared_frame_quality_clean_04/estimates.json')
isolated=read('runs/shared_frame_quality_sensor_only_02/estimates.json')
assert isolated==clean[:len(isolated)]
online=read('runs/shared_frame_quality_live_02/online_estimates.json')
offline=read('runs/shared_frame_quality_live_replay_02/estimates.json')
assert len(online)==len(offline)==161
differences={}
for field in ('T_O_P','T_S_P','direct_clearance_m'):
    differences[field]=max(float(np.max(np.abs(np.asarray(a[field])-b[field]))) for a,b in zip(online,offline))
    assert differences[field]<1e-6
def equivalent(a,b):
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return bool(np.isclose(a,b,atol=1e-7,rtol=1e-7))
    return a==b
assert all(equivalent(a['quality'],b['quality']) and a['valid']==b['valid'] for a,b in zip(online,offline))

occlusion=read('runs/shared_frame_quality_occlusion_03/estimates.json')
blind=[r for r in occlusion if 6.<=r['stamp']<6.5 or 8.<=r['stamp']<10.]
assert len(blind)==150
assert all(not r['valid'] and not r['quality']['quantities']['clearance']['valid'] for r in blind)
stress=read('runs/shared_frame_stress_04/estimates.json')
recovered=[r for r in stress if r['case']==2 and r['stamp']==4.5][0]
assert recovered['registration'].get('recovery')=='reacquired'
assert recovered['quality']['quantities']['lateral']['valid']
reports['contract']=dict(sensor_only_equal_samples=len(isolated), live_replay_frames=len(online),
    live_replay_max_differences=differences, live_replay_quality_equal_at_1e_7=True,
    all_150_missing_depth_samples_abstain=True, blind_slide_recovery_stamp_s=4.5,
    frozen_runtime_sha256=source_hash)
(root/'runs/shared_frame_quality_contract.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
