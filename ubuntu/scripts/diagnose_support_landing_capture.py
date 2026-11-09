"""Explain legacy rejections from recorded raw sensors; labels loaded only below."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from humanoid_sitting.support_landing_geometry import extract_scene,original_measurement
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True);a=p.parse_args()
cap=Path(a.capture);out=Path(a.output);out.mkdir(exist_ok=False)
c=json.loads((cap/'calibration.json').read_text());model=BodyModel(cap/'model.json')
assert model.names==c['joint_names']
data=np.load(cap/'depth.npz');s=np.load(cap/'proprioception.npz');records=[];start=time.perf_counter()
for index,(stamp,case,depth,legacy) in enumerate(zip(data['t'],data['case'],data['depth'],data['legacy'])):
    case=int(case);tick=int(round(stamp/.02))-1;assert abs(s['t'][tick]-stamp)<1e-6
    cal=c['cases'][str(case)];q=s['q'][tick,case];g=s['gravity'][tick,case]
    scene=extract_scene(model,np.array(cal['K']),np.array(cal['T_P_C']),depth,q,g)
    value,info,plane=original_measurement(model,scene,cal['estimated_seat_height_m'])
    np.testing.assert_array_equal(value,legacy)
    reason='accepted' if value[8] else ('plane_extrapolation_unconstrained' if 'allowance' in info else info['basis'])
    records.append(dict(index=index,case=case,t=float(stamp),valid=bool(value[8]),reason=reason,**scene['diagnostic']))
    if index%500==0:print('PROCESSED',index,flush=True)
# Sensor processing is finished. Truth is now used only to score rejection causes.
truth=np.load(cap/'evaluation_labels.npz');roll=np.load(cap/'rollout.npz')
events={i:float(roll['t'][np.flatnonzero(roll['support'][:,i]>.25)[0]]) for i in range(16) if (roll['support'][:,i]>.25).any()}
near=[]
for r in records:
    if r['case'] not in events or not events[r['case']]-1.<=r['t']<=events[r['case']]+.5:continue
    tick=int(round(r['t']/.02))-1
    # Flat rigid surfaces in this capture: known only to this evaluator.
    from humanoid_sitting.shared_frame import level_rotation
    vertices=model.vertices['pelvis']@level_rotation(s['gravity'][tick,r['case']]).T
    height=float(vertices[:,2].min()-truth['true_clearance'][tick,r['case']])
    r['offline_seat_plane_visible']=any(abs(p['height_m']-height)<.015 and p['points']>=60 and p['rms_m']<.008 for p in r['planes'])
    near.append(r)
report=dict(capture=str(cap),frames=len(records),legacy_parity=True,elapsed_s=time.perf_counter()-start,
    reasons=dict(Counter(r['reason'] for r in records)),near_contact_frames=len(near),
    near_contact_reasons=dict(Counter(r['reason'] for r in near)),
    rejected_near_contact_with_visible_seat_plane=sum(not r['valid'] and r['offline_seat_plane_visible'] for r in near),
    support_events=events)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'frames.jsonl').open('w') as f:
    for r in records:f.write(json.dumps(r)+'\n')
print(json.dumps(report,indent=2))
