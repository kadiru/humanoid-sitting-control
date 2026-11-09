"""Rebuild both geometry contracts from sensors; compare/score only afterwards."""
import argparse
import json
from pathlib import Path
import numpy as np
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
from humanoid_sitting.support_continuous_features import ContinuousPatchFeatures
from humanoid_sitting.support_split_observation import distance_features
from seat_depth_estimator import estimate_body_frame

p=argparse.ArgumentParser(); p.add_argument('--capture',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); cap=Path(a.capture); out=Path(a.output); out.mkdir(exist_ok=False)
def load(name):
    with np.load(cap/name) as f:return {k:f[k] for k in f.files}
cal=json.loads((cap/'calibration.json').read_text()); sensors=load('proprioception.npz')
depth=load('depth.npz'); startup=load('startup.npz')
lookup={(int(c),round(t/.02)):i for i,(c,t) in enumerate(zip(depth['case'],depth['t']))}
T,N=sensors['q'].shape[:2]
patch=np.zeros((T,N,10),np.float32); patch[:,:,-1]=1.
distance=np.tile(distance_features({},0),(T,N,1))
accepted=np.zeros((T,N),bool)
for case in range(N):
    if str(case) not in cal['cases']:continue
    c=cal['cases'][str(case)]
    tracker=LandingGeometryTracker(cap/'model.json',c['K'],c['T_P_C'],c['estimated_seat_height_m'])
    idx=int(np.flatnonzero(startup['case']==case)[0]);g=startup['gravity'][idx]
    floor=estimate_body_frame(startup['depth'][idx],startup['K'][idx],
        np.array(cal['startup_T_P_C']['mount']),np.array(cal['startup_T_P_C']['quaternion_xyzw']),g)['floor_height_body_m']
    tracker.initialize_from_measured_floor(.02,startup['q'][idx],g,floor)
    features=ContinuousPatchFeatures(tracker)
    for tick,t in enumerate(sensors['t']):
        q=sensors['q'][tick,case];g=sensors['gravity'][tick,case];w=sensors['gyro'][tick,case]
        index=lookup.get((case,round(t/.02))); image=depth['depth'][index] if index is not None else None
        result=tracker.update(t,q,g,w,image)
        patch[tick,case],_=features.update(t,result,q,g,w,image)
        distance[tick,case]=distance_features(result,t); accepted[tick,case]=result['valid']
    print('SENSOR_REPLAY_CASE',case,flush=True)
np.savez_compressed(out/'reconstructed.npz',t=sensors['t'],patch=patch,distance=distance)
# Nothing above opens labels or controller packets.
recorded=load('controller.npz'); labels=load('evaluation_labels.npz')
expected=recorded['policy'][1:,:,132:142]
match=bool(np.array_equal(expected,patch[:-1]))
new_match=(bool(np.array_equal(recorded['policy'][1:,:,144:],distance[:-1]))
           if recorded['policy'].shape[-1]==151 else None)
# Older captures label the post-step sensor frame; the new entry point labels
# the actual pre-command packet. Join timestamps instead of assuming an offset.
label_index={round(float(t)/.02):i for i,t in enumerate(labels['t'])}
joined=[(i,label_index[round(float(t)/.02)]) for i,t in enumerate(sensors['t'])
        if round(float(t)/.02) in label_index]
si,li=np.array(joined).T
scored_distance=distance[si];scored_patch=patch[si]
valid=scored_distance[:,:,4]>0;error=abs(scored_distance[:,:,0]-labels['true_clearance'][li])
before=np.ones_like(valid)
for case in range(N):
    failed=np.flatnonzero(labels['failure'][li,case])
    if len(failed):before[failed[0]:,case]=False
report=dict(capture=str(cap),sensor_only_reconstruction=True,labels_opened_after_reconstruction=True,
    old_patch_bitwise_equal=match,tracker_accepted_frames=int(accepted.sum()),
    new_distance_bitwise_equal=new_match,
    accepted_distance_lost_by_interface=int((accepted & (distance[:,:,4]==0)).sum()),
    recovered_distance_when_patch_unavailable=int((valid & (scored_patch[:,:,8]==0)).sum()),
    all_frame_distance_availability=float(valid.mean()),patch_availability=float((scored_patch[:,:,8]>0).mean()),
    before_failure_distance_availability=float(valid[before].mean()),
    max_accepted_distance_error_m=float(error[valid].max()),
    before_failure_max_error_m=float(error[valid & before].max()))
report['passed']=bool(match and new_match is not False and report['accepted_distance_lost_by_interface']==0 and
    report['before_failure_distance_availability']>=.99 and report['before_failure_max_error_m']<.02)
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert report['passed'],report
