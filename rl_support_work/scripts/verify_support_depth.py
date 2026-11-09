"""Offline regression of the fast current-scan observer on saved sensor data.

Truth is opened ONLY by this evaluator; the observer gets depth/calibration,
encoders, gravity, and the floor-to-seat height measured in the initial scan.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from humanoid_sitting.support_depth import CurrentSupportDepth
from seat_depth_estimator import estimate_body_frame
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]
capture=root/'runs/shared_frame_rear_down_six_capture_01'
with np.load(capture/'sensors.npz') as archive:
    s={k:archive[k] for k in ('depth','K','T_P_C','q','gravity')}
truth=json.loads((root/'runs/shared_frame_reference_rear_clean_03/errors.json').read_text())
lookup={(r['case'],r['frame']):r['truth_clearance_m'] for r in truth}
T=s['T_P_C']; records=[]; blackout=[]
for case in range(s['depth'].shape[1]):
    initial=estimate_body_frame(s['depth'][0,case],s['K'][case],T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(),s['gravity'][0,case])
    observer=CurrentSupportDepth(root/'runs/g1_kinematics_geometry_04/model.json',s['K'][case],T,initial['surface_center'][2])
    for frame in range(0,len(s['depth']),10):
        values,info=observer.read(s['depth'][frame,case],s['q'][frame,case],s['gravity'][frame,case])
        records.append(dict(case=case,frame=frame,valid=bool(values[8]),
            error_cm=100*abs(float(values[0])-lookup[case,frame]) if values[8] else None))
    values,_=observer.read(np.full_like(s['depth'][0,case],np.nan),s['q'][0,case],s['gravity'][0,case])
    blackout.append(bool(values[8]))
errors=[r['error_cm'] for r in records if r['valid']]
result=dict(samples=len(records),accepted=len(errors),max_error_cm=max(errors,default=None),
    errors_above_2cm=sum(e>2 for e in errors),blackout_accepted=sum(blackout),records=records)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
assert errors and result['errors_above_2cm']==0 and not any(blackout)
