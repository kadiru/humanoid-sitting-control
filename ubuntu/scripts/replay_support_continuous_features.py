"""Reconstruct controller geometry from archived robot sensors, then compare."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
from humanoid_sitting.support_continuous_features import ContinuousPatchFeatures
from seat_depth_estimator import estimate_body_frame
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True);a=p.parse_args()
cap=Path(a.capture);out=Path(a.output);out.mkdir(exist_ok=False)
def load(name):
    with np.load(cap/name) as f:return {k:f[k] for k in f.files}
cal=json.loads((cap/'calibration.json').read_text());s=load('proprioception.npz');d=load('depth.npz');startup=load('startup.npz')
lookup={(int(c),round(t/.02)):i for i,(c,t) in enumerate(zip(d['case'],d['t']))}
T,N=s['q'].shape[:2];values=np.zeros((T,N,10),np.float32);values[:,:,-1]=1.
mount=np.array(cal['startup_T_P_C']['mount']);quat=np.array(cal['startup_T_P_C']['quaternion_xyzw'])
for case in range(N):
    if str(case) not in cal['cases']:continue
    c=cal['cases'][str(case)];tracker=LandingGeometryTracker(cap/'model.json',c['K'],c['T_P_C'],c['estimated_seat_height_m'])
    idx=int(np.flatnonzero(startup['case']==case)[0]);g=startup['gravity'][idx]
    floor=estimate_body_frame(startup['depth'][idx],startup['K'][idx],mount,quat,g)['floor_height_body_m']
    tracker.initialize_from_measured_floor(.02,startup['q'][idx],g,floor)
    features=ContinuousPatchFeatures(tracker)
    for tick,t in enumerate(s['t']):
        q=s['q'][tick,case];g=s['gravity'][tick,case];w=s['gyro'][tick,case];i=lookup.get((case,tick+1))
        depth=d['depth'][i] if i is not None else None
        r=tracker.update(t,q,g,w,depth);v,_=features.update(t,r,q,g,w,depth);values[tick,case]=v
    print('REPLAYED',case,flush=True)
np.savez_compressed(out/'features.npz',t=s['t'],features=values)
# Comparison happens only after reconstruction; no controller packet is an input.
with np.load(cap/'controller.npz') as recorded:
    expected=recorded['policy'][1:,:,132:142];actual=values[:-1]
    comparison=dict(bitwise_equal=bool(np.array_equal(expected,actual)),max_abs_error=float(abs(expected-actual).max()),
                    actor_packets_compared=int(np.prod(expected.shape[:2])),last_unused_frame_excluded=True)
root=Path(__file__).resolve().parents[1]
paths=[Path(__file__).resolve(),root/'sitting_task/src/humanoid_sitting/support_landing_geometry.py',
       root/'sitting_task/src/humanoid_sitting/support_continuous_features.py']
hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
for path in paths:shutil.copy2(path,out/path.name)
report=dict(capture=str(cap),sensor_inputs=['proprioception.npz','depth.npz','startup.npz','calibration.json','model.json'],
            evaluation_labels_loaded=False,training=False,hashes=hashes,comparison=comparison)
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert comparison['bitwise_equal'],comparison
