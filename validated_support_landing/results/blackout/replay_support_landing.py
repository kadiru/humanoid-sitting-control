"""Sensor-only replay. This program does not open simulator-label archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
from seat_depth_estimator import estimate_body_frame
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True)
p.add_argument('--fault',choices=['none','noise','blackout'],default='none');a=p.parse_args()
cap=Path(a.capture);out=Path(a.output);out.mkdir(exist_ok=False)
def load(path):
    with np.load(path) as archive:return {k:archive[k] for k in archive.files}
cal=json.loads((cap/'calibration.json').read_text());s=load(cap/'proprioception.npz');d=load(cap/'depth.npz');startup=load(cap/'startup.npz')
lookup={(int(case),int(round(t/.02))):i for i,(case,t) in enumerate(zip(d['case'],d['t']))}
N=s['q'].shape[1];T=len(s['t']);result={k:np.full((T,N),np.nan,dtype=np.float32) for k in ['clearance','allowance','measurement_age','foot_disagreement']}
result['valid']=np.zeros((T,N),bool);result['source']=np.zeros((T,N),np.int16)
sources=['unavailable','current_floor_and_seat','depth_with_kinematic_floor','kinematic_prediction','depth_with_temporal_identity','seat_from_observed_floor'];times=[];initial=[]
rng=np.random.default_rng(51899)
mount=np.array(cal['startup_T_P_C']['mount']);quat=np.array(cal['startup_T_P_C']['quaternion_xyzw'])
with (out/'records.jsonl').open('w') as f:
 for case in range(N):
    if str(case) not in cal['cases']:continue
    c=cal['cases'][str(case)];obs=LandingGeometryTracker(cap/'model.json',c['K'],c['T_P_C'],c['estimated_seat_height_m'])
    index=int(np.flatnonzero(startup['case']==case)[0])
    measured=estimate_body_frame(startup['depth'][index],startup['K'][index],mount,quat,startup['gravity'][index])
    initialized=obs.initialize_from_measured_floor(.02,startup['q'][index],startup['gravity'][index],measured['floor_height_body_m'])
    initial.append(dict(case=case,initialized=initialized,measured_floor_height=measured['floor_height_body_m']))
    for tick,stamp in enumerate(s['t']):
        q=s['q'][tick,case].copy();gravity=s['gravity'][tick,case].copy();gyro=s['gyro'][tick,case].copy()
        frame=lookup.get((case,tick+1));depth=d['depth'][frame].copy() if frame is not None else None
        if a.fault=='noise':
            q+=rng.normal(0,.001,q.shape);gravity+=rng.normal(0,.001,3);gravity/=np.linalg.norm(gravity)
            gyro+=rng.normal(0,.005,3)
            if depth is not None:depth+=rng.normal(0,.002,depth.shape)
        if a.fault=='blackout' and 11.8-1e-6<=stamp<=12.4+1e-6 and depth is not None:depth[:]=np.nan
        before=time.perf_counter();r=obs.update(stamp,q,gravity,gyro,depth);times.append(time.perf_counter()-before)
        result['valid'][tick,case]=r['valid'];result['source'][tick,case]=sources.index(r['source'])
        for field,key in [('clearance','clearance_m'),('allowance','allowance_m'),('measurement_age','measurement_age_s'),('foot_disagreement','foot_disagreement_m')]:
            result[field][tick,case]=r.get(key) if r.get(key) is not None else np.nan
        f.write(json.dumps(dict(case=case,**r))+'\n')
    print('REPLAYED',case,flush=True)
np.savez_compressed(out/'tracking.npz',t=s['t'],**result)
module=Path(__file__).resolve().parents[1]/'sitting_task/src/humanoid_sitting/support_landing_geometry.py'
for path in [Path(__file__).resolve(),module]:shutil.copy2(path,out/path.name)
manifest=dict(capture=str(cap),fault=a.fault,initial=initial,sources=sources,
    inputs=['proprioception.npz','depth.npz','startup.npz','calibration.json','model.json'],
    sensor_only=True,tracker_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    update_median_ms=float(np.median(times)*1000),update_p95_ms=float(np.quantile(times,.95)*1000),
    total_processing_s=float(sum(times)))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
