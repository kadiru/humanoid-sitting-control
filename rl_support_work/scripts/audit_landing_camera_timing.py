"""Sensor-only timing audit: fitted horizontal planes versus IMU history."""
import argparse,json
from pathlib import Path
import numpy as np
from humanoid_sitting.support_landing_geometry import extract_scene
from humanoid_sitting.shared_frame import BodyModel
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True);a=p.parse_args()
cap=Path(a.capture);out=Path(a.output);out.mkdir(exist_ok=False)
def load(path):
 with np.load(path) as f:return {k:f[k] for k in f.files}
s=load(cap/'proprioception.npz');d=load(cap/'depth.npz');cal=json.loads((cap/'calibration.json').read_text());model=BodyModel(cap/'model.json');rows=[]
for j,(stamp,case) in enumerate(zip(d['t'],d['case'])):
 i=int(case);k=int(round(stamp/.02))-1
 if not (8.<stamp<16.) or np.linalg.norm(s['gyro'][k,i])<.3:continue
 c=cal['cases'][str(i)];scene=extract_scene(model,np.array(c['K']),np.array(c['T_P_C']),d['depth'][j],s['q'][k,i],s['gravity'][k,i])
 planes=[x for x in scene['planes'] if len(x['points'])>=150 and x['rms']<.008]
 if not planes:continue
 plane=max(planes,key=lambda x:len(x['points']));n=scene['R'].T@plane['normal']
 angles=[float(np.degrees(np.arccos(np.clip(n@(-s['gravity'][k-lag,i]/np.linalg.norm(s['gravity'][k-lag,i])),-1,1)))) for lag in range(9)]
 rows.append(dict(case=i,t=float(stamp),angles_deg=angles,best_lag_s=int(np.argmin(angles))*.02))
angles=np.array([r['angles_deg'] for r in rows]);summary=dict(capture=str(cap),sensor_only=True,samples=len(rows),
 median_angle_by_lag_deg=np.median(angles,axis=0).tolist(),current_angle_p95_deg=float(np.quantile(angles[:,0],.95)),
 current_angle_max_deg=float(angles[:,0].max()),median_best_lag_s=float(np.median([r['best_lag_s'] for r in rows])))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'frames.json').write_text(json.dumps(rows)+'\n');print(json.dumps(summary,indent=2))
