"""Repeat body-frame estimation with depth noise/dropout and spatial occlusion.

Ground truth is used only in this evaluation, never by the estimator or actor.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from seat_depth_estimator import estimate_body_frame
p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True);a=p.parse_args();capture=Path(a.capture);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);z=np.load(capture/'rgbd.npz');manifest=json.loads((capture/'manifest.json').read_text());fixture=Path(manifest['fixture']);r=json.loads((fixture/'results.json').read_text());rng=np.random.default_rng(54001);rows=[]
for mode in ('clean','depth_noise_5mm_dropout_10pct','center_occlusion_20pct','center_occlusion_40pct','left_occlusion_50pct'):
 estimates=[];he=[];short=[]
 for i,original in enumerate(z['depth']):
  d=original.copy().squeeze();h,w=d.shape
  if mode=='depth_noise_5mm_dropout_10pct':
   d+=rng.normal(0,.005,d.shape);d[rng.random(d.shape)<.1]=np.inf
  elif mode.startswith('center'):
   fraction=.2 if '20' in mode else .4;side=np.sqrt(fraction);hh=int(h*side);ww=int(w*side);d[(h-hh)//2:(h+hh)//2,(w-ww)//2:(w+ww)//2]=np.inf
  elif mode.startswith('left'):d[:,:w//2]=np.inf
  try:e=dict(valid=True,**estimate_body_frame(d,z['K'][i],z['mount_position'],z['mount_quaternion'],z['imu_gravity'][i]))
  except Exception as error:e=dict(valid=False,error=str(error))
  estimates.append(e)
  if e['valid']:
   g=r['geometry_bank'][r['outcomes'][i]['geometry_index']];he.append(abs(e['surface_center'][2]-g['height']));short.append(g['depth']*g['width']-e['available_area_m2'])
 folder=out/mode;folder.mkdir();(folder/'estimates.json').write_text(json.dumps(estimates,indent=2)+'\n');(folder/'manifest.json').write_text(json.dumps(dict(manifest,perturbation=mode,source_capture=str(capture),ground_truth_fallback=False),indent=2)+'\n')
 rows.append(dict(mode=mode,frames=len(estimates),valid=sum(e['valid'] for e in estimates),height_error_mean_m=float(np.mean(he)) if he else None,height_error_max_m=float(max(he)) if he else None,mean_full_area_minus_visible_area_m2=float(np.mean(short)) if short else None,extent_note='Visible support area may underestimate full bounds; occlusion confidence is not calibrated.'))
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
