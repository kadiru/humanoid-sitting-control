import argparse,json
from pathlib import Path
import numpy as np
from seat_depth_estimator import estimate
from scipy.spatial.transform import Rotation
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--depth-noise-mm',type=float,default=0.);p.add_argument('--dropout-fraction',type=float,default=0.);a=p.parse_args();root=Path(a.dataset);manifest=json.loads((root/'manifest.json').read_text());outcomes=[]
fig,axes=plt.subplots(4,4,figsize=(14,11))
for j,r in enumerate(manifest['records']):
    z=np.load(root/r['file']);g=r['ground_truth'];record={k:v for k,v in r.items() if k!='ground_truth'}
    try:
        depth=z['depth'].copy();rng=np.random.default_rng(27000+j)
        if a.depth_noise_mm:depth+=rng.normal(0,a.depth_noise_mm/1000,depth.shape).astype(depth.dtype)
        if a.dropout_fraction:depth[rng.random(depth.shape)<a.dropout_fraction]=np.inf
        est=estimate(depth,z['K'],z['camera_position'],z['camera_quaternion']);record['prediction']=est
        back=.06 if g['family'] in ('chair','rigid_couch') else 0.
        center=np.array([g['x'],g['y'],g['height']]);center[:2]+=back/2*np.array([np.cos(g['yaw']),np.sin(g['yaw'])]);depth=g['depth']-back
        error_yaw=abs((est['yaw']-g['yaw']+np.pi/2)%np.pi-np.pi/2)
        record.update(detected=True,direction_resolved=est['direction_resolved'],heading_error_deg=float(np.rad2deg(abs((est['heading']-g['yaw']+np.pi)%(2*np.pi)-np.pi))) if g['family'] in ('chair','rigid_couch') and est['direction_resolved'] else None,height_error_m=abs(est['surface_center'][2]-g['height']),surface_position_error_m=float(np.linalg.norm(np.array(est['surface_center'])-center)),depth_error_m=abs(est['depth']-depth),width_error_m=abs(est['width']-g['width']),orientation_error_deg=float(np.rad2deg(error_yaw)),area_relative_error=abs(est['available_area_m2']-depth*g['width'])/(depth*g['width']))
    except Exception as e:record.update(detected=False,error=str(e))
    outcomes.append(record)
    if j%2==0 and j//2<16:
        ax=axes.flat[j//2];ax.imshow(z['rgb'][...,:3]);ax.axis('off');ax.set_title(g['family']+' '+('yaw %.1f° area %.0f%%'%(record['orientation_error_deg'],100*record['area_relative_error']) if record['detected'] else 'NO DETECTION'),fontsize=9)
valid=[r for r in outcomes if r['detected']];metrics={k:dict(mean=float(np.mean([r[k] for r in valid])),max=float(np.max([r[k] for r in valid]))) for k in ('height_error_m','surface_position_error_m','depth_error_m','width_error_m','orientation_error_deg','area_relative_error')} if valid else {}
directional=[r['heading_error_deg'] for r in valid if r['heading_error_deg'] is not None]
result=dict(backrest_heading_resolved_frames=len(directional),heading_error_deg_mean=float(np.mean(directional)) if directional else None,heading_error_deg_max=float(np.max(directional)) if directional else None,depth_noise_mm=a.depth_noise_mm,dropout_fraction=a.dropout_fraction,num_frames=len(outcomes),detected_frames=len(valid),metrics=metrics,outcomes=outcomes,limitations=['Ideal noiseless renderer depth; calibrated external camera; no robot occlusion/clutter','Estimates visible free top surface, not hidden full dimensions','Width exceeds depth; symmetric surfaces have yaw ambiguity modulo pi; backrest direction is resolved only with sufficient visible evidence','Not integrated into sitting policy; no approach/alignment claim'])
suffix='_noisy' if a.depth_noise_mm or a.dropout_fraction else ''
(root/('evaluation'+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n');plt.tight_layout();plt.savefig(root/('representative_rgb'+suffix+'.png'));plt.close();print(json.dumps({k:v for k,v in result.items() if k!='outcomes'},indent=2))
