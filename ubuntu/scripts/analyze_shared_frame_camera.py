"""Offline paired bench-camera comparison; does not feed the observer."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
out=root/'runs/shared_frame_camera_comparison_02'; out.mkdir(exist_ok=False)
cases=[('Original camera','shared_frame_quality_live_02','shared_frame_quality_live_replay_02'),
       ('Rear vertical camera','shared_frame_rear_down_capture_02','shared_frame_rear_down_02')]
report={}; data={}; roots=[]
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for col,(label,capture,run) in enumerate(cases):
    folder=root/'runs'/run
    rows=json.loads((folder/'estimates.json').read_text()); errors=json.loads((folder/'errors.json').read_text())
    with np.load(root/'runs'/capture/'sensors.npz') as ar:
        rgb=ar['rgb'][80,0]; depth=ar['depth'][80,0]
    with np.load(root/'runs'/capture/'truth.npz') as ar: poses=ar['root_pose'][:,0]
    roots.append(poses); data[label]=(rows,errors)
    q={}
    for key,metric,limit in [('clearance','direct_clearance_error_cm',2.),('forward','forward_offset_error_cm',3.),
                            ('lateral','lateral_offset_error_cm',3.),('yaw','yaw_axis_error_deg',5.)]:
        values=[]
        for r,e in zip(rows,errors):
            if r['quality']['quantities'][key]['valid']:
                values.append(100*abs(r['direct_clearance_m']-e['truth_clearance_m']) if key=='clearance' else e[metric])
        q[key]=dict(available=len(values), p95=float(np.quantile(values,.95)) if values else None,
                    max=max(values,default=None), violations=sum(v>limit for v in values))
    report[label]=dict(samples=len(rows),alignment_valid=sum(r['valid'] for r in rows),quantities=q,
                       displayed_rgb_blank=bool(not np.any(rgb)),
                       root_travel_cm=(100*np.ptp(poses[:,:3],axis=0)).tolist())
    axs[0,col].imshow(np.ma.masked_invalid(depth),cmap='viridis',vmin=.1,vmax=1.6)
    axs[0,col].axis('off');axs[0,col].set_title(label+' — axial depth at t = 8.0 s')
    flags=[[r['quality']['quantities'][key]['valid'] for r in rows] for key in ('clearance','forward','lateral','yaw')]
    axs[1,col].imshow(flags,origin='lower',interpolation='nearest',aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,extent=(0,16,-.5,3.5))
    axs[1,col].set_yticks(range(4),['clearance','forward','lateral','yaw'])
    axs[1,col].set_xlabel('Sensor time (s)');axs[1,col].set_title('Measured: green / unconfirmed: red')
report['trajectory_comparison']=dict(root_position_difference_max_cm=float(100*np.linalg.norm(roots[0][:,:3]-roots[1][:,:3],axis=1).max()),
    root_position_difference_rms_cm=float(100*np.sqrt(np.mean(np.sum((roots[0][:,:3]-roots[1][:,:3])**2,axis=1)))))
online=json.loads((root/'runs/shared_frame_rear_down_capture_02/online_estimates.json').read_text())
offline=data['Rear vertical camera'][0]
report['new_camera_live_replay_max_pose_difference']=max(float(np.max(np.abs(np.asarray(a['T_O_P'])-b['T_O_P']))) for a,b in zip(online,offline))
assert report['new_camera_live_replay_max_pose_difference']<1e-6
assert all(a['valid']==b['valid'] for a,b in zip(online,offline))
report['calibration']=json.loads((root/'runs/shared_frame_rear_down_02/manifest.json').read_text())['T_P_C']
report['scope']='One 16 s physics bench scene. Original camera retained for controller startup geometry; experimental view used only by observer. First camera probe (rear_down_01) changed controller input and is rejected as a camera-only comparison; retained for failure analysis.'
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
fig.suptitle('Paired bench trial: actual depth images (same 0.1–1.6 m color scale)',fontsize=15)
fig.savefig(out/'camera_comparison.png',dpi=150);plt.close(fig)
print(json.dumps(report,indent=2))
