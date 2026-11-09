"""Export failure/success traces and a compact perception stress visualization."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);root=out.resolve().parents[1]
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
for col,name in enumerate(['deployment_pilot_01','deployment_geometry_fresh_04']):
 r=json.loads((root/'runs'/name/'results.json').read_text());z=np.load(root/'runs'/name/'rollouts.npz');t=np.arange(len(z['root_pose']))*float(z['dt']);success=[i for i,o in enumerate(r['outcomes']) if o['success']];fail=[i for i,o in enumerate(r['outcomes']) if not o['success']];ids=([success[0]] if success else [])+([fail[0]] if fail else [])
 for i in ids:
  label=f"trial {i}: {'pass' if r['outcomes'][i]['success'] else 'fail'}";axes[0,col].plot(t,z['root_pose'][:,i,2],label=label);axes[1,col].plot(t,z['seat_force_peak'][:,i],label=label);axes[2,col].plot(t,z['support'][:,i],label=label)
 axes[1,col].axhline(r['seat_impact_limit_N'],color='red',ls='--',label='unchanged force limit');axes[0,col].set_title(name);axes[0,col].legend();axes[1,col].legend();axes[2,col].set_xlabel('Time (s)')
for row,label in enumerate(['Pelvis height (m)','Seat normal-force peak (N)','Pelvis/thigh load / bodyweight']):
 for ax in axes[row]:ax.set_ylabel(label);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'deployment_trajectories.png',dpi=160);plt.close(fig)
z=np.load(root/'datasets/deploy_rgbd_fresh_01/rgbd.npz');rgb=z['rgb'][0];depth=z['depth'][0].squeeze();fig,axes=plt.subplots(1,3,figsize=(14,5));axes[0].imshow(rgb);axes[0].set_title('Proposed rear body-mounted RGB-D')
axes[1].imshow(np.where(np.isfinite(depth),depth,np.nan),vmin=.3,vmax=2,cmap='viridis');axes[1].set_title('Robot occlusion included')
h,w=depth.shape;occluded=rgb.copy();side=np.sqrt(.2);hh=int(h*side);ww=int(w*side);occluded[(h-hh)//2:(h+hh)//2,(w-ww)//2:(w+ww)//2,:3]=0;axes[2].imshow(occluded);axes[2].set_title('20% central occlusion: command refused')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(out/'perception_occlusion.png',dpi=160);plt.close(fig)
