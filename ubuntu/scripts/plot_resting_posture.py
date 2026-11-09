"""Render measured resting trajectories, not a synthetic performance claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from humanoid_sitting.shared_frame import BodyModel

p=argparse.ArgumentParser(); p.add_argument('run'); a=p.parse_args(); run=Path(a.run)
capture=(run/'capture.json').exists()
meta=json.loads((run/('capture.json' if capture else 'results.json')).read_text())
trace=np.load(run/'trace.npz'); geo=json.loads((run/'geometry_bank.json').read_text())
names=meta['body_names']; origins=np.array(meta['origins']); n=len(origins)
model=BodyModel(run/'source/model.json')
selected=np.linspace(0,n-1,min(n,4),dtype=int)
fig,axs=plt.subplots(2,len(selected),figsize=(4*len(selected),7),squeeze=False)
for col,i in enumerate(selected):
    g=geo[min(len(geo)-1,int((i+.5)*len(geo)/n))]; ax=axs[0,col]
    ax.add_patch(plt.Rectangle((g['x']-g['depth']/2,g['height']-.1),g['depth'],.1,color='#98b9d0'))
    ax.axhline(0,color='#333333',lw=1)
    body=trace['body'][-1,i,:,:3]-origins[i]
    for name,vertices in model.vertices.items():
        if name not in names or len(vertices)<4:continue
        pose=trace['body'][-1,i,names.index(name)]
        points=vertices@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]-origins[i]
        xz=points[:,[0,2]]
        if np.linalg.matrix_rank(xz-xz.mean(0))==2:
            hull=ConvexHull(xz); boundary=xz[hull.vertices]
            ax.fill(boundary[:,0],boundary[:,1],color='#2a6d87',alpha=.22)
    chains=[['pelvis','torso_link'],
        ['pelvis','left_hip_yaw_link','left_knee_link','left_ankle_roll_link'],
        ['pelvis','right_hip_yaw_link','right_knee_link','right_ankle_roll_link'],
        ['torso_link','left_shoulder_roll_link','left_elbow_pitch_link'],
        ['torso_link','right_shoulder_roll_link','right_elbow_pitch_link']]
    for chain in chains:
        ids=[names.index(x) for x in chain if x in names]
        if len(ids)>1:ax.plot(body[ids,0],body[ids,2],'-o',ms=4,color='#16546f',lw=2)
    ax.set(xlim=(-.85,.4),ylim=(-.05,1.3),title=f"Seat {g['height']*100:.0f} cm · case {i}",xlabel='Forward position (m)',ylabel='Height (m)')
    ax.set_aspect('equal'); ax.grid(alpha=.2)
    ax=axs[1,col]
    if capture:
        ax.plot(trace['t'],trace['support'][:,i],label='Rest contact / body weight')
        ax.plot(trace['t'],trace['good'][:,i].astype(float),label='Qualifying rest',alpha=.7)
    else:
        ax.plot(trace['t'],trace['effort'][:,i],label='Whole-body effort proxy')
        ax.plot(trace['t'],trace['good'][:,i].astype(float),label='Qualifying rest',alpha=.6)
    ax.set(xlabel='Time (s)',ylabel='Dimensionless'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('Recorded final posture and measured support' if capture else 'Fixed-posture / learned resting evaluation',fontsize=15)
fig.tight_layout(); fig.savefig(run/'postures.png',dpi=170); plt.close(fig)
