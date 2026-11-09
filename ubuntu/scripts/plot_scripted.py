"""Summarize recorded physics and save an exportable figure and video contact sheet."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v3 as iio
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();out=Path(args.run)
r=json.loads((out/'results.json').read_text());d=np.load(out/'rollouts.npz');dt=float(d['dt']);t=np.arange(len(d['support']))*dt
late=t>=2;force=d['body_forces'][late,0].mean(axis=0)
contacts={n:f.tolist() for n,f in zip(r['body_names'],force) if np.linalg.norm(f)>1}
foot=d['foot_seat'][late,0]
physics=dict(mean_net_contact_force_N=force.sum(axis=0).tolist(),mean_contacts_N=contacts,
             max_foot_seat_force_N=float(np.linalg.norm(foot,axis=-1).max()),
             mean_foot_seat_force_N=float(np.linalg.norm(foot,axis=-1).mean()))
(out/'contact_validation.json').write_text(json.dumps(physics,indent=2)+'\n')
fig,axs=plt.subplots(4,1,figsize=(10,10),sharex=True)
axs[0].plot(t,d['root_pose'][:,0,2]);axs[0].set_ylabel('Pelvis height (m)')
axs[1].plot(t,d['support'][:,0]);axs[1].set_ylabel('Pelvis/thigh seat load / mg')
axs[2].plot(t,np.linalg.norm(d['lin_vel'][:,0],axis=-1),label='linear m/s');axs[2].plot(t,np.linalg.norm(d['ang_vel'][:,0],axis=-1),label='angular rad/s');axs[2].legend();axs[2].set_ylabel('Speed')
axs[3].plot(t,-d['gravity'][:,0,2]);axs[3].set_ylabel('Upright cosine');axs[3].set_xlabel('Time (s)')
for ax in axs:ax.grid(alpha=.3);ax.axvline(2,color='gray',ls='--')
fig.suptitle('Floating-base G1: joint targets only, gravity −9.81 m/s²');fig.tight_layout();fig.savefig(out/'physics.png',dpi=150);plt.close(fig)
vid=list(out.glob('*.mp4'))
if vid:
    frames=list(iio.imiter(vid[0],plugin='FFMPEG'));indices=np.linspace(0,len(frames)-1,6,dtype=int)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,i in zip(axs.flat,indices):ax.imshow(frames[i]);ax.axis('off');ax.set_title(f'{i*dt*2:.1f} s')
    fig.tight_layout();fig.savefig(out/'frames.jpg',dpi=120);plt.close(fig)
print(json.dumps(physics,indent=2))
