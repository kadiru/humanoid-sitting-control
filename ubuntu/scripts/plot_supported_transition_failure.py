"""Plot total pelvis tilt (the failure metric), rather than Euler pitch alone."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--verified',required=True);a=p.parse_args()
r,v=Path(a.study),Path(a.verified)
recovery=v/'plot_revision_01';recovery.mkdir(exist_ok=False)
shutil.copy2(v/'control_failure_diagnosis.png',recovery/'original_pitch_plot.png')
report=json.loads((v/'diagnosis/diagnosis.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for seed,color in zip([54211,54217],['#267b91','#d18836']):
    rep=report['seeds'][str(seed)]['replay']
    axes[0].plot([x['updates'] for x in rep],[x['maximum_requested_change_deg'] for x in rep],label=f'Seed {seed}',color=color)
    d=r/f'eval_{seed}_train';tr=np.load(d/'trace.npz');s=np.load(d/'startup.npz')
    body=json.loads((d/'results.json').read_text())['body_names'].index('pelvis')
    rotation=Rotation.from_quat(tr['body'][:,0,body,3:]).as_matrix()
    tilt=np.rad2deg(np.arccos(np.clip(rotation[:,2,2],-1,1)))
    window=tr['t']<=1.2
    axes[1].plot(tr['t'][window],tilt[window],label=f'Seed {seed}',color=color)
    shift=np.rad2deg(abs(tr['command'][:,0]-s['command'][0])).max(1)
    axes[2].plot(tr['t'][window],shift[window],label=f'Seed {seed}',color=color,ls='-' if seed==54211 else '--')
tr=np.load(r/'fixed_train/trace.npz');body=json.loads((r/'fixed_train/results.json').read_text())['body_names'].index('pelvis')
tilt=np.rad2deg(np.arccos(np.clip(Rotation.from_quat(tr['body'][:,0,body,3:]).as_matrix()[:,2,2],-1,1)))
window=tr['t']<=1.2
axes[1].plot(tr['t'][window],tilt[window],color='#718079',ls='--',label='Fixed stance')
axes[2].axhline(0,color='#718079',ls=':',label='Fixed stance')
axes[0].set(xlabel='Completed PPO updates',ylabel='Largest requested target change (degrees)',title='Same observations, changing policy')
axes[1].set(xlabel='Seconds after standing start',ylabel='Total pelvis tilt from upright (degrees)',title='Recorded 25 cm seat trial',xlim=(0,1.2),ylim=(0,110))
axes[1].axhline(55,color='#b34545',ls=':',label='Fall tilt threshold')
axes[2].set(xlabel='Seconds after standing start',ylabel='Largest actual command shift (degrees)',title='Recorded motor targets',xlim=(0,1.2),ylim=(-2,60))
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(v/'control_failure_diagnosis.png',dpi=180)
(recovery/'revision.json').write_text(json.dumps({'scope':'Reporting only; no data or study changes','reason':'Visual review identified that Euler pitch alone understates total tilt when the robot also rolls. Use the total tilt used by the failure criterion. Restrict vertical command range to the displayed first1.2seconds; improve labels.','original_preserved':'original_pitch_plot.png','source':str(Path(__file__).resolve())},indent=2))
