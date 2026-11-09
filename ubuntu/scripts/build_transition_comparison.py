"""Build an evidence summary and matched-camera video from completed evaluations."""
from pathlib import Path
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=root/'runs/full_transition_comparison_02'
out.mkdir(exist_ok=True)
a=root/'runs/seated_hold_policy_video_comparison_02'
b=root/'runs/full_transition_policy_video_02'
command=['ffmpeg','-nostdin','-n','-i',str(a/'seated_baseline_0000.mp4'),'-i',str(b/'transition_0000.mp4'),'-filter_complex',
"[0:v]drawtext=text='Preserved policy - starts seated':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[a];[1:v]drawtext=text='New residual PPO - starts standing':x=20:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[b];[a][b]hstack=inputs=2[v]",'-map','[v]','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(out/'comparison.mp4')]
subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for folder,label in [(a,'Preserved seated policy'),(b,'Standing-to-sitting PPO')]:
    with np.load(folder/'rollouts.npz') as z:
        t=np.arange(len(z['root_pose']))*float(z['dt'])
        for ax,key,extract in [(axes[0],'root_pose',lambda x:x[:,0,2]),(axes[1],'seat_force_peak',lambda x:x[:,0]),(axes[2],'hold',lambda x:x[:,0])]:
            ax.plot(t,extract(z[key]),label=label)
axes[0].set_ylabel('Pelvis height (m)');axes[1].set_ylabel('Seat normal resultant (N)');axes[2].set_ylabel('Continuous seated hold (s)');axes[2].set_xlabel('Time (s)')
axes[1].axhline(632.5278,color='red',linestyle='--',label='Impact safety limit')
for ax in axes:ax.grid(alpha=.3);ax.legend()
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
folders=['full_transition_task_baseline_01',*[f'full_transition_policy_seed_{s}_eval_01' for s in (7,42,123)],'full_transition_policy_holdout_01','seated_hold_policy_comparison_02']
summary={name:json.loads((root/'runs'/name/'results.json').read_text()) for name in folders}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'video_command.json').write_text(json.dumps(command,indent=2)+'\n')
print(out)
