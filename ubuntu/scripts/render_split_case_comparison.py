"""Illustrative positive/negative cases, chosen after the fixed evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--output',required=True)
a=p.parse_args();store=Path(a.store);out=Path(a.output);out.mkdir(exist_ok=False)
fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,layout='constrained')
cases=[('regression',14,'Known case 14: longer hold, higher impact'),
       ('fresh',13,'Fresh case 13: improved weight transfer')]
detail={}
for col,(bank,case,title) in enumerate(cases):
    detail[bank]={}
    for role,color,label in [('parent','#3565a5','Frozen controller'),('trained','#e07a32','After PPO')]:
        run=store/f'support_split_{bank}_{role}_01'
        result=json.loads((run/'results.json').read_text())['episodes'][case]
        detail[bank][role]={k:result[k] for k in ('success','hold_s','peak_force_N','seated_leg_rms_Nm','seated_foot_load_fraction')}
        with np.load(run/'evaluation_labels.npz') as labels:
            axes[0,col].plot(labels['t'],labels['force'][:,case],color=color,linewidth=1.2,label=label)
        with np.load(run/'rollout.npz') as trace:
            axes[1,col].plot(trace['t'],trace['foot'][:,case]*100,color=color,linewidth=1.2)
            axes[2,col].plot(trace['t'],trace['torque'][:,case],color=color,linewidth=1.2)
    axes[0,col].set_title(title,fontsize=12)
    for row,label in enumerate(['Seat force (N)','Weight on feet (%)','Leg torque RMS (Nm)']):
        axes[row,col].set_ylabel(label);axes[row,col].spines[['top','right']].set_visible(False)
        axes[row,col].set_ylim(bottom=0)
    axes[2,col].set_xlabel('Simulation time (s)');axes[2,col].set_xlim(0,20)
axes[0,0].legend(fontsize=9)
fig.suptitle('PPO tradeoffs: one improvement and one regression\nPost-hoc illustrations; all 48 paired cases remain in the main comparison',fontsize=13)
fig.savefig(out/'case_comparison.png',dpi=160);plt.close(fig)
(out/'cases.json').write_text(json.dumps(detail,indent=2)+'\n')
