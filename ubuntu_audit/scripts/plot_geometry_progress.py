"""Export the paired development/held-out safety comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'runs/geometry_reference_progress_01';rows={x['run']:x for x in json.loads((out/'metrics.json').read_text())}
sets=[['development_scripted','holdout_scripted','confirmation_scripted'],['development_parent','holdout_ppo','confirmation_ppo'],['development_pilot','holdout_pilot','confirmation_pilot']]
labels=['Geometry scripted','Preserved PPO residual','80-update pilot residual'];x=np.arange(3);width=.25
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for j,(keys,label) in enumerate(zip(sets,labels)):
 data=[rows['geometry_reference_v12_'+k+'_01'] for k in keys];positions=x+(j-1)*width
 values=[[100*d['successes']/d['trials'] for d in data],[d['peak_force_N'] for d in data],[d['minimum_hold_s'] for d in data]]
 for ax,v in zip(axes,values):ax.bar(positions,v,width,label=label)
 for pos,d in zip(positions,data):axes[0].text(pos,100*d['successes']/d['trials']+1,f"{d['successes']}/144",ha='center',fontsize=8,rotation=90)
axes[0].set_ylabel('Whole-trial success (%)');axes[0].set_ylim(0,117)
axes[1].set_ylabel('Maximum seat force (N)');axes[1].axhline(632.527832,color='red',linestyle='--',label='Unchanged force limit');axes[1].legend(fontsize=8)
axes[2].set_ylabel('Minimum continuous seated hold (s)');axes[2].axhline(3,color='red',linestyle='--')
for ax in axes:ax.set_xticks(x,['Development','Held out A','Held out B']);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Identical initial states within each bank; only scripted control passes every bank without violations')
fig.tight_layout();fig.savefig(out/'safety_comparison.png',dpi=160)
