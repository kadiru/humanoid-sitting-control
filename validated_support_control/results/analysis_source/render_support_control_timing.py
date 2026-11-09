"""Evaluation-label plots for camera timing/control comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--candidate',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True)
root=Path('/data/humanoid-sitting-perception')
runs=[('Historical','support_control_timing_none_02'),('Startup current','support_control_timing_startup_02'),
      ('Descent current','support_control_timing_live_02'),('Both current','support_control_timing_both_02'),
      ('Tracked 10 Hz','support_tracking_control_dev_01'),('Tracked 50 Hz',a.candidate)]
colors=['#748094','#ba8b33','#a269b9','#c95548','#5a9e48','#087f8c']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,axes=plt.subplots(3,1,figsize=(11,10),gridspec_kw={'height_ratios':[1,1.25,1.25]},layout='constrained')
forces=[];labels=[];summary=[]
for i,((name,folder),color) in enumerate(zip(runs,colors)):
    f=root/folder;r=json.loads((f/'results.json').read_text());e=r['episodes'][14]
    labels.append(name+'\n'+str(r['successes'])+'/16 sits');forces.append(e['peak_force_N'])
    c=np.load(f/'controller.npz');d=np.load(f/'evaluation_labels.npz')
    axes[1].plot(d['t'],100*d['true_clearance'][:,14],color=color,label=name,lw=1.7)
    live=c['policy'][:,14,132:142];mask=live[:,8]>.5
    axes[2].plot(c['t'],np.where(mask,live[:,0]*100,np.nan),color=color,label=name,lw=1.8)
    summary.append(dict(name=name,run=folder,successes=r['successes'],case14=e))
axes[0].bar(range(len(runs)),forces,color=colors,width=.66)
axes[0].axhline(632.5278,color='#aa3333',ls='--',lw=1,label='Unchanged impact limit (~633 N)')
axes[0].set_xticks(range(len(runs)),labels);axes[0].set_ylabel('Case 14 peak seat force (N)')
for i,v in enumerate(forces):axes[0].text(i,v+12,f'{v:.0f}',ha='center')
axes[0].set_ylim(0,max(forces)*1.22);axes[0].legend(loc='upper left',frameon=False)
axes[1].set_title('Actual pelvis-to-seat clearance — simulator truth used only for evaluation',loc='left',fontsize=11)
axes[1].set_ylabel('Clearance (cm)');axes[1].set_xlim(10,13.2);axes[1].set_ylim(-.6,8)
axes[1].axhline(0,color='#555555',ls=':',lw=1)
axes[2].set_title('Seat clearance actually supplied to the actor — gaps mean unavailable',loc='left',fontsize=11)
axes[2].set_ylabel('Input clearance (cm)');axes[2].set_xlim(10,13.2);axes[2].set_ylim(-.6,8)
axes[2].set_xlabel('Simulation time (s)');axes[2].legend(ncol=3,frameon=False,fontsize=10)
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.suptitle('Camera timing regression and tracked controller input\nIdentical starts • fixed weights and motor limits • full 20-second trials',fontsize=15)
fig.savefig(out/'timing_control_comparison.png',dpi=160)
(out/'plot_data.json').write_text(json.dumps(summary,indent=2)+'\n')
print(out/'timing_control_comparison.png')
