"""Offline diagnosis figures; simulator truth is used for evaluation only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/humanoid-sitting-perception/contact_effort_repair_01')
OUT=ROOT/'diagnosis_report_02'
OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for label,folder,case in [('Original corrected reference','sitting_reference_diagnostics',0),('Front-contact prototype (not accepted)','sitting_front_contact_probe_07',0)]:
    p=ROOT/folder;r=read(p/'results.json');c=np.load(p/'contact_diagnostics.npz');a=np.load(p/'trace.npz')
    namespath=p/'contact_diagnostics_names.json'
    names=read(namespath)['body_names'] if namespath.exists() else read(Path('/data/humanoid-sitting-perception/split_support_curriculum_02/old_reference_diagnosis_01/contact_probe_manifest.json'))['body_names']
    t=np.arange(len(c['root']))*.02
    origin=np.array(r['origins'][case]);root=c['root'][:,case,:3]-origin
    axes[0,0].plot(t,root[:,0],label=label)
    axes[0,1].plot(t,root[:,2],label=label)
    ids=[names.index(x+'_hip_roll_link') for x in ('left','right')]
    axes[1,0].plot(t,c['body_forces'][:,case,ids].max(1),label=label)
    axes[1,1].plot(a['t'],a['geometry'][:,case,0],label=label)
for ax,title,y in zip(axes.flat,['Pelvis moves backward before contact','Pelvis height during lowering','Forbidden hip-housing contact','Available measured seat clearance'],['Pelvis x (m; offline truth)','Pelvis z (m; offline truth)','Hip-roll housing force (N)','Sensor-derived clearance (m)']):
    ax.set(xlim=(7,12),title=title,xlabel='Time (s)',ylabel=y);ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8)
axes[1,0].axhline(10,color='black',ls=':',lw=1)
fig.suptitle('Sitting diagnosis: 38 cm seat at −5°\nNeither illustrated motion passes the unchanged physical criteria')
fig.savefig(OUT/'sitting_contact_diagnosis.png',dpi=160)
plt.close(fig)

p=ROOT/'leaning_action_probe_01';rows=read(p/'results.json')['episodes'];profiles=list(dict.fromkeys(x['profile'] for x in rows));baseline=[x for x in rows if x['profile']=='parent'];summary=[]
for profile in profiles:
    subset=[x for x in rows if x['profile']==profile];pairs=[(x,b) for x,b in zip(subset,baseline) if x['success'] and b['success']]
    changes=[]
    for window in ['metrics_whole_trial','metrics_last5s']:
        policy=np.mean([x[window]['total_squared_torque_Nm2'] for x,b in pairs]);ref=np.mean([b[window]['total_squared_torque_Nm2'] for x,b in pairs]);changes.append(100*(policy/ref-1))
    summary.append(dict(profile=profile,successes=sum(x['success'] for x in subset),conditions=len(subset),common_successes=len(pairs),whole_change_pct=changes[0],last5_change_pct=changes[1]))
fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(len(summary));w=.36
ax.bar(x-w/2,[v['whole_change_pct'] for v in summary],w,label='Whole 20 seconds')
ax.bar(x+w/2,[v['last5_change_pct'] for v in summary],w,label='Final 5 seconds')
ax.set(xticks=x,xticklabels=[v['profile']+'\n'+str(v['successes'])+'/9 stable' for v in summary],ylabel='Total squared joint torque change vs parent (%)',title='Leaning diagnostic: coordinated posture changes can save effort\nThese are prescribed perturbations, not learned PPO results')
ax.axhline(0,color='black',lw=1);ax.grid(axis='y',alpha=.2);ax.legend()
fig.savefig(OUT/'leaning_action_diagnosis.png',dpi=160)
(OUT/'leaning_action_summary.json').write_text(json.dumps(summary,indent=2))
print(OUT)
