"""Paired complete-trial PPO results and a scientific evaluation visualization."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--baseline-only',action='store_true'); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; store=Path('/data/humanoid-sitting-perception')
out=Path(a.output); out.mkdir(parents=True,exist_ok=False)


def load(name):
    path=store/name
    return dict(path=path,result=json.loads((path/'results.json').read_text()),
        manifest=json.loads((path/'manifest.json').read_text()),bank=json.loads((path/'geometry_bank.json').read_text()),
        viz=json.loads((path/'visualization_metadata.json').read_text()),trace=dict(np.load(path/'rollout.npz')))


def summarize(run):
    rows=run['result']['episodes']; z=run['trace']; ok=z['live'][...,8]>0
    error=100*abs(z['live'][...,0]-z['true_clearance'])
    return dict(trials=len(rows),successes=sum(r['success'] for r in rows),
        maximum_impact_N=max(r['peak_force_N'] for r in rows),
        median_impact_N=float(np.median([r['peak_force_N'] for r in rows])),
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']},
        depth_accepted=int(ok.sum()),depth_samples=ok.size,
        max_accepted_clearance_error_cm=float(error[ok].max()) if ok.any() else None,
        accepted_clearance_errors_above_2cm=int(((error>2)&ok).sum()))


baseline=load('support_rl_zero_train_01')
if a.baseline_only:
    runs={'baseline':baseline}; pairs=[]
else:
    runs=dict(train_baseline=baseline,train_ppo=load('support_rl_eval_train_01'),
              heldout_baseline=load('support_rl_zero_holdout_01'),heldout_ppo=load('support_rl_eval_holdout_01'))
    pairs=[('train_baseline','train_ppo'),('heldout_baseline','heldout_ppo')]
summaries={name:summarize(run) for name,run in runs.items()}
paired={}
for before,after in pairs:
    b,c=runs[before],runs[after]
    assert b['bank']==c['bank']
    assert b['manifest']['arguments']['seed']==c['manifest']['arguments']['seed']
    assert b['manifest']['hashes']==c['manifest']['hashes'], 'Numerical sources or scaffold differ'
    br=b['result']['episodes']; cr=c['result']['episodes']
    good=[i for i,(x,y) in enumerate(zip(br,cr)) if x['success'] and y['success']]
    paired[after]=dict(success_gained=sum(not x['success'] and y['success'] for x,y in zip(br,cr)),
        success_lost=sum(x['success'] and not y['success'] for x,y in zip(br,cr)),
        both_successful=len(good),
        mean_paired_leg_rms_change_Nm=float(np.mean([cr[i]['seated_leg_rms_Nm']-br[i]['seated_leg_rms_Nm'] for i in good])) if good else None,
        mean_paired_foot_load_change=float(np.mean([cr[i]['seated_foot_load_fraction']-br[i]['seated_foot_load_fraction'] for i in good])) if good else None)

fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
names=list(runs); colors=['#718096','#0e7490','#9ca3af','#10b981'][:len(names)]
axes[0,0].bar(names,[summaries[n]['successes']/summaries[n]['trials'] for n in names],color=colors)
axes[0,0].set(ylim=(0,1.12),ylabel='Complete-trial success fraction',title='Full 20-second attempts; failures latched')
for i,n in enumerate(names):axes[0,0].text(i,summaries[n]['successes']/summaries[n]['trials']+.025,f"{summaries[n]['successes']}/{summaries[n]['trials']}",ha='center')
for i,n in enumerate(names):
    rows=runs[n]['result']['episodes']
    axes[0,1].scatter(np.full(len(rows),i),[r['peak_force_N'] for r in rows],color=colors[i],alpha=.75)
    good=[r for r in rows if r['success']]
    axes[1,0].scatter(np.full(len(good),i),[r['seated_leg_rms_Nm'] for r in good],color=colors[i],alpha=.75)
    axes[1,1].scatter(np.full(len(good),i),[r['seated_foot_load_fraction'] for r in good],color=colors[i],alpha=.75)
for ax in axes.flat:
    ax.set_xticks(range(len(names)),names,rotation=15,fontsize=8); ax.grid(axis='y',alpha=.2)
axes[0,1].axhline(632.527832,ls='--',color='firebrick',label='Existing two-body-weight limit')
axes[0,1].set(ylabel='Peak normal seat force (N)',title='All trials, including failures'); axes[0,1].legend(fontsize=8)
axes[1,0].set(ylabel='Seated leg RMS torque (Nm)',title='Successful trials only; paired changes in report')
axes[1,1].set(ylabel='Fraction of body weight on feet',title='Successful trials only; feet may stabilize')
fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)

model=json.loads((root/'runs/g1_kinematics_geometry_04/model.json').read_text())


def draw_robot(ax,run,frame,case):
    z=run['trace']; origin=np.asarray(run['viz']['origins'][case]); positions=z['body'][frame,case,:,:3]-origin
    lookup={name:positions[i] for i,name in enumerate(run['viz']['body_names'])}
    for j in model['joints']:
        if j['body0'] and j['body0'][0] in lookup and j['body1'][0] in lookup:
            pair=np.array([lookup[j['body0'][0]],lookup[j['body1'][0]]])
            ax.plot(*pair.T,color='#263b50',lw=2)
    ax.scatter(*lookup['pelvis'],s=70,color='#f59e0b')
    g=run['bank'][run['viz']['geometry_indices'][case]]
    yaw=g['yaw']; R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    xy=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[g['depth'],g['width']]
    xy=xy@R.T+[g['x'],g['y']]
    polygon=np.c_[xy,np.full(4,g['height'])]
    ax.add_collection3d(Poly3DCollection([polygon],facecolor='#10b981',alpha=.35,edgecolor='#047857'))
    if g['family'] in ('chair','rigid_couch'):
        back=np.array([[-g['depth']/2,-g['width']/2],[-g['depth']/2,g['width']/2]])@R.T+[g['x'],g['y']]
        wall=np.array([[*back[0],g['height']],[*back[1],g['height']],[*back[1],g['height']+.35],[*back[0],g['height']+.35]])
        ax.add_collection3d(Poly3DCollection([wall],facecolor='#10b981',alpha=.15))
    ax.set(xlim=(-.8,.4),ylim=(-.7,.7),zlim=(0,1.5),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_box_aspect((1.2,1.4,1.5)); ax.view_init(elev=17,azim=-66)


clips=[('baseline','baseline',0)] if a.baseline_only else [
    ('train_baseline','train_ppo',0),('train_baseline','train_ppo',5),('heldout_baseline','heldout_ppo',0),('heldout_baseline','heldout_ppo',5)]
fig=plt.figure(figsize=(14,9)); writer=FFMpegWriter(fps=10,codec='libx264',bitrate=1800)
with writer.saving(fig,str(out/'sitting_comparison.mp4'),100):
    for ci,(left,right,case) in enumerate(clips):
        for frame in range(0,200,2):
            fig.clear()
            grid=fig.add_gridspec(2,2,height_ratios=[1.35,1.],hspace=.40,wspace=.16,
                                 top=.85,bottom=.08,left=.06,right=.98)
            for col,key in enumerate((left,right)):
                run=runs[key]; z=run['trace']; t=float(z['t'][frame]); row=run['result']['episodes'][case]
                ax=fig.add_subplot(grid[0,col],projection='3d'); draw_robot(ax,run,frame,case)
                status='PASS' if row['success'] else 'FAIL: '+', '.join(k for k,v in row['failures'].items() if v)
                ax.set_title(f'{key.replace("_"," ")} | case {case} | t={t:.1f}s\nWhole-trial result: {status}',fontsize=10)
                ax=fig.add_subplot(grid[1,col])
                ax.plot(z['t'],z['support'][:,case],color='#047857',label='Seat load / body weight')
                ax.plot(z['t'],z['foot'][:,case],color='#2563eb',label='Foot load / body weight')
                ax.axvline(t,color='black',lw=1); ax.axhline(.75,color='#047857',ls=':',alpha=.5)
                ax.set(xlim=(0,20),ylim=(0,2.5),xlabel='Time (s)',ylabel='Load fraction'); ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper right')
                live=z['live'][frame,case]
                depth=f'{live[0]*100:.1f} cm' if live[8] else 'unavailable'
                ax.set_title(f'Leg torque RMS: {z["torque"][frame,case]:.1f} Nm | depth clearance: {depth}',fontsize=10)
            fig.suptitle('Simulator evaluation reconstruction — physical poses/forces shown only for evaluation\nPPO joint corrections over a frozen sensor-based controller | 2x playback',fontsize=11)
            if frame==130:fig.savefig(out/f'case_{ci}.png',dpi=120)
            writer.grab_frame()
plt.close(fig)
report=dict(summaries=summaries,paired=paired,limitations='Small matched pilot; rigid prealigned seats; frozen motion scaffold remains; no hardware or arbitrary-surface claim.')
if not a.baseline_only:
    report['training_verification']=json.loads((store/'support_rl_train_01/verification.json').read_text())
    report['training_results']={k:v for k,v in json.loads((store/'support_rl_train_01/results.json').read_text()).items() if k!='episodes'}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Supported-sitting PPO pilot results','',report['limitations'],'',
       '| Evaluation | Success | Median / max impact N | Accepted depth error max cm |',
       '|---|---:|---:|---:|']
for name,v in summaries.items():
    lines.append(f'| {name} | {v["successes"]}/{v["trials"]} | {v["median_impact_N"]:.1f} / {v["maximum_impact_N"]:.1f} | {v["max_accepted_clearance_error_cm"]:.3f} |')
lines.extend(['','Paired torque/load changes use only cases successful under both controllers. Complete-trial success and all-trial impact are reported separately to avoid hiding failures.','',json.dumps(paired,indent=2),''])
(out/'report.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
