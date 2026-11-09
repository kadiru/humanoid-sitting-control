"""Audited complete-horizon comparisons for the independent-reset PPO pilot."""
import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from seat_depth_estimator import estimate_body_frame

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--skip-video',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception')
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'report_source.py')
shutil.copytree(store/'support_rl_sensor_frame_01',out/'sensor_frame')
shutil.copy2(root/'scripts/visualize_support_sensor_frame.py',out/'sensor_frame'/'visualization_source.py')
names={k:'support_rl_independent_'+k+'_04' for k in (
    'fresh_zero','fresh_prior','fresh_candidate','regression_prior','regression_candidate')}
labels=dict(fresh_zero='Frozen controller',fresh_prior='Previous PPO',fresh_candidate='New PPO',
            regression_prior='Previous PPO',regression_candidate='New PPO')
colors=dict(fresh_zero='#64748b',fresh_prior='#c2410c',fresh_candidate='#047857',
            regression_prior='#c2410c',regression_candidate='#047857')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def load(name):
    path=store/name
    return dict(path=path,result=read(path/'results.json'),manifest=read(path/'manifest.json'),
        bank=read(path/'geometry_bank.json'),trace=dict(np.load(path/'rollout.npz')),
        viz=read(path/'visualization_metadata.json'))
def source_hashes(run):return {posixpath.normpath(k):v for k,v in run['manifest']['hashes'].items()}
def audit_sources(path):
    manifest=read(path/'manifest.json');checked={}
    for key,value in manifest['hashes'].items():
        archive=path/'source'/Path(key).name
        source=archive if archive.exists() else root/posixpath.normpath(key)
        assert digest(source)==value,(source,value)
        checked[key]=value
    return checked
def summary(run):
    rows=run['result']['episodes'];z=run['trace']
    assert len(rows)==16 and all(r['steps']==1000 and r['duration_s']==20 for r in rows)
    assert len(z['t'])==200 and np.allclose(z['t'],np.arange(1,201)*.1)
    ok=z['live'][...,8]>0;error=abs(z['live'][...,0]-z['true_clearance'])*100
    correction=abs(z['correction'])
    return dict(trials=len(rows),successes=sum(r['success'] for r in rows),
        failed_cases=[r['env'] for r in rows if not r['success']],
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']},
        worst_impact_N=max(r['peak_force_N'] for r in rows),
        median_impact_N=float(np.median([r['peak_force_N'] for r in rows])),
        depth_accepted=int(ok.sum()),depth_samples=int(ok.size),
        maximum_accepted_clearance_error_cm=float(error[ok].max()) if ok.any() else None,
        accepted_errors_above_2cm=int(((error>2)&ok).sum()),
        correction_max_rad=float(correction.max()*.08),
        correction_p95_rad=float(np.quantile(correction,.95)*.08),
        correction_at_bound_fraction=float((correction>=.999).mean()),
        elapsed_s=run['result']['elapsed_s'])
def compare(bkey,ckey):
    b,c=runs[bkey],runs[ckey]
    assert b['bank']==c['bank'] and b['manifest']['arguments']['seed']==c['manifest']['arguments']['seed']
    assert source_hashes(b)==source_hashes(c)
    br,cr=b['result']['episodes'],c['result']['episodes']
    common=[i for i,(x,y) in enumerate(zip(br,cr)) if x['success'] and y['success']]
    bt=float(np.mean([br[i]['seated_leg_rms_Nm'] for i in common])) if common else None
    ct=float(np.mean([cr[i]['seated_leg_rms_Nm'] for i in common])) if common else None
    return dict(baseline=bkey,candidate=ckey,matched_successes=len(common),
        gained_cases=[i for i,(x,y) in enumerate(zip(br,cr)) if not x['success'] and y['success']],
        lost_cases=[i for i,(x,y) in enumerate(zip(br,cr)) if x['success'] and not y['success']],
        baseline_leg_rms_Nm=bt,candidate_leg_rms_Nm=ct,
        leg_torque_change_percent=100*(ct/bt-1) if bt else None)

runs={k:load(v) for k,v in names.items()}
summaries={k:summary(r) for k,r in runs.items()}
comparisons=[compare('fresh_zero','fresh_prior'),compare('fresh_prior','fresh_candidate'),
             compare('fresh_zero','fresh_candidate'),compare('regression_prior','regression_candidate')]
train=store/'support_rl_independent_train_04'
tr=read(train/'results.json');account=read(train/'attempt_accounting.json');verification=read(train/'verification.json')
assert account['completed_transitions']+account['unfinished_transitions']==verification['transitions']==98304
for env in range(16):
    steps=sum(r['steps'] for r in tr['episodes'] if r['env']==env)
    steps+=sum(r['steps'] for r in account['unfinished_attempts'] if r['env']==env)
    assert steps==6144,(env,steps)
assert verification['finite'] and verification['max_parameter_change']>0 and verification['optimizer_contains_actor']
parent=torch.load(store/'support_rl_diverse_train_02/final.pt',map_location='cpu',weights_only=False)
initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False)
final=torch.load(train/'final.pt',map_location='cpu',weights_only=False)
for group in ('actor_state_dict','critic_state_dict'):
    assert parent[group].keys()==initial[group].keys()==final[group].keys()
    for key,value in parent[group].items():
        assert torch.equal(value,initial[group][key]),(group,key,'warm start mismatch')
        assert torch.isfinite(final[group][key]).all(),(group,key,'nonfinite checkpoint')
probe=read(store/'support_rl_reset_probe_04/reset_probe.json');assert probe['passed']
audit={name:audit_sources(store/name) for name in [*names.values(),train.name,'support_rl_reset_probe_04']}
training=dict(verification=verification,accounting=account,
    successes=tr['successes'],complete_attempts=len(tr['episodes']),
    refusals=sum(not r['startup_ready'] for r in tr['episodes']),
    failures={k:sum(r['failures'][k] for r in tr['episodes']) for k in tr['episodes'][0]['failures']},
    elapsed_s=tr['elapsed_s'],final_checkpoint_sha256=digest(train/'final.pt'),
    warm_start_actor_and_critic_exact=True,final_actor_and_critic_finite=True,
    parent_checkpoint_sha256=digest(store/'support_rl_diverse_train_02/final.pt'))
report=dict(runs=names,summaries=summaries,comparisons=comparisons,training=training,reset_probe=probe,
    promotion='Experimental candidate only; the existing default controller is unchanged.',
    protocol='Full 20-second evaluation with latched failures. Training resets independently at failure or horizon.',
    startup_change='Indexed reset with a one-control-tick pose hold before camera admission; no five-step global settling interval. All comparisons here use this same new startup protocol.',
    limitations=['Small pilot with 8 fresh geometries and two starts each, not statistical reliability proof.',
        'Rigid seats 26–35 cm high; no deformable cushions, navigation or hardware validation.',
        'Frozen motion scaffold and bounded +/-0.08 rad residual remain. Lowering timing is not a learned action yet.',
        'Sensor-only actor and critic; true pose/contact used only for reward, termination and offline evaluation.',
        'Foot force remains an evaluation label, not an assumed hardware input.',
        'Geometry prototypes are fixed within a run; robot starts vary at each independent reset.',
        'Unobservable starts are counted as refusals; unfinished training attempts are recorded separately.',
        'Video force curves are sampled at 10 Hz and can miss brief spikes; the separate whole-trial peak line uses every 50 Hz evaluation step.',
        'Numerical source, reward and final-checkpoint selection were fixed before the corrected fresh holdout. Earlier camera-reset diagnostics are excluded.'])
camera_checks={}
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for label,name,color in [('Before','support_rl_camera_probe_01','#c2410c'),('After','support_rl_camera_probe_02','#047857')]:
    path=store/name;capture=np.load(path/'startup_capture.npz');trace=np.load(path/'rollout.npz')
    measured=np.array([estimate_body_frame(d,K,capture['mount'],capture['quat'],g)['surface_center'][2]
                       for d,K,g in zip(capture['depth'],capture['K'],capture['gravity'])])
    error=measured-capture['true_height'];ok=trace['live'][...,8]>0
    camera_checks[label]=dict(maximum_startup_height_error_mm=float(abs(error).max()*1000),
        accepted_live_samples=int(ok.sum()),live_samples=int(ok.size))
    axes[0].plot(range(16),error*1000,'o-',color=color,label=label)
    axes[1].bar(label,int(ok.sum()),color=color)
axes[0].axhline(0,color='black',lw=.8);axes[0].legend();axes[0].grid(alpha=.2)
axes[0].set(xlabel='Diagnostic robot',ylabel='Measured minus true seat height (mm)',title='Startup camera freshness')
axes[1].set(ylim=(0,35),ylabel='Usable live measurements / 32',title='One normal hold tick before camera admission')
fig.savefig(out/'camera_reset_fix.png',dpi=150);plt.close(fig)
report['camera_reset_diagnostic']=camera_checks
report['excluded_debug_runs']=['support_rl_independent_fresh_zero_03','support_rl_independent_fresh_prior_03']
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'bundle_audit.json').write_text(json.dumps(dict(sources=audit,reset_probe_passed=True,
    transition_accounting_passed=True,evaluation_case_frames=sum(s['depth_samples'] for s in summaries.values())),indent=2)+'\n')

fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for col,(title,keys) in enumerate([('Fresh, wider seats',['fresh_zero','fresh_prior','fresh_candidate']),
                                 ('Known regression seats',['regression_prior','regression_candidate'])]):
    x=np.arange(len(keys));success=[summaries[k]['successes'] for k in keys]
    axes[0,col].bar(x,success,color=[colors[k] for k in keys])
    for j,n in enumerate(success):axes[0,col].text(j,n+.25,f'{n}/16',ha='center')
    axes[0,col].set(title=title,ylabel='Successful complete attempts',ylim=(0,18))
    axes[1,col].bar(x,[summaries[k]['worst_impact_N'] for k in keys],color=[colors[k] for k in keys])
    axes[1,col].axhline(632.527832,color='#b91c1c',ls='--',label='Project impact threshold')
    axes[1,col].set(ylabel='Worst seat impact across all attempts (N)');axes[1,col].legend(fontsize=8)
    for row in range(2):
        axes[row,col].set_xticks(x,[labels[k] for k in keys]);axes[row,col].grid(axis='y',alpha=.2)
fig.suptitle('New training infrastructure; full-attempt evaluation',fontsize=15)
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)

fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True)
palette={True:'#047857',False:'#dc2626'}
for env in range(16):
    start=0
    for row in sorted((r for r in tr['episodes'] if r['env']==env),key=lambda r:r['episode']):
        ax.broken_barh([(start,row['duration_s'])],(env-.35,.7),facecolors=palette[row['success']])
        start+=row['duration_s']
    ax.broken_barh([(start,122.88-start)],(env-.35,.7),facecolors='#94a3b8')
ax.set(xlabel='Simulation time per robot (s)',ylabel='Parallel robot',yticks=range(16),xlim=(0,122.88),
       title='Each robot starts again independently after a failure\nGreen: full successful attempt · red: failed/refused attempt · gray: unfinished at budget')
ax.grid(axis='x',alpha=.25);fig.savefig(out/'independent_attempts.png',dpi=150);plt.close(fig)

lines=['# Independent-reset PPO pilot','',report['promotion'],'',report['protocol'],'',
    '| Evaluation | Success | Worst impact | Failures |','|---|---:|---:|---|']
for key,s in summaries.items():
    failures=', '.join(f'{k}: {v}' for k,v in s['failures'].items() if v) or 'none'
    lines.append(f"| {key} | {s['successes']}/16 | {s['worst_impact_N']:.1f} N | {failures} |")
lines+=['',f"Training added 98,304 transitions in {training['elapsed_s']:.1f} seconds. "
    f"Completed {training['complete_attempts']} attempts, with {training['successes']} successes and {training['refusals']} startup refusals. "
    'Training failures end early; these counts are not a held-out success score. All transitions and unfinished attempts are accounted for.','']
for comp in comparisons:
    change=comp['leg_torque_change_percent'];delta='unavailable' if change is None else f'{change:+.2f}%'
    lines.append(f"- {comp['baseline']} → {comp['candidate']}: gained {comp['gained_cases']}; lost {comp['lost_cases']}; "
        f"leg torque {delta} on {comp['matched_successes']} matched successful attempts.")
lines+=['',report['startup_change'],'','Reset probe: other robots had exactly zero measured state changes; no physics tick; selected controller matched a cold start. Twelve targeted reward/admission/reset tests pass.','']
lines+=['The original no-warmup reset exposed old camera transforms. A paired diagnostic reduced maximum startup height error '
    f"from {camera_checks['Before']['maximum_startup_height_error_mm']:.2f} to {camera_checks['After']['maximum_startup_height_error_mm']:.2f} mm, "
    'and increased usable live depth samples from 6/32 to 27/32. The two preliminary `_03` full evaluations are preserved as debug runs and excluded from this comparison. No main PPO training had run before that correction.','']
lines+=['- '+s for s in report['limitations']]
current=summaries['fresh_candidate']
lines+=['',f"On the fresh candidate rollout, 95% of absolute joint corrections were below {np.rad2deg(current['correction_p95_rad']):.3f} degrees "
    f"and the maximum was {np.rad2deg(current['correction_max_rad']):.3f} degrees. "
    f"The fraction reaching the existing bound was {current['correction_at_bound_fraction']:.6f}. "
    'This run does not establish that the hard action bound caused the failures. The policy used only a small part of its available range; structured exploration and learning the lowering motion are the next hypotheses to test, not a validated fix.','']
lines+=['','The earlier reset probe 02 stopped on a comparison between different GPU batch sizes (maximum latent difference 0.00005245). The final probe compares the same batch size at unchanged tight tolerances and passes. Its earlier log and source are preserved.','']
(out/'report.md').write_text('\n'.join(lines))
print('INDEPENDENT_REPORT',json.dumps(dict(summaries=summaries,comparisons=comparisons,training=training)),flush=True)
if a.skip_video:raise SystemExit

model=read(root/'runs/g1_kinematics_geometry_04/model.json')
def draw_robot(ax,run,frame,case):
    positions=run['trace']['body'][frame,case,:,:3]-np.asarray(run['viz']['origins'][case])
    lookup={n:positions[i] for i,n in enumerate(run['viz']['body_names'])}
    for j in model['joints']:
        if j['body0'] and j['body0'][0] in lookup and j['body1'][0] in lookup:
            points=np.array([lookup[j['body0'][0]],lookup[j['body1'][0]]]);ax.plot(*points.T,color='#263b50',lw=2)
    ax.scatter(*lookup['pelvis'],s=55,color='#f59e0b')
    g=run['bank'][run['viz']['geometry_indices'][case]];yaw=g['yaw']
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    xy=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[g['depth'],g['width']]
    xy=xy@R.T+[g['x'],g['y']]
    ax.add_collection3d(Poly3DCollection([np.c_[xy,np.full(4,g['height'])]],facecolor='#10b981',alpha=.35,edgecolor='#047857'))
    if g['family'] in ('chair','rigid_couch'):
        back=np.array([[-g['depth']/2,-g['width']/2],[-g['depth']/2,g['width']/2]])@R.T+[g['x'],g['y']]
        wall=np.array([[*back[0],g['height']],[*back[1],g['height']],[*back[1],g['height']+.35],[*back[0],g['height']+.35]])
        ax.add_collection3d(Poly3DCollection([wall],facecolor='#10b981',alpha=.15))
    ax.set(xlim=(-1.3,.5),ylim=(-.8,.8),zlim=(0,1.5))
    ax.set_xticks([-1.,0.]);ax.set_yticks([-.5,.5]);ax.set_zticks([0.,.75,1.5])
    ax.set_box_aspect((1.8,1.6,1.5));ax.view_init(elev=18,azim=-65)

comp=comparisons[1]
if comp['lost_cases']:
    failure=comp['lost_cases'][0];first_label='first lost success'
elif summaries['fresh_candidate']['failed_cases']:
    failure=summaries['fresh_candidate']['failed_cases'][0];first_label='first failed case'
else:
    failure=0;first_label='first successful case'
successes=[i for i,r in enumerate(runs['fresh_candidate']['result']['episodes']) if r['success'] and i!=failure]
if comp['gained_cases']:
    gain=comp['gained_cases'][0];second_label='first gained success'
elif successes:
    gain=successes[0];second_label='next successful case'
else:
    gain=next(i for i in range(16) if i!=failure);second_label='next failed case'
clips=[(failure,'Fresh wider seat: '+first_label),(gain,'Fresh wider seat: '+second_label)]
(out/'clip_selection.json').write_text(json.dumps(clips,indent=2)+'\n')
fig=plt.figure(figsize=(16,8));writer=FFMpegWriter(fps=10,codec='libx264',bitrate=2000)
with writer.saving(fig,str(out/'sitting_comparison.mp4'),100):
    for ci,(case,title) in enumerate(clips):
        for frame in range(0,200,2):
            fig.clear();grid=fig.add_gridspec(2,3,height_ratios=[1.5,1],hspace=.35,wspace=.3,top=.83,bottom=.08,left=.05,right=.98)
            for col,key in enumerate(('fresh_zero','fresh_prior','fresh_candidate')):
                run=runs[key];z=run['trace'];row=run['result']['episodes'][case];t=float(z['t'][frame])
                ax=fig.add_subplot(grid[0,col],projection='3d');draw_robot(ax,run,frame,case)
                status='PASS' if row['success'] else 'FAIL: '+', '.join(k for k,v in row['failures'].items() if v)
                ax.set_title(f'{labels[key]} | case {case} | {t:.1f}s\n{status}',fontsize=10)
                ax=fig.add_subplot(grid[1,col]);ax.plot(z['t'],z['force'][:,case],color=colors[key],label='Seat force')
                ax.axhline(632.527832,color='#b91c1c',ls='--',label='Impact threshold');ax.axvline(t,color='black',lw=1)
                ax.axhline(row['peak_force_N'],color=colors[key],ls=':',lw=1.5,label='Whole-trial peak (50 Hz)')
                ymax=max(750,max(runs[k]['result']['episodes'][case]['peak_force_N'] for k in ('fresh_zero','fresh_prior','fresh_candidate'))*1.08)
                ax.set(xlim=(0,20),ylim=(0,ymax),xlabel='Time (s)',ylabel='Seat force (N)');ax.grid(alpha=.2)
                live=z['live'][frame,case];clearance=f'{live[0]*100:.1f} cm' if live[8] else 'unavailable'
                ax.set_title(f'Measured clearance: {clearance}\nLeg torque: {z["torque"][frame,case]:.1f} Nm | trial peak: {row["peak_force_N"]:.0f} N',fontsize=10)
                if col==0:ax.legend(fontsize=8)
            fig.suptitle(title+'\nSimulator pose/force reconstruction for evaluation only · 2x playback\nForce curve: 10 Hz samples · dotted line: full-trial peak from 50 Hz checks',fontsize=12)
            if frame==130:fig.savefig(out/f'case_{ci}.png',dpi=120)
            writer.grab_frame()
plt.close(fig)
