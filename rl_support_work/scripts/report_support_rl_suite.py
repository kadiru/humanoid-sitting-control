"""Report both bounded PPO pilots, including regressions and complete failures."""
import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--skip-video', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
store = Path('/data/humanoid-sitting-perception')
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)

names = dict(
    train0='support_rl_zero_train_01', train1='support_rl_eval_train_01',
    regression0='support_rl_zero_holdout_01', regression1='support_rl_eval_holdout_01',
    extended0='support_rl_zero_extended_01', extended1='support_rl_eval_extended_01',
    ablation1='support_rl_no_live_depth_01', regression2base='support_rl_zero_regression_02',
    regression2='support_rl_evaluate_regression_02',
    fresh0='support_rl_zero_fresh_02', fresh2='support_rl_evaluate_fresh_02',
    balanced0='support_rl_zero_balanced_02', balanced2='support_rl_evaluate_balanced_02')
pairs = [
    ('Pilot 1 training bank', 'train0', 'train1'),
    ('Pilot 1 unseen bank', 'regression0', 'regression1'),
    ('Pilot 1 larger unseen bank', 'extended0', 'extended1'),
    ('Pilot 1 live-depth ablation', 'regression0', 'ablation1'),
    ('Pilot 2 known regression bank', 'regression2base', 'regression2'),
    ('Pilot 2 fresh unseen bank', 'fresh0', 'fresh2'),
    ('Pilot 2 training bank', 'balanced0', 'balanced2')]


def load(name):
    path = store/name
    return dict(path=path, result=json.loads((path/'results.json').read_text()),
        manifest=json.loads((path/'manifest.json').read_text()),
        bank=json.loads((path/'geometry_bank.json').read_text()),
        viz=json.loads((path/'visualization_metadata.json').read_text()),
        trace=dict(np.load(path/'rollout.npz')))


def summarize(run):
    rows = run['result']['episodes']; z = run['trace']; ok = z['live'][..., 8] > 0
    error = 100*abs(z['live'][..., 0]-z['true_clearance'])
    return dict(trials=len(rows), successes=sum(r['success'] for r in rows),
        maximum_impact_N=max(r['peak_force_N'] for r in rows),
        median_impact_N=float(np.median([r['peak_force_N'] for r in rows])),
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']},
        failed_cases=[r['env'] for r in rows if not r['success']],
        depth_accepted=int(ok.sum()), depth_samples=ok.size,
        max_accepted_clearance_error_cm=float(error[ok].max()) if ok.any() else None,
        accepted_clearance_errors_above_2cm=int(((error > 2) & ok).sum()),
        elapsed_s=run['result']['elapsed_s'],
        step_median_s=run['result']['step_median_s'],
        step_p95_s=run['result']['step_p95_s'])


runs = {k:load(v) for k,v in names.items()}
summaries = {k:summarize(v) for k,v in runs.items()}
comparisons = []
for label, bkey, ckey in pairs:
    b,c = runs[bkey],runs[ckey]
    assert b['bank'] == c['bank']
    assert b['manifest']['arguments']['seed'] == c['manifest']['arguments']['seed']
    # Direct CLI and imported entry points record the same source as
    # IsaacLab/../scripts/x.py and scripts/x.py. Compare normalized path keys;
    # retain and compare every original content hash without modification.
    def normalized_hashes(run):
        original=run['manifest']['hashes']
        normalized={posixpath.normpath(k):v for k,v in original.items()}
        assert len(normalized)==len(original)
        return normalized
    assert normalized_hashes(b) == normalized_hashes(c)
    br,cr = b['result']['episodes'],c['result']['episodes']
    assert len(br) == len(cr)
    common = [i for i,(x,y) in enumerate(zip(br,cr)) if x['success'] and y['success']]
    baseline_torque=float(np.mean([br[i]['seated_leg_rms_Nm'] for i in common])) if common else None
    candidate_torque=float(np.mean([cr[i]['seated_leg_rms_Nm'] for i in common])) if common else None
    comparisons.append(dict(label=label, baseline=bkey, candidate=ckey,
        common_successes=len(common),
        gained_cases=[i for i,(x,y) in enumerate(zip(br,cr)) if not x['success'] and y['success']],
        lost_cases=[i for i,(x,y) in enumerate(zip(br,cr)) if x['success'] and not y['success']],
        paired_leg_rms_change_Nm=float(np.mean([cr[i]['seated_leg_rms_Nm']-br[i]['seated_leg_rms_Nm'] for i in common])) if common else None,
        paired_baseline_leg_rms_Nm=baseline_torque, paired_candidate_leg_rms_Nm=candidate_torque,
        paired_leg_rms_change_percent=100*(candidate_torque/baseline_torque-1) if common and baseline_torque else None,
        paired_foot_fraction_change=float(np.mean([cr[i]['seated_foot_load_fraction']-br[i]['seated_foot_load_fraction'] for i in common])) if common else None))

training = {}
training_runs = {}
for key,name in [('pilot1','support_rl_train_01'),('pilot2','support_rl_diverse_train_02')]:
    path = store/name
    result = json.loads((path/'results.json').read_text())
    training_runs[key] = dict(result=result, bank=json.loads((path/'geometry_bank.json').read_text()))
    training[key] = dict(path=str(path), verification=json.loads((path/'verification.json').read_text()),
        results={k:v for k,v in result.items() if k != 'episodes'},
        final_checkpoint_sha256=hashlib.sha256((path/'final.pt').read_bytes()).hexdigest())

report = dict(runs=names, summaries=summaries, comparisons=comparisons, training=training,
    interrupted_run=json.loads((store/'support_rl_diverse_train_01/interruption.json').read_text()),
    promotion='Neither checkpoint replaces the existing default controller.',
    scope='Two small residual PPO pilots over a frozen sensor-based motion scaffold; rigid reachable seats behind the robot.',
    limitations=['No navigation, soft cushions, arbitrary surfaces, hardware validation or statistical superiority established.',
        'Actor and critic use sensor-derived channels. Simulator truth is used only for reward/failure labels and evaluation.',
        'Ideal simulated depth and IMU equivalents; existing small proprioceptive noise/delay, not calibrated hardware sensing.',
        'Foot force is a reward/evaluation label, not an actor input. Physical foot-pressure sensors are unverified.',
        'Synchronized batch resets waste samples after a latched failure; this is a bounded pilot implementation.',
        'The first balanced run stopped at its first reset on hidden startup geometry and is preserved. Pilot 2 counts unobservable starts as refused failures and holds posture; no resampling or truth fallback.',
        'Torque comparisons include only matched successful attempts; all-trial failures and impact are reported separately.',
        'PPO final checkpoints were fixed for evaluation. Pilot 1 failures informed Pilot 2 training coverage; that old bank is a regression test, not a fresh holdout.'])
old_baseline=runs['regression0']['result']['episodes']
guarded_baseline=runs['regression2base']['result']['episodes']
report['admission_regression_check']=dict(
    old_successes=sum(r['success'] for r in old_baseline),
    guarded_successes=sum(r['success'] for r in guarded_baseline),
    guarded_refusals=sum(not r['startup_ready'] for r in guarded_baseline),
    maximum_paired_peak_force_difference_N=max(abs(b['peak_force_N']-c['peak_force_N']) for b,c in zip(old_baseline,guarded_baseline)))
(out/'results.json').write_text(json.dumps(report, indent=2)+'\n')

fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for key,color in [('pilot1','#d97706'),('pilot2','#0e7490')]:
    tr=training_runs[key]; bank=tr['bank']; episodes=tr['result']['episodes']
    axes[0,0].scatter([g['depth'] for g in bank],np.rad2deg([g['yaw'] for g in bank]),color=color,label=key)
    axes[0,1].scatter([g['width'] for g in bank],[g['height'] for g in bank],color=color,label=key)
    batch_ids=sorted({r['episode'] for r in episodes})
    groups=[[r for r in episodes if r['episode']==i] for i in batch_ids]
    axes[1,0].plot(np.arange(1,len(groups)+1),[np.mean([r['success'] for r in rows]) for rows in groups],marker='o',color=color,label=key)
    axes[1,1].plot(np.arange(1,len(groups)+1),[np.mean([r['reward'] for r in rows]) for rows in groups],marker='o',color=color,label=key)
axes[0,0].set(xlabel='Seat depth (m)',ylabel='Yaw (degrees)',title='Training geometry coverage')
axes[0,1].set(xlabel='Seat width (m)',ylabel='Seat height (m)',title='Training geometry coverage')
axes[1,0].set(xlabel='Completed synchronized batch',ylabel='Full-attempt success fraction',ylim=(-.05,1.05),title='Stochastic training attempts, not held-out tests')
axes[1,1].set(xlabel='Completed synchronized batch',ylabel='Mean episode reward',title='Real gated reward; failed attempts stay at zero afterward')
for ax in axes.flat:
    ax.grid(alpha=.2); ax.legend()
fig.savefig(out/'training_diagnostics.png',dpi=150); plt.close(fig)

fig,axes = plt.subplots(3,1,figsize=(13,11),constrained_layout=True)
x = np.arange(len(pairs)); width=.34
for shift,role,color in [(-width/2,'baseline','#64748b'),(width/2,'candidate','#0e7490')]:
    keys = [c[role] for c in comparisons]
    vals = [summaries[k]['successes']/summaries[k]['trials'] for k in keys]
    axes[0].bar(x+shift, vals, width, color=color, label=role)
    for xx,key,val in zip(x+shift,keys,vals):
        s=summaries[key]; axes[0].text(xx,val+.015,f"{s['successes']}/{s['trials']}",ha='center',fontsize=8)
    axes[1].scatter(x+shift,[summaries[k]['maximum_impact_N'] for k in keys],color=color,label=role)
axes[0].set(ylabel='Full-trial success',ylim=(0,1.13),title='Complete 20-second attempts; no early-reset success counting')
axes[0].legend(loc='lower left')
axes[1].axhline(632.527832,color='firebrick',ls='--',label='Two-body-weight project threshold')
axes[1].set(ylabel='Worst seat force (N)',title='All attempts, including falls'); axes[1].legend(fontsize=8)
changes=[c['paired_leg_rms_change_Nm'] for c in comparisons]
axes[2].bar(x,[np.nan if c is None else c for c in changes],color='#7c3aed')
axes[2].axhline(0,color='black',lw=.8)
axes[2].set(ylabel='PPO minus baseline (Nm)',title='Mean leg torque change on matched successful attempts only; lower is less torque')
labels=['P1 training','P1 unseen','P1 more unseen','P1 no live depth','P2 regression','P2 fresh unseen','P2 training']
for ax in axes:
    ax.set_xticks(x,labels,fontsize=9); ax.grid(axis='y',alpha=.2)
fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)

lines=['# Two supported-sitting PPO pilots','',report['promotion'],'',report['scope'],'',
    '| Paired evaluation | Baseline success | PPO success | Baseline / PPO worst seat force (N) | Matched success leg torque change (Nm) |',
    '|---|---:|---:|---:|---:|']
for comp in comparisons:
    b,c=summaries[comp['baseline']],summaries[comp['candidate']]
    delta=comp['paired_leg_rms_change_Nm']; delta='n/a' if delta is None else f'{delta:+.3f}'
    lines.append(f"| {comp['label']} | {b['successes']}/{b['trials']} | {c['successes']}/{c['trials']} | {b['maximum_impact_N']:.1f} / {c['maximum_impact_N']:.1f} | {delta} ({comp['common_successes']} pairs) |")
lines += ['', 'Each completed main pilot used 98,304 transitions; 196,608 total, plus a separate 512-transition plumbing smoke test and 15,360 optimized transitions in the interrupted balanced run. These are small initialization pilots, not a converged large-scale RL result.', '',
    'Training reward alone does not qualify a checkpoint. The initial unseen bank becomes a known regression bank for Pilot 2. Only its new fresh bank remains untouched by the first pilot analysis.', '',
    '## Measurement and limits','']
fresh=next(c for c in comparisons if c['candidate']=='fresh2')
lines[2:2]=['The fresh bank has '+str(summaries['fresh2']['successes'])+'/'+str(summaries['fresh2']['trials'])+' PPO successes, versus '+str(summaries['fresh0']['successes'])+'/'+str(summaries['fresh0']['trials'])+' for the baseline. Gained cases: '+str(fresh['gained_cases'])+'; lost cases: '+str(fresh['lost_cases'])+'. Holding-torque changes are conditional on the '+str(fresh['common_successes'])+' matched successful attempts. This is evidence of adaptation, not established reliability superiority.','']
lines += ['- '+s for s in report['limitations']]
lines += ['', 'Detailed per-run counts, depth-error checks, timing, checkpoint hashes, and successful-case paired comparisons are in `results.json`. The archived source and exact banks accompany every run.', '']
(out/'report.md').write_text('\n'.join(lines))

if a.skip_video:
    print(json.dumps(dict(comparisons=comparisons,summaries=summaries),indent=2)); raise SystemExit

model=json.loads((root/'runs/g1_kinematics_geometry_04/model.json').read_text())


def draw_robot(ax,run,frame,case):
    z=run['trace']; positions=z['body'][frame,case,:,:3]-np.asarray(run['viz']['origins'][case])
    lookup={name:positions[i] for i,name in enumerate(run['viz']['body_names'])}
    for j in model['joints']:
        if j['body0'] and j['body0'][0] in lookup and j['body1'][0] in lookup:
            points=np.array([lookup[j['body0'][0]],lookup[j['body1'][0]]]); ax.plot(*points.T,color='#263b50',lw=2)
    ax.scatter(*lookup['pelvis'],s=70,color='#f59e0b')
    g=run['bank'][run['viz']['geometry_indices'][case]]
    yaw=g['yaw']; R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    xy=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[g['depth'],g['width']]
    xy=xy@R.T+[g['x'],g['y']]
    ax.add_collection3d(Poly3DCollection([np.c_[xy,np.full(4,g['height'])]],facecolor='#10b981',alpha=.35,edgecolor='#047857'))
    if g['family'] in ('chair','rigid_couch'):
        back=np.array([[-g['depth']/2,-g['width']/2],[-g['depth']/2,g['width']/2]])@R.T+[g['x'],g['y']]
        wall=np.array([[*back[0],g['height']],[*back[1],g['height']],[*back[1],g['height']+.35],[*back[0],g['height']+.35]])
        ax.add_collection3d(Poly3DCollection([wall],facecolor='#10b981',alpha=.15))
    ax.set(xlim=(-1.3,.5),ylim=(-.8,.8),zlim=(0,1.5),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_xticks([-1.,-.5,0.,.5]); ax.set_yticks([-.5,0.,.5]); ax.set_zticks([0.,.5,1.,1.5])
    ax.set_box_aspect((1.8,1.6,1.5)); ax.view_init(elev=17,azim=-66)


# Include the first regression explicitly. For fresh data show the first lost
# success if present; otherwise the first failure, then a matched success.
freshcomp=next(c for c in comparisons if c['candidate']=='fresh2')
fresh_failure=(freshcomp['lost_cases'] or summaries['fresh2']['failed_cases'] or freshcomp['gained_cases'] or [0])[0]
fresh_selection=('first lost success' if freshcomp['lost_cases'] else 'first failed attempt' if summaries['fresh2']['failed_cases'] else 'first recovered baseline failure' if freshcomp['gained_cases'] else 'first matched successful attempt')
fresh_success=next((i for i,(b,c) in enumerate(zip(runs['fresh0']['result']['episodes'],runs['fresh2']['result']['episodes'])) if b['success'] and c['success'] and i!=fresh_failure),0)
clips=[('regression0','regression1',2,'First pilot: previously unseen bench'),
       ('regression2base','regression2',2,'Second pilot: same bench regression test'),
       ('fresh0','fresh2',fresh_failure,'Second pilot: fresh bank, '+fresh_selection),
       ('fresh0','fresh2',fresh_success,'Second pilot: next matched successful fresh case')]
(out/'clip_selection.json').write_text(json.dumps(clips,indent=2)+'\n')
labels=dict(regression0='Frozen baseline',regression2base='Frozen baseline',regression1='PPO pilot 1',regression2='PPO pilot 2',fresh0='Frozen baseline',fresh2='PPO pilot 2')
fig=plt.figure(figsize=(14,9)); writer=FFMpegWriter(fps=10,codec='libx264',bitrate=1800)
with writer.saving(fig,str(out/'sitting_comparison.mp4'),100):
    for ci,(left,right,case,caption) in enumerate(clips):
        load_limit=max(2.5,max(float(runs[k]['trace'][v][:,case].max()) for k in (left,right) for v in ('support','foot'))+.1)
        for frame in range(0,200,2):
            fig.clear()
            grid=fig.add_gridspec(2,2,height_ratios=[1.35,1.],hspace=.4,wspace=.16,top=.84,bottom=.08,left=.06,right=.98)
            for col,key in enumerate((left,right)):
                run=runs[key]; z=run['trace']; t=float(z['t'][frame]); row=run['result']['episodes'][case]
                ax=fig.add_subplot(grid[0,col],projection='3d'); draw_robot(ax,run,frame,case)
                status='PASS' if row['success'] else 'FAIL: '+', '.join(k for k,v in row['failures'].items() if v)
                ax.set_title(f'{labels[key]} | case {case} | t={t:.1f}s\nWhole-trial result: {status}',fontsize=10)
                ax=fig.add_subplot(grid[1,col])
                ax.plot(z['t'],z['support'][:,case],color='#047857',label='Seat load / body weight')
                ax.plot(z['t'],z['foot'][:,case],color='#2563eb',label='Foot load / body weight')
                ax.axvline(t,color='black',lw=1); ax.axhline(.75,color='#047857',ls=':',alpha=.5)
                ax.set(xlim=(0,20),ylim=(0,load_limit),xlabel='Time (s)',ylabel='Load fraction'); ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper right')
                live=z['live'][frame,case]; depth=f'{live[0]*100:.1f} cm' if live[8] else 'unavailable'
                ax.set_title(f'Leg torque RMS: {z["torque"][frame,case]:.1f} Nm | measured clearance: {depth}',fontsize=10)
            fig.suptitle(caption+'\nSimulator pose/force reconstruction for evaluation only | 2x playback',fontsize=12)
            if frame==130:fig.savefig(out/f'case_{ci}.png',dpi=120)
            writer.grab_frame()
plt.close(fig)
print(json.dumps(dict(comparisons=comparisons,summaries=summaries),indent=2))
