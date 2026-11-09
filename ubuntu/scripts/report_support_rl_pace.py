"""Full-horizon paired report for bounded learned progression PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from humanoid_sitting.support_pace import migrate_state

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--skip-video',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception')
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'report_source.py')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
banks=['easy','wide','fresh'];titles=['Known easier seats','Known wider seats','New wider holdout']
names={f'{bank}_{which}':f'support_rl_pace_{bank}_{which}_01' for bank in banks for which in ['prior','candidate']}
runs={}
for key,name in names.items():
    path=store/name
    runs[key]=dict(path=path,result=read(path/'results.json'),manifest=read(path/'manifest.json'),
        bank=read(path/'geometry_bank.json'),trace=dict(np.load(path/'rollout.npz')),
        viz=read(path/'visualization_metadata.json'))

def audit(path):
    m=read(path/'manifest.json');verified=0
    for key,value in m['hashes'].items():
        assert digest(root/key)==value,(path,key,'current')
        if key.endswith('.py'):assert digest(path/'source'/Path(key).name)==value,(path,key,'archive')
        verified+=1
    if 'placement_validation' in m:
        v=m['placement_validation']
        assert digest(Path(v['mixed_hard_source']))==v['source_sha256']
        assert digest(path/'source/prior_training_geometry_bank.json')==v['source_sha256']
        verified+=1
    return verified

def summary(r):
    rows=r['result']['episodes'];z=r['trace'];live=z['live'];accepted=live[...,8]>0
    assert len(rows)==16 and all(x['duration_s']==20 and x['steps']==1000 for x in rows)
    assert len(z['t'])==200 and np.allclose(z['t'],np.arange(1,201)*.1)
    assert all(np.isfinite(v).all() for v in z.values())
    err=abs(live[...,0]-z['true_clearance'])*100
    rate=z['command_pace'];assert rate.min()>=.25 and rate.max()<=1.
    return dict(successes=sum(x['success'] for x in rows),trials=16,
        failed_cases=[x['env'] for x in rows if not x['success']],
        failures_within_first_100ms=int((~z['alive'][0]).sum()),
        worst_impact_N=max(x['peak_force_N'] for x in rows),median_impact_N=float(np.median([x['peak_force_N'] for x in rows])),
        failures={k:sum(x['failures'][k] for x in rows) for k in rows[0]['failures']},
        missed_standing_checks=sum(not x['standing'] for x in rows),
        insufficient_supported_hold=sum(x['hold_s']<3. for x in rows),
        failed_without_physical_flag=sum(not x['success'] and not any(x['failures'].values()) for x in rows),
        accepted_depth_samples=int(accepted.sum()),depth_samples=int(accepted.size),
        max_accepted_clearance_error_cm=float(err[accepted].max()) if accepted.any() else None,
        accepted_errors_above_2cm=int(((err>2)&accepted).sum()),
        progression_rate_min=float(rate.min()),progression_rate_median=float(np.median(rate)),
        progression_fraction_below_90pct=float((rate<.9).mean()),
        joint_correction_p95_deg=float(np.rad2deg(np.quantile(abs(z['correction']),.95)*.08)),
        elapsed_s=r['result']['elapsed_s'])

summaries={k:summary(r) for k,r in runs.items()};comparisons={}
for bank in banks:
    b,c=runs[bank+'_prior'],runs[bank+'_candidate'];br,cr=b['result']['episodes'],c['result']['episodes']
    assert b['bank']==c['bank'] and b['manifest']['arguments']['seed']==c['manifest']['arguments']['seed']
    assert b['manifest']['hashes']==c['manifest']['hashes']
    assert np.all(b['trace']['command_pace']==1.)
    common=[i for i in range(16) if br[i]['success'] and cr[i]['success']]
    bt=float(np.mean([br[i]['seated_leg_rms_Nm'] for i in common])) if common else None
    ct=float(np.mean([cr[i]['seated_leg_rms_Nm'] for i in common])) if common else None
    comparisons[bank]=dict(matched_successes=len(common),
        gained_cases=[i for i in range(16) if not br[i]['success'] and cr[i]['success']],
        lost_cases=[i for i in range(16) if br[i]['success'] and not cr[i]['success']],
        prior_leg_rms_Nm=bt,candidate_leg_rms_Nm=ct,leg_torque_change_pct=100*(ct/bt-1) if bt else None)

training={};sources={name:audit(store/name) for name in names.values()}
parent=store/'support_rl_independent_train_04/final.pt'
for stage,steps in [('easy',65536),('mixed',131072)]:
    path=store/f'support_rl_pace_{stage}_train_{"01" if stage=="easy" else "02"}';sources[path.name]=audit(path)
    result=read(path/'results.json');account=read(path/'attempt_accounting.json');v=read(path/'verification.json')
    assert account['completed_transitions']+account['unfinished_transitions']==v['transitions']==steps
    for env in range(16):
        assert sum(r['steps'] for r in result['episodes']+account['unfinished_attempts'] if r['env']==env)==steps//16
    assert v['finite'] and v['max_parameter_change']>0 and v['optimizer_contains_actor']
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
    previous=torch.load(parent,map_location='cpu',weights_only=False)
    final=torch.load(path/'final.pt',map_location='cpu',weights_only=False)
    for group,actor in [('actor_state_dict',True),('critic_state_dict',False)]:
        expected=migrate_state(previous[group],initial[group],actor) if stage=='easy' else previous[group]
        for key,value in expected.items():
            assert torch.equal(value,initial[group][key]),(stage,group,key,'initialization')
            assert torch.isfinite(final[group][key]).all(),(stage,group,key,'final')
    pace_delta=max(float((final['actor_state_dict'][k][-1]-initial['actor_state_dict'][k][-1]).abs().max()) for k in ('mlp.4.weight','mlp.4.bias'))
    assert pace_delta>0.,(stage,'pace head did not learn')
    training[stage]=dict(path=str(path),transitions=steps,verification=v,accounting=account,
        pace_head_max_parameter_change=pace_delta,
        final_pace_action_std=float(final['actor_state_dict']['distribution.log_std_param'][-1].exp()),
        complete_attempts=len(result['episodes']),successes=result['successes'],
        refusals=sum(not r['startup_ready'] for r in result['episodes']),elapsed_s=result['elapsed_s'],
        failure_flags={k:sum(r['failures'][k] for r in result['episodes']) for k in result['episodes'][0]['failures']},
        failed_without_physical_flag=sum(not r['success'] and not any(r['failures'].values()) for r in result['episodes']),
        missed_standing_checks=sum(not r['standing'] for r in result['episodes']),
        final_checkpoint_sha256=digest(path/'final.pt'),parent_sha256=digest(parent),
        initial_weights_verified=True,final_actor_and_critic_finite=True)
    parent=path/'final.pt'

placement=store/'support_rl_pace_placement_probe_02';sources[placement.name]=audit(placement)
assert read(placement/'placement_probe.json')['passed']
placement_records=[json.loads(line) for line in (placement/'completed_attempts.jsonl').read_text().splitlines()]
# step() resets failed training rows; inspect the pre-reset latched records too.
assert not any(any(v for k,v in r['failures'].items() if k!='perception_unavailable')
               for r in placement_records if r['steps']==1)

probe=store/'support_rl_pace_probe_01';sources[probe.name]=audit(probe)
assert read(probe/'reset_probe.json')['passed'] and read(probe/'pace_probe.json')['passed']
oldknown={}
for bank,oldname in [('easy','support_rl_independent_regression_candidate_04'),('wide','support_rl_independent_fresh_candidate_04')]:
    old=read(store/oldname/'results.json')['episodes'];new=runs[bank+'_prior']['result']['episodes']
    oldtrace=dict(np.load(store/oldname/'rollout.npz'));newtrace=runs[bank+'_prior']['trace']
    exact={k:bool(np.array_equal(v,newtrace[k])) for k,v in oldtrace.items()}
    oldknown[bank]=dict(shared_trace_arrays_bitwise_equal=exact,
        max_body_position_difference_m=float(abs(oldtrace['body'][...,:3]-newtrace['body'][...,:3]).max()),
        old_successes=sum(r['success'] for r in old),new_successes=sum(r['success'] for r in new),
        success_flags_equal=[r['success'] for r in old]==[r['success'] for r in new],
        max_peak_force_difference_N=max(abs(x['peak_force_N']-y['peak_force_N']) for x,y in zip(old,new)))
report=dict(runs=names,summaries=summaries,comparisons=comparisons,training=training,
    total_new_transitions=196608,reset_probe=read(probe/'reset_probe.json'),pace_probe=read(probe/'pace_probe.json'),
    neutral_regression_reproduction=oldknown,placement_probe=read(placement/'placement_probe.json'),
    excluded_debug_run=read(store/'support_rl_pace_mixed_train_01/interruption.json'),
    promotion='Experimental checkpoint only; original deployment default remains unchanged.',
    limitations=['8 geometries with two starts per bank; insufficient to establish arbitrary-surface reliability.',
        'Rigid initially aligned seats; no compliant cushions, navigation or hardware validation.',
        'Pace changes recurrent progression, not guaranteed physical vertical speed.',
        'Training ends at first failure; all evaluation attempts continue for 20 seconds with failures latched.',
        'No privileged actor/critic inputs. True poses, forces and velocity are reward/evaluation labels only.',
        'Pace saturation is asymmetric: positive requests cannot accelerate the scaffold.',
        'This pilot jointly changes action representation, exploration, curriculum and budget; it does not isolate causal contributions.',
        'CSV landing margins describe the pelvis center inside the true rectangle, not full contact-area coverage. Landing velocity is a 10 Hz finite difference, not an instantaneous sensor reading.'])
# Evaluation-only diagnostic timing. At 10 Hz, event times are approximate;
# the full-trial force metric is independent of this subsampled trace.
import csv
cases=[]
for key,r in runs.items():
    z=r['trace']
    for row in r['result']['episodes']:
        i=row['env'];g=r['bank'][row['geometry']]
        failed=np.flatnonzero(~z['alive'][:,i]);touch=np.flatnonzero(z['support'][:,i]>.25)
        contact_geometry={k:None for k in ('support_pelvis_x_cm','support_pelvis_y_cm',
            'support_pelvis_edge_margin_cm','support_pelvis_height_above_top_cm','approx_support_vz_mps')}
        if len(touch):
            frame=int(touch[0]);position=z['root'][frame,i,:3]-np.asarray(r['viz']['origins'][i])
            xy=position[:2]-[g['x'],g['y']];c,s=np.cos(g['yaw']),np.sin(g['yaw'])
            local=np.array([c*xy[0]+s*xy[1],-s*xy[0]+c*xy[1]])
            contact_geometry.update(support_pelvis_x_cm=float(local[0]*100),support_pelvis_y_cm=float(local[1]*100),
                support_pelvis_edge_margin_cm=float(min(g['depth']/2-abs(local[0]),g['width']/2-abs(local[1]))*100),
                support_pelvis_height_above_top_cm=float((position[2]-g['height'])*100),
                approx_support_vz_mps=float((z['root'][frame,i,2]-z['root'][frame-1,i,2])/.1) if frame else None)
        cases.append(dict(run=key,case=i,**contact_geometry,family=g['family'],height_cm=g['height']*100,
            depth_cm=g['depth']*100,success=row['success'],peak_force_N=row['peak_force_N'],
            first_latched_failure_s=float(z['t'][failed[0]]) if len(failed) else None,
            first_quarter_weight_support_s=float(z['t'][touch[0]]) if len(touch) else None,
            minimum_pace=float(z['command_pace'][:,i].min()),
            continuous_supported_hold_s=row['hold_s'],standing_check_passed=row['standing'],
            final_support_fraction=float(z['support'][-1,i]),
            failure_flags=','.join(k for k,v in row['failures'].items() if v)))
with (out/'case_diagnostics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
report['case_diagnostic_times']='Approximate 10 Hz evaluation-only events; full-trial force peaks use 50 Hz checks.'
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'bundle_audit.json').write_text(json.dumps(dict(passed=True,sources=sources,
    full_evaluation_attempts=96,case_frames=19200,transition_accounting=True,checkpoint_continuity=True),indent=2)+'\n')
colors={'prior':'#c2410c','candidate':'#047857'};labels={'prior':'Previous PPO','candidate':'PPO + pace'}
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for col,(bank,title) in enumerate(zip(banks,titles)):
    for j,which in enumerate(['prior','candidate']):
        s=summaries[f'{bank}_{which}'];axes[0,col].bar(j,s['successes'],color=colors[which])
        axes[0,col].text(j,s['successes']+.3,f"{s['successes']}/16",ha='center')
        axes[1,col].bar(j,s['worst_impact_N'],color=colors[which])
    axes[0,col].set(title=title,ylim=(0,18),ylabel='Successful full attempts')
    axes[1,col].axhline(632.527832,color='#b91c1c',ls='--',label='Impact threshold')
    axes[1,col].set(ylabel='Worst impact, all attempts (N)');axes[1,col].legend(fontsize=8)
    for ax in axes[:,col]:ax.set_xticks([0,1],list(labels.values()));ax.grid(axis='y',alpha=.2)
fig.suptitle('Learned motion pace: paired full 20-second evaluations',fontsize=15)
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,bank,title in zip(axes,banks,titles):
    z=runs[bank+'_candidate']['trace']
    for case in range(16):ax.plot(z['t'],z['command_pace'][:,case],alpha=.55,lw=1)
    ax.set(title=title,xlabel='Time (s)',ylabel='Motion pace setting (1 = original)',ylim=(.2,1.03));ax.grid(alpha=.2)
fig.suptitle('Learned motion timing · one line per attempt\nThis setting changes the motion network; it is not measured vertical speed')
fig.savefig(out/'learned_pace.png',dpi=150);plt.close(fig)
lines=['# Learned pace PPO pilot','',report['promotion'],'',
    '| Bank | Previous success | Pace success | Previous worst impact | Pace worst impact |','|---|---:|---:|---:|---:|']
for bank,title in zip(banks,titles):
    b,c=summaries[bank+'_prior'],summaries[bank+'_candidate']
    lines.append(f"| {title} | {b['successes']}/16 | {c['successes']}/16 | {b['worst_impact_N']:.1f} N | {c['worst_impact_N']:.1f} N |")
lines+=['', '196,608 new transitions: 65,536 on the easier bank, then 131,072 on the mixed bank. Final checkpoint only.',
    'Before any holdout evaluation, mixed stage 01 was interrupted: a box overlapped the initial standing robot. Stage 02 restarts from the completed easy checkpoint and uses the first eight prior TRAINING-bank wider prototypes, selected in original order rather than by success. A 192-start placement probe passes. The interrupted run is preserved and excluded.',
    'Six paired full-horizon runs contain 96 attempts and 19,200 recorded case-frames. Refusals count as failures.',
    'The actor/critic use 145 sensor-derived channels, including the observable pace filter. Four new unique tests plus the previous twelve pass; discovery executes twenty tests because four admission tests are imported twice.',
    'Simulator probes confirm cold reset behavior, isolation of other robots, fresh camera admission and per-robot pace reset.','']
for bank,title in zip(banks,titles):
    d=comparisons[bank];change=d['leg_torque_change_pct'];change='N/A' if change is None else f'{change:+.2f}%'
    lines.append(f"- {title}: gained {d['gained_cases']}, lost {d['lost_cases']}; matched-success leg torque {change} ({d['matched_successes']} pairs).")
lines+=['','A full-trial force peak uses all 50 Hz checks. Video curves use 10 Hz samples and may miss spikes. Robot poses/forces in the reconstruction are evaluation truth, never policy input.','']+['- '+s for s in report['limitations']]
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('PACE_REPORT',json.dumps(dict(summaries=summaries,comparisons=comparisons)),flush=True)
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

comp=comparisons['fresh']
first=(comp['lost_cases'] or summaries['fresh_candidate']['failed_cases'] or [0])[0]
other=[i for i,r in enumerate(runs['fresh_candidate']['result']['episodes']) if r['success'] and i!=first]
second=(comp['gained_cases'] or other or [i for i in range(16) if i!=first])[0]
clips=[first,second];(out/'clip_selection.json').write_text(json.dumps(dict(bank='fresh',cases=clips,
    selection='first lost/failed case, then first gained/successful other case'),indent=2)+'\n')
fig=plt.figure(figsize=(13,8));writer=FFMpegWriter(fps=10,codec='libx264',bitrate=2000)
with writer.saving(fig,str(out/'sitting_comparison.mp4'),100):
    for clip,case in enumerate(clips):
        for frame in range(0,200,2):
            fig.clear();grid=fig.add_gridspec(2,2,height_ratios=[1.4,1],hspace=.35,wspace=.3,top=.82,bottom=.08,left=.07,right=.95)
            for col,which in enumerate(['prior','candidate']):
                run=runs['fresh_'+which];z=run['trace'];row=run['result']['episodes'][case];t=float(z['t'][frame])
                ax=fig.add_subplot(grid[0,col],projection='3d');draw_robot(ax,run,frame,case)
                reasons=[k for k,v in row['failures'].items() if v]
                if not row['standing']:reasons.append('standing check')
                if row['hold_s']<3.:reasons.append('short supported hold')
                status='PASS' if row['success'] else 'FAIL: '+', '.join(reasons)
                ax.set_title(f'{labels[which]} | case {case} | {t:.1f}s\n{status}',fontsize=10)
                ax=fig.add_subplot(grid[1,col]);ax.plot(z['t'],z['force'][:,case],color=colors[which],label='Sampled force')
                ax.axhline(632.527832,color='#b91c1c',ls='--',label='Impact threshold');ax.axvline(t,color='black',lw=1)
                ax.axhline(row['peak_force_N'],color=colors[which],ls=':',label='Full-trial peak (50 Hz)')
                ymax=max(750,max(runs['fresh_'+w]['result']['episodes'][case]['peak_force_N'] for w in labels)*1.08)
                ax.set(xlim=(0,20),ylim=(0,ymax),xlabel='Time (s)',ylabel='Seat force (N)');ax.grid(alpha=.2)
                live=z['live'][frame,case];clearance=f'{live[0]*100:.1f} cm' if live[8] else 'unavailable'
                ax.set_title(f'Measured clearance: {clearance} | motion pace: {z["command_pace"][frame,case]:.2f}\nLeg torque: {z["torque"][frame,case]:.1f} Nm | full peak: {row["peak_force_N"]:.0f} N',fontsize=10)
                if col==0:ax.legend(fontsize=8)
            fig.suptitle('New wider holdout: previous PPO versus learned motion pace\nSimulator reconstruction for evaluation only · 2x playback\nForce curve sampled at 10 Hz; dotted line shows full-trial peak',fontsize=12)
            if frame==130:fig.savefig(out/f'case_{clip}.png',dpi=120)
            writer.grab_frame()
plt.close(fig)
subprocess.run(['ffmpeg','-v','error','-i',str(out/'sitting_comparison.mp4'),'-f','null','-'],check=True)
