"""Independent read-only audit and additive final review of bounded pilot 02."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/kadir/humanoid-sitting')
STUDY=Path('/data/humanoid-sitting-perception/resting_bounded_ppo_02')
OUT=STUDY/'verified_review'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(values):return float(np.mean(values))
def equal_archive(a,b):
    with np.load(a) as x,np.load(b) as y:
        assert set(x.files)==set(y.files)
        assert all(np.array_equal(x[k],y[k],equal_nan=True) for k in x.files),(a,b)

status=read(STUDY/'status.json');assert status['phase']=='complete'
protocol=read(STUDY/'protocol.json');original=read(STUDY/'results/results.json')
manifest=read(STUDY/'frozen/manifest.json')
assert all(sha(Path(p))==value for p,value in manifest.items())
assert len(manifest)==46
assert protocol['seeds']==[53111,53117] and protocol['updates']==64
deployment=ast.parse((ROOT/'sitting_task/src/humanoid_sitting/deployment.py').read_text())
controlled=None
for node in deployment.body:
    if isinstance(node,(ast.Assign,ast.AnnAssign)):
        names=node.targets if isinstance(node,ast.Assign) else [node.target]
        if any(isinstance(x,ast.Name) and x.id=='BODY_JOINT_NAMES' for x in names):controlled=ast.literal_eval(node.value)
assert controlled is not None and len(controlled)==23
banks={'starts':16,'heights':12,'impulse':12,'novel':16}
variants=('ppo','ppo_hold')
fixed={};episodes={};budgets=[];audits=[];hold=[]
capture=read(STUDY/'novel_bank/capture.json');assert capture['valid_count']==16
with np.load(STUDY/'novel_bank/candidates.npz') as z:assert z['valid'].all()
for bank,n in banks.items():
    path=STUDY/f'fixed_{bank}';fixed[bank]=read(path/'results.json')['episodes']
    assert len(fixed[bank])==n
    with np.load(path/'trace.npz') as z,np.load(path/'startup.npz') as start:
        assert z['command'].shape[0]==200
        assert np.array_equal(z['command'],np.broadcast_to(start['command'],z['command'].shape))
        assert np.all(z['alive'])
    assert all(r['success'] and r['steps']==1000 and r['duration_s']==20 and not any(r['failures'].values()) for r in fixed[bank])

for seed in protocol['seeds']:
    train=STUDY/f'train_{seed}';v=read(train/'verification.json');progress=read(train/'progress.json')
    assert v['updates']==progress['update']==64 and v['transitions']==65536
    assert v['active_transitions']+v['warmup_transitions']==65536
    training=[json.loads(x) for x in (train/'episodes.jsonl').read_text().splitlines()]
    assert sum(r['steps'] for r in training)+sum(v['unfinished_episode_steps'])==v['active_transitions']
    initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(train/'final.pt',map_location='cpu',weights_only=False)
    assert initial['actor_state_dict']['mlp.0.weight'].shape==(128,393)
    assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.weight'])==0
    assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.bias'])==0
    changes={role:max(float((x-final[role][k]).abs().max()) for k,x in initial[role].items()) for role in ('actor_state_dict','critic_state_dict')}
    assert all(x>0 for x in changes.values())
    assert all(torch.isfinite(x).all() for role in changes for x in final[role].values())
    budgets.append(dict(seed=seed,**v,changes=changes,final_sha256=sha(train/'final.pt'),
        completed_training_episodes=len(training),successful_training_episodes=sum(r['success'] for r in training)))
    for bank,n in banks.items():
        plain=STUDY/f'{bank}_{seed}_ppo';held=STUDY/f'{bank}_{seed}_ppo_hold'
        assert read(plain/'results.json')==read(held/'results.json')
        equal_archive(plain/'trace.npz',held/'trace.npz')
        for variant in variants:
            name=f'{bank}_{seed}_{variant}';path=STUDY/name
            rows=read(path/'results.json')['episodes'];episodes[(seed,variant,bank)]=rows
            assert len(rows)==n and [r['case'] for r in rows]==list(range(n))
            assert all(r['success'] and r['steps']==1000 and r['duration_s']==20 and r['good_fraction']>=.95 and not any(r['failures'].values()) for r in rows)
            equal_archive(path/'startup.npz',STUDY/f'fixed_{bank}/startup.npz')
            assert read(path/'geometry_bank.json')==read(STUDY/f'fixed_{bank}/geometry_bank.json')
            assert read(path/'disturbances.json')==read(STUDY/f'fixed_{bank}/disturbances.json')
            loaded=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(x,loaded[role][k]) for role in changes for k,x in final[role].items())
            sensor=read(path/'sensor_audit.json')
            assert sensor['actor_channels']==393 and not sensor['privileged_actor_inputs'] and not sensor['depth_errors']
            assert all(x['physics_unchanged'] for x in sensor['camera'])
            names=read(path/'results.json')['joint_names']
            spans=np.array([(.025 if 'roll_joint' in name and ('hip' in name or 'ankle' in name) else .05) if name in controlled else 0 for name in names])
            with np.load(path/'trace.npz') as z,np.load(path/'startup.npz') as start:
                assert z['command'].shape[0]==200 and np.allclose(z['t'],np.arange(1,201)/10)
                assert np.all(z['alive']) and np.all(z['good'][-20:])
                drift=np.abs(z['command']-start['command'][None])
                assert np.all(drift<=spans[None,None]+1e-6)
                assert np.all(drift[...,spans==0]==0)
                assert np.isfinite(z['torque']).all()
                torque_rms=np.sqrt(np.mean(z['torque']**2,axis=-1))
                metric_error=max(abs(float(torque_rms[:,i][z['good'][:,i]].mean())-r['metrics_on_good']['all_joint_rms_Nm']) for i,r in enumerate(rows))
                # The archived trace samples 10 Hz; official metrics integrate all 50 Hz ticks.
                assert metric_error<.025,(name,metric_error)
                max_by_joint=drift.max((0,1));top=np.argsort(max_by_joint)[-5:][::-1]
                audits.append(dict(run=name,full_horizon=True,startup_bitwise_equal=True,checkpoint_equal=True,
                    max_command_offset_rad=float(drift.max()),max_command_offset_deg=float(np.rad2deg(drift.max())),
                    bounded=True,uncontrolled_commands_fixed=True,torque_rms_10hz_vs_50hz_max_error_Nm=metric_error,
                    largest_joint_offsets=[dict(joint=names[i],rad=float(max_by_joint[i])) for i in top]))
            gate=read(path/'hold_goal.json')
            assert gate['enabled']==(variant=='ppo_hold') and sum(gate['entries'])==sum(gate['releases'])==0
            assert not any(any(row['held']) for row in gate['trace'])
            if gate['enabled']:
                ratio=np.array([x['effort'] for x in gate['trace']])/np.array([x['reference_effort'] for x in gate['trace']])
                assert (ratio>.9).all()
                hold.append(dict(seed=seed,bank=bank,entries=0,releases=0,minimum_recorded_effort_ratio=float(ratio.min()),
                    recorded_samples_meeting_effort_threshold=int((ratio<=.9).sum()),bitwise_same_trace_and_results_as_plain=True,
                    note='No hold episodes: physical validity or benefit after a hold entry cannot be assessed. 10 Hz samples; gate counters record all 50 Hz steps.'))

comparisons=[];heights=[]
keys=('normalized_torque_squared','all_joint_rms_Nm','leg_rms_Nm','torso_arm_hand_rms_Nm','foot_load_bw','rest_normal_load_bw')
def compare(seed,bank,a,b):
    assert len(a)==len(b) and all(x['success'] and y['success'] for x,y in zip(a,b))
    metrics={}
    for k in keys:
        x=mean([r['metrics_on_good'][k] for r in a]);y=mean([r['metrics_on_good'][k] for r in b])
        metrics[k]=dict(fixed=x,ppo=y,percent_change=100*(y/x-1))
    return dict(seed=seed,bank=bank,total=len(a),fixed_successes=len(a),ppo_successes=len(b),metrics=metrics,
        cases_10percent_saving=sum(y['metrics_on_good']['normalized_torque_squared']<=.9*x['metrics_on_good']['normalized_torque_squared'] for x,y in zip(a,b)))
for seed in protocol['seeds']:
    for bank in (*banks,'known','unperturbed','all'):
        include=tuple(banks) if bank=='all' else ('starts','heights','impulse') if bank=='known' else ('starts','heights','novel') if bank=='unperturbed' else (bank,)
        row=compare(seed,bank,sum((fixed[b] for b in include),[]),sum((episodes[(seed,'ppo',b)] for b in include),[]))
        prior=next(r for r in original['comparisons'] if r['seed']==seed and r['bank']==bank)
        for k in keys:assert abs(row['metrics'][k]['percent_change']-prior['ppo_vs_fixed']['metrics'][k]['percent_change'])<1e-10
        comparisons.append(row)
    for bank in banks:
        for h in sorted({r['height_m'] for r in fixed[bank]}):
            rows=compare(seed,bank,[r for r in fixed[bank] if r['height_m']==h],
                         [r for r in episodes[(seed,'ppo',bank)] if r['height_m']==h])
            heights.append(dict(height_m=h,**rows))
assert sum(v['transitions'] for v in budgets)==131072
audit=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_utc=status['utc'],
    source_hashes_verified=len(manifest),budgets=budgets,startup_and_checkpoint_and_command_checks=audits,hold_checks=hold,
    baseline_successes=56,conditions=56,unique_states=44,impulse_repeats=12,new_holdout_states=16,
    all_evaluation_failures_zero=True,comparisons= comparisons,by_height=heights,
    outcome='Both bounded PPO policies retain 56/56 stable trials, but neither reduces normalized squared torque. Hold never enters and therefore has no effect. No policy promoted.',
    notes=['All reported effort means use qualifying ticks of successful trials; trace reconstruction samples at10Hz versus50Hz official metrics.',
           'Two seeds reuse the same44 states;12 additional conditions repeat states under impulses.',
           'Normalized squared torque is an effort proxy, not energy; Nm averages instantaneous all-joint RMS.',
           'Older-study comparison is not an isolated ablation: action/observation contract, seeds and budgets differ.'])
OUT.mkdir(exist_ok=False)
(OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
shutil.copy2(Path(__file__),OUT/'verifier.py')
shutil.copy2(STUDY/'frozen/manifest.json',OUT/'frozen_manifest.json')
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
colors={53111:'#247b93',53117:'#b86135'}
for seed in protocol['seeds']:
    rs=sorted([r for r in heights if r['seed']==seed and r['bank']!='impulse'],key=lambda r:r['height_m'])
    x=[r['height_m']*100 for r in rs]
    for ax,key in ((axes[0,0],'normalized_torque_squared'),(axes[0,1],'all_joint_rms_Nm')):
        ax.plot(x,[r['metrics'][key]['percent_change'] for r in rs],'-o',label=f'Seed {seed}',color=colors[seed])
        ax.axhline(0,color='#555',linewidth=.8)
        ax.set_xlabel('Seat height (cm)');ax.set_ylabel('Change from matched fixed posture (%)');ax.legend()
axes[0,0].axhline(-10,color='#55815d',linestyle='--',label='10% reduction target');axes[0,0].legend()
axes[0,0].set_title('Normalized squared torque: worse at every height')
axes[0,1].set_title('Whole-body RMS torque: small, mixed changes')
labels=['Original\nheights','Intermediate\nheights','Impulse\nrepeats','New\nheights'];positions=np.arange(4)
for i,seed in enumerate(protocol['seeds']):
    rs=[r for b in banks for r in comparisons if r['seed']==seed and r['bank']==b]
    bars=axes[1,0].bar(positions+(i-.5)*.34,[r['metrics']['normalized_torque_squared']['percent_change'] for r in rs],width=.32,label=f'Seed {seed}',color=colors[seed])
    axes[1,0].bar_label(bars,fmt='%+.2f',fontsize=8,padding=3)
axes[1,0].set_xticks(positions,labels);axes[1,0].set(ylabel='Normalized squared torque change (%)',title='No saving on the fresh holdout or disturbances',ylim=(0,3.1));axes[1,0].legend()
ax=axes[1,1];ax.axis('off')
ax.text(0,1,'Physical stability retained',fontsize=19,weight='bold',va='top')
ax.text(0,.86,'Fixed: 56/56\nPPO seed 53111: 56/56\nPPO seed 53117: 56/56\nPPO + hold: identical to PPO\n\n0 physical failure flags\n0 hold entries; 0 trials reach the 10% saving target\n\n44 unique states + 12 impulse repeats\n20 seconds per trial; rigid level seats only',fontsize=13,va='top',linespacing=1.6)
for ax in axes.flat[:3]:ax.grid(alpha=.18);ax.set_axisbelow(True)
fig.savefig(OUT/'effort_by_height.png',dpi=170);plt.close(fig)
print(json.dumps(dict(outcome=audit['outcome'],budgets=budgets,all_bank=[r for r in comparisons if r['bank']=='all'],
    max_command_offsets={s:max(a['max_command_offset_deg'] for a in audits if str(s) in a['run']) for s in protocol['seeds']}),indent=2))
