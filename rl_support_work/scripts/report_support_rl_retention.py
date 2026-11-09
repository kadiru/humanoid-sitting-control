"""Audit the matched retention experiment and render its outcomes."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];STORE=Path('/data/humanoid-sitting-perception')
OUT=STORE/'support_rl_retention_results_01';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent_path=STORE/'support_rl_independent_train_04/final.pt'
parent=torch.load(parent_path,map_location='cpu',weights_only=False)
assert sha(parent_path)=='2c0f6907196be68bce9a1ee2e6b6ef6e4c82bf0ecdff622dc55b81fe8ef58d13'
assert (ROOT/'logs/support_rl_retention_01.exit').read_text().strip()=='0'
shutil.copy2(__file__,OUT/'report_source.py')
for name in ['SUPPORT_RL_RETENTION_PROTOCOL.md']:
    shutil.copy2(ROOT/'docs'/name,OUT/name)
for name in ['run_support_rl_retention_01.sh','check_support_rl_retention.py','test_support_retention.py']:
    shutil.copy2(ROOT/'scripts'/name,OUT/name)
shutil.copy2(ROOT/'logs/support_rl_retention_tests_01.log',OUT/'unit_tests.log')

def audit_source(path):
    m=read(path/'manifest.json')
    for key,value in m['hashes'].items():
        assert sha(ROOT/key)==value,(path,key)
        if key.endswith('.py'):assert sha(path/'source'/Path(key).name)==value
    assert m['retention']['parent_sha256']==sha(parent_path)
    assert sha(path/'source/rsl_rl_ppo_original.py')==m['retention']['upstream_ppo_sha256']
    if m['retention']['training_bank']:
        bank=Path(m['retention']['training_bank'])
        assert sha(bank)==sha(path/'source/training_geometry_bank.json')==m['retention']['training_bank_sha256']
        assert read(path/'geometry_bank.json')==read(bank)
    return m

training={}
for arm,coefficient in [('control',0.),('anchored',1.)]:
    path=STORE/f'support_rl_retention_{arm}_train_01';m=audit_source(path)
    assert m['retention']['coefficient']==coefficient
    v=read(path/'verification.json');r=read(path/'results.json');account=read(path/'attempt_accounting.json')
    assert v['transitions']==32768==account['completed_transitions']+account['unfinished_transitions']
    assert v['finite'] and v['optimizer_contains_actor'] and v['max_parameter_change']>0
    for i in range(16):assert sum(x['steps'] for x in r['episodes']+account['unfinished_attempts'] if x['env']==i)==2048
    initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(path/'final.pt',map_location='cpu',weights_only=False)
    assert len(initial['optimizer_state_dict']['state'])==0
    for group in ['actor_state_dict','critic_state_dict']:
        for key,value in parent[group].items():
            assert torch.equal(value,initial[group][key]),(arm,key,'initial')
            assert torch.isfinite(final[group][key]).all(),(arm,key,'finite')
            if key.startswith('obs_normalizer.'):
                assert torch.equal(value,final[group][key]),(arm,key,'normalizer')
    updates=[json.loads(line) for line in (path/'retention_updates.jsonl').read_text().splitlines()]
    assert len(updates)==32 and all(np.isfinite(list(x.values())).all() for x in updates)
    action=read(path/'action_audit.json');assert action['sample_rows']==32768 and not action['pace_action']
    training[arm]=dict(transitions=32768,complete_attempts=len(r['episodes']),successes=r['successes'],
        elapsed_s=r['elapsed_s'],final_sha256=sha(path/'final.pt'),initial_equals_parent=True,
        frozen_normalizers_verified=True,fresh_optimizer_verified=True,updates=updates,
        actor_parameter_change=v['max_parameter_change'],action_audit=action,
        first_tick_physical_failures=sum(x['steps']==1 and any(v for k,v in x['failures'].items() if k!='perception_unavailable') for x in r['episodes']))

parity=read(STORE/'support_rl_retention_easy_parent_01/gate.json');assert parity['passed']
# Counterfactual actions on exactly the same real sensor observations. These
# known evaluation inputs are used only after training, never in the loss.
replay=np.load(STORE/'support_rl_retention_easy_parent_01/sensor_replay.npz')['policy']
assert replay.shape==(200,16,144)
obs=TensorDict({'policy':torch.from_numpy(replay.reshape(-1,144))},batch_size=[3200])
model=MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[128,128],obs_normalization=True,
    distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=.04,std_type='log'))
with torch.inference_mode():
    model.load_state_dict(parent['actor_state_dict']);ref_mean=model(obs).clone()
    model.distribution.update(ref_mean);ref_std=model.output_std.clone()
    for arm in ['control','anchored']:
        checkpoint=torch.load(STORE/f'support_rl_retention_{arm}_train_01/final.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['actor_state_dict']);mean=model(obs);model.distribution.update(mean)
        kl=model.get_kl_divergence((ref_mean,ref_std),model.output_distribution_params)
        drift=(mean.clamp(-1,1)-ref_mean.clamp(-1,1)).abs()*.08*180/np.pi
        training[arm]['parent_sensor_replay_action_drift']=dict(mean_kl=float(kl.mean()),
            correction_p95_deg=float(torch.quantile(drift,.95)),correction_max_deg=float(drift.max()),
            note='Offline same-input comparison on 3200 known parent sensor observations; no physics or learning, no command-slew simulation.')
runs={};summaries={};comparisons={};cases=[]
for bank in ['easy','wide','fresh']:
    if bank=='wide' and not any((STORE/f'support_rl_retention_wide_{arm}_01/results.json').exists() for arm in ['control','anchored']):
        continue
    for arm in ['parent','control','anchored']:
        path=STORE/f'support_rl_retention_{bank}_{arm}_01'
        if bank=='wide' and arm=='parent':path=STORE/'support_rl_pace_wide_prior_01'
        if not (path/'results.json').exists():continue
        if path.name.startswith('support_rl_retention'):audit_source(path)
        r=read(path/'results.json');rows=r['episodes'];z=dict(np.load(path/'rollout.npz'))
        assert len(rows)==16 and all(x['duration_s']==20 and x['steps']==1000 for x in rows)
        assert len(z['t'])==200 and np.allclose(z['t'],np.arange(1,201)*.1)
        assert all(np.isfinite(x).all() for x in z.values())
        key=bank+'_'+arm;runs[key]=path.name
        accepted=z['live'][...,8]>0;error=abs(z['live'][...,0]-z['true_clearance'])*100
        summaries[key]=dict(successes=sum(x['success'] for x in rows),trials=16,
            worst_impact_N=max(x['peak_force_N'] for x in rows),failed_cases=[x['env'] for x in rows if not x['success']],
            accepted_depth_frames=int(accepted.sum()),case_frames=int(accepted.size),
            max_accepted_clearance_error_cm=float(error[accepted].max()),accepted_errors_above_2cm=int(((error>2)&accepted).sum()),
            gate=read(path/'gate.json') if (path/'gate.json').exists() else None)
        for x in rows:
            i=x['env'];failure=np.flatnonzero(~z['alive'][:,i])
            cases.append(dict(bank=bank,arm=arm,case=i,success=x['success'],peak_force_N=x['peak_force_N'],
                hold_s=x['hold_s'],standing=x['standing'],seated_leg_rms_Nm=x['seated_leg_rms_Nm'],
                first_latched_failure_s=float(z['t'][failure[0]]) if len(failure) else None,
                failures=','.join(k for k,v in x['failures'].items() if v)))
    if bank+'_parent' not in runs:continue
    bp=STORE/runs[bank+'_parent'];br=read(bp/'results.json')['episodes'];bm=read(bp/'manifest.json')
    for arm in ['control','anchored']:
        key=bank+'_'+arm
        if key not in runs:continue
        cp=STORE/runs[key];cr=read(cp/'results.json')['episodes'];cm=read(cp/'manifest.json')
        assert read(bp/'geometry_bank.json')==read(cp/'geometry_bank.json')
        assert bm['arguments']['seed']==cm['arguments']['seed']
        initial=torch.load(cp/'initial.pt',map_location='cpu',weights_only=False)
        candidate=torch.load(STORE/f'support_rl_retention_{arm}_train_01/final.pt',map_location='cpu',weights_only=False)
        for group in ['actor_state_dict','critic_state_dict']:
            assert all(torch.equal(value,initial[group][k]) for k,value in candidate[group].items())
        common=[i for i,(a,b) in enumerate(zip(br,cr)) if a['success'] and b['success']]
        bt=float(np.mean([br[i]['seated_leg_rms_Nm'] for i in common])) if common else None
        ct=float(np.mean([cr[i]['seated_leg_rms_Nm'] for i in common])) if common else None
        comparisons[key]=dict(lost_cases=[i for i,(a,b) in enumerate(zip(br,cr)) if a['success'] and not b['success']],
            gained_cases=[i for i,(a,b) in enumerate(zip(br,cr)) if not a['success'] and b['success']],
            matched_successes=len(common),conditional_leg_torque_change_pct=100*(ct/bt-1) if bt else None)

# Original code and baseline audit: retain the old script's root-reserve failure,
# then independently check the output volume actually used for this experiment.
integrity_path=OUT/'preservation.json'
proc=subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),str(ROOT/'scripts/audit_shared_frame_integrity.py'),
    '--output',str(integrity_path)],capture_output=True,text=True)
(OUT/'legacy_preservation_check.log').write_text(proc.stdout+'\n'+proc.stderr)
integrity=read(integrity_path)
assert not integrity['original_source_changed'] and not integrity['archived_baseline_changed']
assert not integrity['current_checkpoint_changed'] and not integrity['isaaclab_status']
assert sha(STORE/'support_rl_pace_mixed_train_02/final.pt')=='b21fd9d2db493865ab49843cf427223d8d1baf505acdee1d3f346c18da72b873'
storage=dict(legacy_check_exit=proc.returncode,root_free_gib=shutil.disk_usage(ROOT).free/2**30,
             output_free_gib=shutil.disk_usage(STORE).free/2**30,output_volume=str(STORE))
assert storage['output_free_gib']>30
report=dict(training=training,runs=runs,summaries=summaries,comparisons=comparisons,
    parent_trace_parity=parity,total_training_transitions=65536,storage=storage,
    preservation_verified=True,new_full_evaluation_attempts=16*sum(n.startswith('support_rl_retention') for n in runs.values()),
    fresh_holdout_used=any(k.startswith('fresh_') for k in runs),deployment_default_changed=False,
    scope='Two matched single-seed continuations; known-bank gates, rigid prealigned seats only. Does not isolate freezing normalization from historical training or prove universal sitting.')
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'case_diagnostics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)

labels={'parent':'Preserved policy','control':'Continued PPO','anchored':'PPO + parent penalty'}
colors={'parent':'#64748b','control':'#c2410c','anchored':'#0f766e'}
fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
for j,arm in enumerate(labels):
    s=summaries['easy_'+arm]
    axes[0].bar(j,s['successes'],color=colors[arm]);axes[0].text(j,s['successes']+.3,f"{s['successes']}/16",ha='center')
    axes[1].bar(j,s['worst_impact_N'],color=colors[arm]);axes[1].text(j,s['worst_impact_N']+15,f"{s['worst_impact_N']:.0f} N",ha='center')
for arm in ['control','anchored']:
    axes[2].plot(np.arange(1,33),[x['parent_kl'] for x in training[arm]['updates']],label=labels[arm],color=colors[arm])
axes[0].set(ylim=(0,18),ylabel='Complete successful attempts',title='Known easier seats: retention gate')
axes[1].set(ylabel='Worst full-trial seat impact (N)',title='All 16 attempts, including failures')
axes[1].axhline(632.527832,color='#b91c1c',ls='--',label='Existing impact threshold');axes[1].legend(fontsize=8)
axes[2].set(xlabel='PPO update',ylabel='Mean KL from parent',title='Measured change on training observations');axes[2].legend(fontsize=8)
for ax in axes[:2]:ax.set_xticks(range(3),['Preserved\npolicy','Continued\nPPO','PPO + parent\npenalty'])
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('Fixed progression: matched training, explicit retention checks',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
lines=['# Fixed-progression PPO retention study','',
    'Both candidates started from the preserved policy with identical training seats, seed and budget. The extra pace action was removed. The parent penalty was the only difference between these two runs.','',
    '| Bank / controller | Complete successes | Worst impact |','|---|---:|---:|']
for key,s in summaries.items():
    bank,arm=key.split('_',1);lines.append(f"| {bank} / {labels[arm]} | {s['successes']}/16 | {s['worst_impact_N']:.1f} N |")
lines+=['','The full-horizon parent run reproduces all twelve earlier trace arrays exactly. Each training arm performed 32,768 transitions; completed and unfinished attempts account for all of them. Parent normalizers stayed exactly fixed, initial weights were identical, optimizers started fresh, and both learned finite nonzero weight changes.','',
    f"Fresh holdout used: {report['fresh_holdout_used']}. Known gates determine whether further evaluation is permitted; failed arms were stopped. No deployment default changed.",'',
    'The policy and critic use depth, robot kinematics and simulated onboard sensor equivalents. True poses and contact forces are evaluation/reward labels only. Gaussian joint exploration remains; this study removes the added asymmetric pace exploration.','',
    'This is one matched pair of short training runs, not proof of generalization to arbitrary surfaces. Freezing normalization and resetting the optimizer are shared changes, so comparisons to older experiments do not isolate those effects. Conditional torque results refer only to matched successful trials. If the wider gate is reached, its parent numbers reuse the preserved prior full-horizon evaluation on identical seeds and geometry.','',
    f"Preserved original files/checkpoints and clean IsaacLab checkout verified. Legacy storage-check exit: {proc.returncode}; root free {storage['root_free_gib']:.2f} GiB, actual output volume free {storage['output_free_gib']:.2f} GiB. Existing files were not deleted.",'',
    '![Matched experiment](comparison.png)','']
(OUT/'report.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in report.items() if k!='training'},indent=2))
