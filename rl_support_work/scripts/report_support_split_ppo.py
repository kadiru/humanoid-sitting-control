"""Paired full-horizon comparison of the one predeclared distance-interface pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from humanoid_sitting.support_split_observation import install_split_linear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--output',required=True)
a=p.parse_args();store=Path(a.store);out=Path(a.output);out.mkdir(exist_ok=False)
def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(rows):
    forces=np.array([x['peak_force_N'] for x in rows])
    good=[x for x in rows if x['success']]
    return dict(trials=len(rows),successes=len(good),max_force_N=float(forces.max()),
        p95_force_N=float(np.quantile(forces,.95)),median_force_N=float(np.median(forces)),
        mean_peak_hold_s=float(np.mean([x['hold_s'] for x in rows])),
        success_mean_leg_rms_Nm=float(np.mean([x['seated_leg_rms_Nm'] for x in good])) if good else None,
        success_mean_foot_fraction=float(np.mean([x['seated_foot_load_fraction'] for x in good])) if good else None,
        failures={k:sum(x['failures'][k] for x in rows) for k in rows[0]['failures']})

report=dict(protocol='SUPPORT_SPLIT_PPO_PROTOCOL.md',training=read(store/'support_split_ppo_train_01/verification.json'),banks={})
all_rows=[]
for bank in ('regression','fresh','starts'):
    paths={role:store/f'support_split_{bank}_{role}_01' for role in ('parent','trained')}
    rows={role:read(path/'results.json')['episodes'] for role,path in paths.items()}
    assert all(len(r)==16 for r in rows.values())
    geometry_equal=read(paths['parent']/'geometry_bank.json')==read(paths['trained']/'geometry_bank.json')
    with np.load(paths['parent']/'controller.npz') as before,np.load(paths['trained']/'controller.npz') as after:
        starts_equal=bool(np.array_equal(before['policy'][:2,:,:109],after['policy'][:2,:,:109]))
    assert geometry_equal and starts_equal
    paired=[]
    for old,new in zip(rows['parent'],rows['trained']):
        assert old['env']==new['env'] and old['geometry']==new['geometry']
        paired.append(dict(case=old['env'],parent_success=old['success'],trained_success=new['success'],
            parent_force_N=old['peak_force_N'],trained_force_N=new['peak_force_N'],
            parent_hold_s=old['hold_s'],trained_hold_s=new['hold_s'],
            parent_failures=old['failures'],trained_failures=new['failures']))
    entry=dict(parent=summarize(rows['parent']),trained=summarize(rows['trained']),
        newly_successful=[x['case'] for x in paired if not x['parent_success'] and x['trained_success']],
        lost_successes=[x['case'] for x in paired if x['parent_success'] and not x['trained_success']],
        geometry_equal=geometry_equal,initial_sensor_packets_equal=starts_equal,paired=paired,
        observation_scores={role:read(path/'split_observation_score.json') for role,path in paths.items()})
    common=[(old,new) for old,new in zip(rows['parent'],rows['trained']) if old['success'] and new['success']]
    entry['common_successes']=dict(count=len(common),mean_leg_rms_change_Nm=float(np.mean([
        new['seated_leg_rms_Nm']-old['seated_leg_rms_Nm'] for old,new in common])) if common else None)
    report['banks'][bank]=entry
    for role,path in paths.items():
        dest=out/bank/role;dest.mkdir(parents=True)
        for name in ('results.json','manifest.json','geometry_bank.json','split_observation_score.json','attempt_accounting.json'):
            shutil.copy2(path/name,dest/name)

checkpoint=store/'support_split_ppo_train_01/final.pt'
trained=torch.load(checkpoint,map_location='cpu',weights_only=False)
initial=torch.load(store/'support_split_ppo_train_01/initial.pt',map_location='cpu',weights_only=False)
obs=TensorDict({'policy':torch.zeros(16,151)},[16])
model=MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[128,128],obs_normalization=True,
    distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=.04,std_type='log'))
install_split_linear(model);model.load_state_dict(trained['actor_state_dict']);model.eval()
sensitivity=[];sensitive_missing_patch=[]
with torch.no_grad():
    for bank in report['banks']:
        with np.load(store/f'support_split_{bank}_trained_01/controller.npz') as f:
            x=torch.from_numpy(f['policy']).flatten(0,1)
        for batch in x.split(512):
            masked=batch.clone();masked[:,144:]=masked.new_tensor([0,0,0,0,0,1,.02])
            diff=(model(TensorDict({'policy':batch},[len(batch)]))-
                  model(TensorDict({'policy':masked},[len(batch)]))).abs().max(1).values.numpy()
            sensitivity.extend(diff.tolist())
            gate=(batch[:,148]>0)&(batch[:,140]==0)
            sensitive_missing_patch.extend(diff[gate.numpy()].tolist())
normalization_equal=all(torch.equal(value,trained[role][key]) for role in ('actor_state_dict','critic_state_dict')
    for key,value in initial[role].items() if key.startswith('obs_normalizer.'))
report['learning_evidence']=dict(new_weight_max=float(trained['actor_state_dict']['mlp.0.weight'][:,144:].abs().max()),
    initial_new_weight_max=float(initial['actor_state_dict']['mlp.0.weight'][:,144:].abs().max()),
    normalization_bitwise_unchanged=normalization_equal,
    masked_distance_action_difference_max=float(max(sensitivity)),
    valid_distance_missing_patch_samples=len(sensitive_missing_patch),
    valid_distance_missing_patch_action_difference_max=float(max(sensitive_missing_patch,default=0)),
    note='Offline input masking measures action sensitivity, not a closed-loop success ablation.')
report['checkpoint_sha256']=digest(checkpoint)
frozen=store/'support_split_ppo_frozen_01';root=Path(__file__).resolve().parents[1]
hashes=read(frozen/'manifest.json')
report['preservation']=dict(frozen_sources_and_original_checkpoints_unchanged=all(digest(root/path)==sha for path,sha in hashes.items()),
    original_parent_sha256=digest(store/'support_rl_independent_train_04/final.pt'),
    defaults_changed=False)
heldout=sum(report['banks'][k]['trained']['successes']-report['banks'][k]['parent']['successes'] for k in ('fresh','starts'))
report['promotion_checks']=dict(heldout_success_improved=heldout>0,
    no_lost_successes=all(not e['lost_successes'] for e in report['banks'].values()),
    no_worse_bank_peak_force=all(e['trained']['max_force_N']<=e['parent']['max_force_N'] for e in report['banks'].values()))
report['eligible_for_promotion']=all(report['promotion_checks'].values())
report['promoted']=False
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')

fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
labels=['Known regression','Fresh shapes / starts','New starts']
x=np.arange(3);colors=['#3565a5','#e07a32']
for i,(role,label) in enumerate([('parent','Frozen controller'),('trained','After 32 PPO updates')]):
    entries=[e[role] for e in report['banks'].values()]
    axes[0].bar(x+(i-.5)*.32,[e['successes'] for e in entries],.32,color=colors[i],label=label)
    axes[1].bar(x+(i-.5)*.32,[e['max_force_N'] for e in entries],.32,color=colors[i])
    axes[2].bar(x+(i-.5)*.32,[e['mean_peak_hold_s'] for e in entries],.32,color=colors[i])
for ax,title in zip(axes,['Successful sits / 16','Worst seat force (N)','Mean longest supported hold (s)']):
    ax.set_title(title);ax.set_xticks(x,labels,rotation=15,ha='right');ax.spines[['top','right']].set_visible(False)
axes[0].set_ylim(0,17);axes[0].legend(loc='upper left',fontsize=8)
fig.suptitle('One small PPO pilot • matched 20-second simulations • all failures retained')
fig.savefig(out/'ppo_comparison.png',dpi=160);plt.close(fig)

with np.load(store/'support_split_regression_parent_01/controller.npz') as f:
    t=f['t'];ob=f['policy'][:,14]
with np.load(store/'support_split_regression_parent_01/evaluation_labels.npz') as f:gt=f['true_clearance'][:,14]
fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
patch=ob[:,140]>0;dist=ob[:,148]>0
axes[0].plot(t,gt*100,color='black',label='Simulator truth (evaluation only)',linewidth=1)
axes[0].plot(t,np.where(dist,ob[:,144]*100,np.nan),color='#207f5b',label='Independent distance',linewidth=1.4)
axes[0].scatter(t[patch],ob[patch,132]*100,s=7,color='#3565a5',label='Original complete-patch input')
axes[0].set_ylabel('Pelvis clearance (cm)');axes[0].legend(fontsize=8)
axes[1].step(t,patch.astype(float),where='post',label='Complete patch',color='#3565a5')
axes[1].step(t,dist.astype(float)+1.4,where='post',label='Independent distance',color='#207f5b')
axes[1].set_yticks([0,1,1.4,2.4],['Patch missing','Patch valid','Distance missing','Distance valid'])
axes[1].set_xlim(10,14);axes[1].set_xlabel('Simulation time (s)')
fig.suptitle('Known case 14: distance remains available through seat-edge uncertainty')
fig.savefig(out/'distance_interface.png',dpi=160);plt.close(fig)

lines=['# Independent distance interface and bounded PPO pilot','',
 'The observation interface now preserves reliable clearance when seat edges are uncertain. The first144 inputs and the parent action function were preserved at initialization; seven independent distance channels were added.','',
 '## Paired full-horizon results','',
 '| Bank | Frozen controller | Trained policy | Worst force: frozen → trained | Newly lost successes |',
 '|---|---:|---:|---:|---|']
for label,(bank,e) in zip(labels,report['banks'].items()):
    lines.append(f"| {label} | {e['parent']['successes']}/16 | {e['trained']['successes']}/16 | {e['parent']['max_force_N']:.1f} → {e['trained']['max_force_N']:.1f} N | {e['lost_successes']} |")
lines+=['','![PPO comparison](ppo_comparison.png)','','![Distance interface](distance_interface.png)','',
 '## Interpretation','',
 f"Exactly {report['training']['ppo_iterations']} updates / {report['training']['transitions']} transitions. Final checkpoint evaluated without selection. Promotion checks: {report['promotion_checks']}. No defaults were changed.",
 '', 'The held-out banks contain16 attempts each; this is a small paired experiment, not evidence of universal sitting or hardware readiness. Rigid, aligned simulation seats only. Whole-trial failures, impact extremes, all paired case outcomes and distance coverage are retained in results.json.',
 '', 'Runtime estimates use depth, joint kinematics and IMU. Simulator state supplies reward/evaluation labels only. Patch and distance validity are independent; unavailable distance is explicit, not held indefinitely. The engineering error bound is conditional, not calibrated uncertainty.',
 '', f"New-input learning evidence: {report['learning_evidence']}",
 '', f"Preservation: {report['preservation']}",
 '', f"Checkpoint: `{checkpoint}` SHA256 `{report['checkpoint_sha256']}`.",
 '', 'Sources/protocol, interface checks, training verification and paired per-case results accompany this report.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copytree(frozen,out/'frozen')
for source in [store/'support_split_replay_01/results.json',store/'support_split_online_replay_01/results.json',store/'support_split_regression_parent_01/split_observation_score.json',
               store/'support_split_reset_probe_01/reset_probe.json',store/'support_split_ppo_train_01/verification.json',
               store/'support_split_ppo_train_01/ppo_updates.jsonl']:
    shutil.copy2(source,out/(source.parent.name+'_'+source.name))
print(json.dumps({k:v for k,v in report.items() if k!='banks'},indent=2))
