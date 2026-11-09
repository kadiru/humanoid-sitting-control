"""Post-hoc diagnosis of a frozen experiment. Does not train or alter scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('--study',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); study=Path(a.study); store=study.parent; out=Path(a.output); out.mkdir(exist_ok=False)
root=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def mean(x): return float(np.mean(x)) if len(x) else None
report=read(study/'results/results.json')
roles=('initial','legacy','gentle'); banks=('regression','fresh','starts','novel')
def path(bank,role):
    if role!='initial': return study/f'{bank}_{role}'
    return study/'novel_initial' if bank=='novel' else store/f'support_split_{bank}_trained_01'
diag=dict(primary_totals=report['totals'],paired_startup_equal=all(e['paired_startup_equal'] for e in report['banks'].values()))
rows={role:[r for bank in banks for r in report['banks'][bank]['cases'][role]] for role in roles}
common=[i for i in range(64) if all(rows[r][i]['success'] for r in roles)]
keys=('peak_force_N','seated_leg_rms_Nm','seated_foot_load_fraction','hold_s')
diag['common_successes_all_three']=dict(count=len(common),means={role:{key:mean([rows[role][i][key] for i in common]) for key in keys} for role in roles})
diag['pairwise']={}
for ref,candidate in (('initial','gentle'),('legacy','gentle'),('initial','legacy')):
    pairs=list(zip(rows[ref],rows[candidate])); successful=[(x,y) for x,y in pairs if x['success'] and y['success']]
    diag['pairwise'][f'{candidate}_vs_{ref}']=dict(common=len(successful),
        gained=sum(not x['success'] and y['success'] for x,y in pairs),lost=sum(x['success'] and not y['success'] for x,y in pairs),
        common_mean_differences={key:mean([y[key]-x[key] for x,y in successful]) for key in keys})

# The accumulated float32 duration is a diagnostic bug, not a new physical criterion.
f=torch.zeros(1)
for _ in range(150): f+=.02
diag['hold_roundoff']=dict(reproduced_150_ticks_s=f.item(),original_test_passes=bool(f>=3),changed_cases=[],tick_based_totals={})
for role in roles:
    count=0
    for bank in banks:
        for row in report['banks'][bank]['cases'][role]:
            ticks=round(row['hold_s']/.02)
            assert abs(ticks*.02-row['hold_s'])<.001
            corrected=bool(row['standing'] and row['standing_checked'] and row['startup_ready'] and
                           not any(row['failures'].values()) and ticks>=150)
            count+=corrected
            if corrected!=row['success']:
                diag['hold_roundoff']['changed_cases'].append(dict(bank=bank,role=role,case=row['env'],
                    hold_s=row['hold_s'],ticks=ticks,original=row['success'],tick_based=corrected))
    diag['hold_roundoff']['tick_based_totals'][role]=count

mass=sum(read(root/'runs/g1_kinematics_geometry_com_01/model.json')['body_mass']); weight=mass*9.81
diag['body_weight_from_export_N']=weight
diag['training']={}
for arm in ('legacy','gentle'):
    train=study/f'train_{arm}'; updates=[json.loads(s) for s in (train/'study_updates.jsonl').read_text().splitlines()]
    totals={k:sum(r['reward_contributions'][k] for r in updates) for k in updates[0]['reward_contributions']}
    episodes=read(train/'results.json')['episodes']
    admitted=[e for e in episodes if e['startup_ready']]
    costs=np.array([12*max(e['peak_force_N']/weight-1,0)**2 for e in admitted])
    terminal_impact=np.array([e['failures']['impact'] for e in admitted],dtype=bool)
    entry=dict(reward_totals=totals,positive_total=sum(v for k,v in totals.items() if k!='actual_total' and v>0),
        supported_good_time_s=totals['hold']/6,
        value_loss_first32=mean([r['losses']['value'] for r in updates[:32]]),
        value_loss_last32=mean([r['losses']['value'] for r in updates[-32:]]),
        parent_kl_final=updates[-1]['losses']['parent_kl'])
    if arm=='gentle':
        entry['completed_admitted_peak_cost_diagnostic']=dict(total=float(costs.sum()),
            fraction_on_impact_failure_attempts=float(costs[terminal_impact].sum()/costs.sum()),
            max_single_attempt=float(costs.max()),median=float(np.median(costs)),
            note='Reconstructed from complete-attempt peak forces and exported mass; excludes refused/unfinished attempts.')
    diag['training'][arm]=entry

# Contact-window diagnostics only use archived commands and separate truth labels.
diag['lost_successes']=[]; diag['input_distance_sensitivity']={}
for bank in banks:
    e=report['banks'][bank]
    lost=[i for i,(x,y) in enumerate(zip(e['cases']['initial'],e['cases']['gentle'])) if x['success'] and not y['success']]
    for case in lost:
        item=dict(bank=bank,case=case,roles={})
        for role in roles:
            with np.load(path(bank,role)/'controller.npz') as z:
                t=z['t']; obs=z['policy'][:,case]; actions=z['actions'][:,case]
            with np.load(path(bank,role)/'evaluation_labels.npz') as z:
                assert np.array_equal(t,z['t']); truth=z['true_clearance'][:,case]; force=z['force'][:,case]
                failed=z['failure'][:,case]; support=z['support'][:,case]
            contact=np.flatnonzero(support>.25); fail=np.flatnonzero(failed)
            event=min(contact[0] if len(contact) else 999,fail[0] if len(fail) else 999)
            mask=(t>=max(0,t[event]-1))&(t<=t[event]); valid=obs[:,148]>.5
            accepted=mask&valid
            item['roles'][role]=dict(first_support_s=float(t[contact[0]]) if len(contact) else None,
                first_failure_s=float(t[fail[0]]) if len(fail) else None,window_end_s=float(t[event]),
                distance_valid_percent=100*mean(valid[mask]),
                max_accepted_error_mm=float(np.max(abs(obs[accepted,144]-truth[accepted]))*1000) if accepted.any() else None,
                max_raw_action=float(abs(actions[mask]).max()),
                saturated_action_fraction=float(np.mean(abs(actions[mask])>=1)),
                last_100ms_clearance_change_m=float(truth[event]-truth[max(0,event-5)]))
        diag['lost_successes'].append(item)

from tensordict import TensorDict
from rsl_rl.models import MLPModel
from humanoid_sitting.support_split_observation import install_split_linear
torch.set_num_threads(1)
example=TensorDict({'policy':torch.zeros(1,151)},[1])
for role in roles:
    checkpoint=store/'support_split_ppo_train_01/final.pt' if role=='initial' else study/f'train_{role}/final.pt'
    model=MLPModel(example,{'actor':['policy']},'actor',23,hidden_dims=[128,128],obs_normalization=True,
        distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=.04,std_type='log'))
    install_split_linear(model); model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict']); model.eval()
    diffs=[]
    # Fixed starting-policy inputs for all networks, away from clipping limits.
    with torch.no_grad():
        for bank in banks:
            with np.load(path(bank,'initial')/'controller.npz') as z: x=torch.from_numpy(z['policy']).flatten(0,1)
            x=x[x[:,148]>.5]
            for batch in x.split(1024):
                altered=batch.clone(); altered[:,144]+=.01
                old=model(TensorDict({'policy':batch},[len(batch)])).clamp(-1,1)
                new=model(TensorDict({'policy':altered},[len(batch)])).clamp(-1,1)
                diffs.extend((.08*(new-old).abs().max(1).values).tolist())
    diag['input_distance_sensitivity'][role]=dict(samples=len(diffs),max_correction_change_rad=max(diffs),
        p95_correction_change_rad=float(np.quantile(diffs,.95)),note='Scalar-only +1cm perturbation on fixed inputs; not a closed-loop causal ablation.')

fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
labels=['Starting\npolicy','More PPO\nold reward','More PPO\nnew reward']; colors=['#788695','#3982b4','#ce7750']
values=[[diag['primary_totals'][r] for r in roles],
    [diag['common_successes_all_three']['means'][r]['peak_force_N'] for r in roles],
    [diag['common_successes_all_three']['means'][r]['seated_leg_rms_Nm'] for r in roles]]
for ax,v,title in zip(axes,values,['Successful sits /64','Mean peak force (N)\n36 common successful cases','Mean seated leg torque (Nm)\nSame36 successful cases']):
    bars=ax.bar(labels,v,color=colors); ax.set_title(title,fontsize=11); ax.spines[['top','right']].set_visible(False)
    ax.bar_label(bars,labels=[f'{x:.0f}' if ax is axes[0] else f'{x:.2f}' for x in v],padding=4)
    ax.set_ylim(0,max(v)*1.2)
fig.suptitle('Matched final-policy evaluations: the new reward did not improve the overall result')
fig.savefig(out/'matched_comparison.png',dpi=160); plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
for role,color,label in zip(roles,colors,['Starting policy','Old reward','New reward']):
    with np.load(path('regression',role)/'controller.npz') as z: t=z['t']; obs=z['policy'][:,15]
    with np.load(path('regression',role)/'evaluation_labels.npz') as z: force=z['force'][:,15]; clearance=z['true_clearance'][:,15]
    with np.load(path('regression',role)/'rollout.npz') as z: rt=z['t']; torque=z['torque'][:,15]
    axes[0].plot(t,clearance*100,color=color,label=label)
    axes[0].plot(t,np.where(obs[:,148]>.5,obs[:,144]*100,np.nan),color=color,ls=':',lw=1)
    axes[1].plot(t,force,color=color); axes[2].plot(rt,torque,color=color)
axes[1].axhline(2*weight,color='#333333',ls='--',lw=1,label='Failure threshold')
axes[0].legend(ncol=3,fontsize=9); axes[1].legend(fontsize=8)
for ax,label in zip(axes,['Pelvis clearance (cm)','Seat force (N)','Leg RMS torque (Nm)']):
    ax.set_ylabel(label); ax.spines[['top','right']].set_visible(False)
axes[2].set_xlim(9,16); axes[2].set_xlabel('Simulation time (s)')
fig.suptitle('Known case15: new reward produces a harder landing\nSolid clearance: evaluation truth; dotted: sensor estimate')
fig.savefig(out/'lost_case15.png',dpi=160); plt.close(fig)

diag['preservation']=dict(frozen_files_unchanged=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h
    for p,h in read(study/'frozen/manifest.json').items()),no_new_simulation_or_training=True)
assert diag['preservation']['frozen_files_unchanged'] and diag['paired_startup_equal']
(out/'diagnosis.json').write_text(json.dumps(diag,indent=2)+'\n')
shutil.copy2(__file__,out/'diagnose_support_reward_study.py')
print(json.dumps({k:v for k,v in diag.items() if k!='lost_successes'},indent=2))
