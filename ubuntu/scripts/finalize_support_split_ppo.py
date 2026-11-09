"""Delivery presentation and explicitly post-hoc diagnostics; no policy edits."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception')
out=store/'support_split_ppo_results_01'
r=json.loads((out/'results.json').read_text())
audit=json.loads((store/'support_split_policy_audit_01/results.json').read_text())
verification=dict(paired_startup_archives={},admitted_only_secondary={},late_trial_effort_secondary={})
for bank in r['banks']:
    paths={role:store/f'support_split_{bank}_{role}_01' for role in ('parent','trained')}
    with np.load(paths['parent']/'startup.npz') as old,np.load(paths['trained']/'startup.npz') as new:
        checks={k:bool(np.array_equal(old[k],new[k])) for k in old.files}
        assert all(checks.values());verification['paired_startup_archives'][bank]=checks
    for role,path in paths.items():
        rows=json.loads((path/'results.json').read_text())['episodes'];admitted=[x for x in rows if x['startup_ready']]
        verification['admitted_only_secondary'][bank+'_'+role]=dict(
            refused_cases=[x['env'] for x in rows if not x['startup_ready']],
            admitted_count=len(admitted),max_admitted_force_N=max(x['peak_force_N'] for x in admitted))
        with np.load(path/'rollout.npz') as trace:
            late=trace['t']>=18.
            verification['late_trial_effort_secondary'][bank+'_'+role]=dict(
                mean_leg_rms_last_2s_Nm=trace['torque'][late].mean(0).tolist())
verification.update(training=audit,total_evaluations=96,all_trials_full_20s=True,
    parent_successes=sum(e['parent']['successes'] for e in r['banks'].values()),
    trained_successes=sum(e['trained']['successes'] for e in r['banks'].values()),
    isaaclab_clean=not subprocess.check_output(['git','-C',str(root/'IsaacLab'),'status','--short'],text=True).strip())
assert verification['isaaclab_clean']
(out/'extended_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
shutil.copy2(store/'support_split_policy_audit_01/results.json',out/'policy_input_audit.json')
shutil.copy2(store/'support_split_case_visuals_01/case_comparison.png',out/'case_comparison.png')
shutil.copy2(store/'support_split_case_visuals_01/cases.json',out/'illustrative_cases.json')
posthoc=out/'posthoc_sources';posthoc.mkdir(exist_ok=False)
for name in ('audit_split_policy_inputs.py','render_split_case_comparison.py','finalize_support_split_ppo.py'):
    shutil.copy2(root/'scripts'/name,posthoc/name)
for name in ('support_split_interface_tests_02.log','support_split_validation_01.exit','support_split_ppo_01.exit'):
    shutil.copy2(root/'logs'/name,out/name)

# Preserve the original generated presentation before visual QA adjustments.
for name in ('REPORT.md','ppo_comparison.png','distance_interface.png'):
    p=out/name;shutil.copy2(p,p.with_name(p.stem+'_generated'+p.suffix))
fig,axes=plt.subplots(1,3,figsize=(13.3,4.9))
fig.subplots_adjust(top=.74,bottom=.18,left=.045,right=.99,wspace=.31)
x=np.arange(3);labels=['Known\nregression','Fresh seats\nand starts','New\nstarts']
for i,(role,label,color) in enumerate([('parent','Frozen controller','#3565a5'),('trained','After 32 PPO updates','#e07a32')]):
    entries=[e[role] for e in r['banks'].values()]
    for ax,key,fmt in zip(axes,['successes','max_force_N','mean_peak_hold_s'],['%.0f','%.0f','%.2f']):
        bars=ax.bar(x+(i-.5)*.32,[e[key] for e in entries],.32,color=color,label=label)
        ax.bar_label(bars,fmt=fmt,padding=3,fontsize=9)
for ax,title in zip(axes,['Successful sits / 16','Worst seat force (N)','Mean longest supported hold (s)']):
    ax.set_title(title,fontsize=12);ax.set_xticks(x,labels,fontsize=10);ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(top=ax.get_ylim()[1]*1.10)
axes[0].set_ylim(0,18)
fig.suptitle('One PPO pilot: 48 matched cases per policy',y=.99,fontsize=16)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False)
fig.text(.5,.015,'New-starts maximum includes the same refused startup case in both policies.',ha='center',fontsize=10)
fig.savefig(out/'ppo_comparison.png',dpi=160);plt.close(fig)

path=store/'support_split_regression_parent_01'
with np.load(path/'controller.npz') as f:t=f['t'];ob=f['policy'][:,14]
with np.load(path/'evaluation_labels.npz') as f:truth=f['true_clearance'][:,14]*100
fig,axes=plt.subplots(2,1,figsize=(10,5.2),sharex=True,layout='constrained',height_ratios=[1.4,1])
valid=ob[:,148]>0;patch=ob[:,140]>0;window=(t>=10)&(t<=14)
axes[0].plot(t,truth,color='black',linewidth=1,label='Simulator truth (evaluation only)')
axes[0].plot(t,np.where(valid,ob[:,144]*100,np.nan),color='#207f5b',linewidth=1.4,label='Independent distance')
axes[0].scatter(t[patch],ob[patch,132]*100,s=7,color='#3565a5',label='Original complete-patch input')
axes[0].set_ylim(min(-.5,float(truth[window].min())-.3),float(truth[window].max())+.8)
axes[0].set_ylabel('Pelvis clearance (cm)');axes[0].legend(fontsize=8,loc='upper right')
axes[1].step(t,valid.astype(float)+1.4,where='post',color='#207f5b')
axes[1].step(t,patch.astype(float),where='post',color='#3565a5')
axes[1].set_yticks([0,1,1.4,2.4],['Patch missing','Patch valid','Distance missing','Distance valid'])
axes[1].set_xlim(10,14);axes[1].set_xlabel('Simulation time (s)')
fig.suptitle('Known case 14: reliable distance survives seat-edge uncertainty')
fig.savefig(out/'distance_interface.png',dpi=160);plt.close(fig)

report='''# Distance interface fixed; one PPO pilot completed

The policy now receives reliable hip-to-seat clearance even when the visible seat
edges become uncertain. Distance has its own validity, age and error-budget
channels. Missing or expired distance is explicitly marked unavailable.

The expanded controller initially reproduced the old controller exactly:
16/16 known successes, identical actions and all12 recorded motion arrays.
During known-bank landings, distance availability increased from **82.73% to
99.67%**, with worst accepted error **1.74mm**. Sensor-only replay reproduced all
15,984 consumed post-startup geometry packets exactly. CPU/GPU migration,
invalid-data tests and isolated-reset checks passed.

## PPO comparison

One fixed32-update run,32,768 transitions, unchanged rewards/gains/action limits,
fixed normalization and a fresh optimizer. The final checkpoint was evaluated
without selecting among intermediate checkpoints. Each policy ran48 full20s
trials; failures and startup refusals remain in the scores.

| Test bank | Frozen controller | After PPO | Worst seat force: frozen → PPO |
|---|---:|---:|---:|
| Known regression |16/16|16/16|521 →616N|
| Fresh seats and starts |12/16|13/16|1078 →1176N|
| Mixed training seats, new starts |7/16|8/16|6448 →6448N|
| **Total** |**35/48**|**37/48**| |

No previously successful case was lost. Both new successes came from the fresh
tests. However, impact severity worsened on two banks, and mean seated leg torque
increased among cases both controllers completed: +0.112,+0.199,+0.132Nm across
the three banks. **The trained policy is preserved but not promoted.**

![Paired PPO comparison](ppo_comparison.png)

## What improved, and what remains limited

Fresh case13 increased supported hold from2.68 to5.14s. Mean foot load during
qualifying support decreased from66.0% to45.4% of body weight; mean leg torque
decreased from7.00 to6.37Nm over those intervals. Its leg torque nevertheless
rose late in the trial, so sustained low-effort sitting is not established.

Known case14 held longer but increased impact from363 to616N. The examples below
were chosen after scoring to show both outcomes; all48 paired cases remain in
the main results.

![One improvement and one regression](case_comparison.png)

The new-start bank contains the same startup refusal (case15) in both policies:
the landing patch was not observable. Sitting was never enabled, yet subsequent
physical contacts are retained; this case accounts for the6448N whole-bank
maximum. Admitted-only diagnostics are secondary and do not replace that score.

Distance accuracy remained below7.65mm for all accepted inputs across all96
evaluations. Landing availability ranges from96.31% to99.92%. Before-first-failure
coverage on the hard new-start bank remains below the99% diagnostic threshold:
98.88% for the parent and98.94% after PPO. Both coverage failures remain recorded.
The interface forwards accepted distance independently of patch validity; it
does not invent estimates when the tracker or startup detector refuses them.

![Distance and patch validity during landing](distance_interface.png)

New distance weights learned nonzero values, but scalar-clearance sensitivity is
very weak: a1cm perturbation changed the bounded correction by at most3.0e-7rad
on recorded training inputs (2.77e-7rad with patch unavailable). This experiment
compares continued PPO against a frozen controller; it does **not** isolate the
benefit of the extra channels from PPO using the original inputs. See
[functional audit](policy_input_audit.json).

## Evidence and reproduction

- [All paired cases, metrics and failures](results.json)
- [Startup parity, training accounting and secondary diagnostics](extended_verification.json)
- [Frozen protocol](frozen/docs/SUPPORT_SPLIT_PPO_PROTOCOL.md)
- [Original automatically generated report](REPORT_generated.md)
- Frozen runtime/training sources are in `frozen/`; subsequent analysis scripts
  are in `posthoc_sources/`. All original checkpoints and frozen source hashes
  verified unchanged. IsaacLab is clean. No software was installed, and no
  runtime defaults changed. Both persistent jobs finished with exit0.

Ubuntu checkpoint: `/data/humanoid-sitting-perception/support_split_ppo_train_01/final.pt`.
SHA256: `ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda`.
Load it through `scripts/train_support_split_observation.py`; it requires the
151-channel layout and split first-layer computation.

Scope remains aligned rigid seats in simulation. The geometry tracker uses
clean simulated depth/encoder/IMU equivalents; the original109-channel packet
retains its existing noise/delay. No new camera-noise or hardware validation was
performed. Simulator pose/forces supply rewards and evaluation labels only.
The geometry budget is conditional, not calibrated statistical confidence.
'''
(out/'REPORT.md').write_text(report)
hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
(out/'FILES.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({k:v for k,v in verification.items() if k not in ('training','paired_startup_archives','late_trial_effort_secondary')},indent=2))
