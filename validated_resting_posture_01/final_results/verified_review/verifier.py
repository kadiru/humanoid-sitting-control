"""Read-only final audit of resting PPO; write additive delivery evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/humanoid-sitting-perception/resting_ppo_01')
OUT = ROOT / 'verified_review'
SEEDS = (52911, 52917)
ARMS = ('stability', 'effort')
BANKS = ('starts', 'heights', 'impulse')

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mean(values):
    return float(np.mean(values)) if len(values) else None

protocol = read(ROOT/'protocol.json')
assert read(ROOT/'status.json')['phase'] == 'completed'
manifest = read(ROOT/'frozen/manifest.json')
assert all(sha(Path(p)) == digest for p, digest in manifest.items())
probe = read(ROOT/'reset_probe/reset_probe.json')
assert probe['passed']
for bank, n in (('heldout_starts_bank',16), ('heldout_heights_bank',12)):
    assert read(ROOT/bank/'capture.json')['valid_count'] == n
    with np.load(ROOT/bank/'candidates.npz') as z:
        assert len(z['valid']) == n and z['valid'].all()
tests = (ROOT/'unit_tests.log').read_text()
assert 'Ran 12 tests' in tests and '\nOK' in tests
baseline = {}
for bank in BANKS:
    p = ROOT/f'fixed_{bank}'
    baseline[bank] = read(p/'results.json')['episodes']
    with np.load(p/'trace.npz') as z:
        assert np.array_equal(z['command'], np.broadcast_to(z['command'][0], z['command'].shape))

budgets, parity, summaries, diagnosis, heights = [], [], [], [], []
data = {}
for seed in SEEDS:
    initials = {}
    for arm in ARMS:
        train = ROOT/f'train_{seed}_{arm}'
        v = read(train/'verification.json')
        progress = read(train/'progress.json')
        assert v['updates'] == progress['update'] == 128
        assert v['transitions'] == v['active_transitions']+v['warmup_transitions'] == 131072
        episodes = [json.loads(line) for line in (train/'episodes.jsonl').read_text().splitlines()]
        assert sum(r['steps'] for r in episodes)+sum(v['unfinished_episode_steps']) == v['active_transitions']
        initial = torch.load(train/'initial.pt',map_location='cpu',weights_only=False)
        final = torch.load(train/'final.pt',map_location='cpu',weights_only=False)
        initials[arm] = initial
        assert read(train/'initial_policy_check.json')['zero_mean_joint_increment']
        for role in ('actor_state_dict','critic_state_dict'):
            assert all(torch.isfinite(x).all() for x in final[role].values())
        change = max(float(abs(v-final['actor_state_dict'][k]).max()) for k,v in initial['actor_state_dict'].items())
        assert change > 0
        budgets.append(dict(seed=seed,arm=arm,**v,actor_max_change=change,initial_sha256=sha(train/'initial.pt'),
                            final_sha256=sha(train/'final.pt'),training_successes=sum(r['success'] for r in episodes),
                            completed_attempts=len(episodes),completed_active_ticks=sum(r['steps'] for r in episodes)))
        for bank in BANKS:
            p=ROOT/f'{bank}_{seed}_{arm}'; ref=ROOT/f'fixed_{bank}'
            rows=read(p/'results.json')['episodes']; fixed=baseline[bank]
            data[(seed,arm,bank)]=rows
            assert len(rows)==len(fixed)==(16 if bank=='starts' else 12)
            assert [r['case'] for r in rows]==list(range(len(rows)))
            assert all(r['steps']==1000 and r['duration_s']==20 for r in rows+fixed)
            with np.load(ref/'startup.npz') as a,np.load(p/'startup.npz') as b:
                assert set(a.files)==set(b.files)
                error=max(float(np.max(np.abs(a[k]-b[k]))) for k in a.files)
                assert error <= protocol['startup_parity_absolute_tolerance']
            for name in ('geometry_bank.json','disturbances.json'):
                assert read(ref/name)==read(p/name)
            loaded=torch.load(p/'initial.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(v,loaded[role][k]) for role in ('actor_state_dict','critic_state_dict') for k,v in final[role].items())
            parity.append(dict(run=p.name,startup_max_error=error,full_20_seconds=True,
                               disturbances_match=True,geometry_matches=True,loaded_final_weights_match=True))
            with np.load(p/'trace.npz') as z,np.load(p/'startup.npz') as start:
                for i,row in enumerate(rows):
                    failed=np.flatnonzero(~z['alive'][:,i].astype(bool))
                    j=int(failed[0]) if len(failed) else len(z['t'])-1
                    drift=float(np.max(np.abs(z['command'][j,i]-start['command'][i])))
                    diagnosis.append(dict(seed=seed,arm=arm,bank=bank,case=i,height_m=row['height_m'],
                        first_failure_trace_s=float(z['t'][j]) if len(failed) else None,
                        command_max_drift_at_first_failure_rad=drift,qualifying_seconds=row['good_ticks']*.02,
                        failure_flags=row['failures']))
    for role in ('actor_state_dict','critic_state_dict'):
        assert all(torch.equal(v,initials['effort'][role][k]) for k,v in initials['stability'][role].items())

def summarize(rows):
    good=[r for r in rows if r['success']]
    return dict(total=len(rows),successes=len(good),mean_total_qualifying_seconds=mean([r['good_ticks']*.02 for r in rows]),
        failures={k:sum(r['failures'][k] for r in rows) for k in rows[0]['failures']},
        successful_effort={k:mean([r['metrics_on_good'][k] for r in good]) for k in
            ('normalized_torque_squared','all_joint_rms_Nm','leg_rms_Nm','torso_arm_hand_rms_Nm','foot_load_bw')})

for seed in SEEDS:
    for bank in (*BANKS,'all'):
        include=BANKS if bank=='all' else (bank,)
        rows={arm:sum((baseline[b] if arm=='fixed' else data[(seed,arm,b)] for b in include),[]) for arm in ('fixed',*ARMS)}
        pairs={}
        for name,a,b in [('effort_vs_fixed','fixed','effort'),('stability_vs_fixed','fixed','stability'),('effort_vs_stability','stability','effort')]:
            shared=[(x,y) for x,y in zip(rows[a],rows[b]) if x['success'] and y['success']]
            pairs[name]=dict(common_successes=len(shared),metrics={k:dict(reference=mean([x['metrics_on_good'][k] for x,y in shared]),
                candidate=mean([y['metrics_on_good'][k] for x,y in shared])) for k in
                ('normalized_torque_squared','all_joint_rms_Nm','leg_rms_Nm','torso_arm_hand_rms_Nm')})
        summaries.append(dict(seed=seed,bank=bank,policies={a:summarize(r) for a,r in rows.items()},pairs=pairs))
    for arm in ('fixed',*ARMS):
        rows=sum((baseline[b] if arm=='fixed' else data[(seed,arm,b)] for b in ('starts','heights')),[])
        for h in sorted({r['height_m'] for r in rows}):
            heights.append(dict(seed=seed,arm=arm,height_m=h,**summarize([r for r in rows if r['height_m']==h])))

audit=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,frozen_files_verified=len(manifest),
           budgets=budgets,total_transitions=sum(r['transitions'] for r in budgets),
           total_active_transitions=sum(r['active_transitions'] for r in budgets),
           total_warmup_transitions=sum(r['warmup_transitions'] for r in budgets),
           matched_initial_arm_weights=True,constant_baseline_commands=True,evaluations=parity,
           reset_probe=probe,tests=12,original_report_sha256=sha(ROOT/'results/report.html'),
           original_results_sha256=sha(ROOT/'results/results.json'),verifier_sha256=sha(Path(__file__)),promoted=False)
OUT.mkdir(exist_ok=False)
(OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
(OUT/'comparisons.json').write_text(json.dumps(dict(summaries=summaries,by_height=heights,trace_diagnostics=diagnosis),indent=2)+'\n')
shutil.copy2(__file__,OUT/'verifier.py')
for name in ('protocol.json','status.json','unit_tests.log','first_optimizer_validation.json'):
    shutil.copy2(ROOT/name,OUT/name)
shutil.copy2(ROOT/'frozen/manifest.json',OUT/'frozen_manifest.json')
shutil.copy2(ROOT/'reset_probe/reset_probe.json',OUT/'reset_probe.json')
for seed in SEEDS:
    for arm in ARMS:
        dest=OUT/f'train_{seed}_{arm}';dest.mkdir()
        for name in ('verification.json','progress.json','manifest.json','initial_policy_check.json'):
            shutil.copy2(ROOT/dest.name/name,dest/name)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for i,seed in enumerate(SEEDS):
    row=next(r for r in summaries if r['seed']==seed and r['bank']=='all')
    values=[row['policies'][a]['mean_total_qualifying_seconds'] for a in ARMS]
    bars=axes[0].bar(np.arange(2)+(i-.5)*.3,values,.3,label=f'Seed {seed}',color=['#4785a6','#24816a'][i])
    axes[0].bar_label(bars,labels=[f'{v:.2f}s' for v in values],padding=3)
axes[0].axhline(19,color='#b34437',linestyle='--',label='95% minimum time requirement')
axes[0].set(xticks=[0,1],xticklabels=['Stability PPO','Effort PPO'],ylim=(0,21),ylabel='Mean qualifying seconds within a 20s trial',title='Partial progress, but zero complete successes')
axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0, .88))
base=[r for r in heights if r['seed']==52911 and r['arm']=='fixed']
axes[1].plot([r['height_m']*100 for r in base],[r['successful_effort']['all_joint_rms_Nm'] for r in base],'o-',color='#555f6b')
axes[1].set(xlabel='Seat height (cm)',ylabel='Time average of instantaneous all-joint RMS (Nm)',ylim=(0,7.5),title='Fixed-posture effort: 4/4 successes at each height')
fig.suptitle('Resting PPO: both objectives failed to retain the fixed baseline\nQualifying time is diagnostic; it does not establish an effort saving')
for ax in axes:
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'diagnostics.png',dpi=170);plt.close(fig)
print(json.dumps(dict(output=str(OUT),passed=True,total_transitions=audit['total_transitions'],
    active=audit['total_active_transitions'],warmup=audit['total_warmup_transitions'],
    totals=[r for r in summaries if r['bank']=='all']),indent=2))
