"""Independent final audit; append review artifacts without changing the study."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/kadir/humanoid-sitting')
STORE = Path('/data/humanoid-sitting-perception')
STUDY = STORE / 'support_capped_impact_01'
OUT = STUDY / 'verified_review'
SEEDS = (52813, 52817)
BANKS = ('regression', 'fresh', 'starts', 'novel', 'unseen')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert read(STUDY / 'status.json')['phase'] == 'complete'
for manifest in ('frozen/manifest.json', 'recovery/manifest.json'):
    assert all(sha(ROOT / path) == digest for path, digest in read(STUDY / manifest).items())
parent_path = STORE / 'support_split_ppo_train_01/final.pt'
assert sha(parent_path) == 'ece102b139323af18bd1c7aad6d8fb06e478f38946f07e52406a31bd4acc7dda'
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
budgets = []
for seed in SEEDS:
    for arm in ('legacy', 'capped'):
        path = STUDY / f'train_{seed}_{arm}'
        v = read(path / 'verification.json')
        accounting = read(path / 'attempt_accounting.json')
        assert v['ppo_iterations'] == 32 and v['transitions'] == 32768 and v['finite']
        assert accounting['completed_transitions'] + accounting['unfinished_transitions'] == 32768
        assert sum(r['steps'] for r in accounting['unfinished_attempts']) == accounting['unfinished_transitions']
        assert len((path / 'study_updates.jsonl').read_text().splitlines()) == 32
        initial = torch.load(path / 'initial.pt', map_location='cpu', weights_only=False)
        final = torch.load(path / 'final.pt', map_location='cpu', weights_only=False)
        for role in ('actor_state_dict', 'critic_state_dict'):
            assert set(parent[role]) == set(initial[role]) == set(final[role])
            assert all(torch.equal(value, initial[role][key]) for key, value in parent[role].items())
            assert all(torch.isfinite(value).all() for value in final[role].values())
            assert all(torch.equal(value, final[role][key]) for key, value in parent[role].items() if key.startswith('obs_normalizer.'))
        assert any(not torch.equal(value, final['actor_state_dict'][key]) for key, value in initial['actor_state_dict'].items())
        budgets.append(dict(seed=seed, arm=arm, updates=32, transitions=32768,
                            completed=accounting['completed_transitions'], unfinished=accounting['unfinished_transitions'],
                            finite=True, initial_equals_parent=True, normalizers_unchanged=True, final_sha256=sha(path / 'final.pt')))

old = STORE / 'support_reward_study_01/novel_gentle'
new = STUDY / 'timer_regression'
exceptions = []
for name in ('controller.npz', 'rollout.npz', 'startup.npz', 'evaluation_labels.npz'):
    with np.load(old / name) as a, np.load(new / name) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if np.array_equal(a[key], b[key], equal_nan=True):
                continue
            assert key == 'force' and name in ('rollout.npz', 'evaluation_labels.npz')
            x, y = a[key], b[key]
            assert x.dtype == y.dtype == np.float32 and np.isfinite(x).all() and np.isfinite(y).all()
            assert np.all(abs(x-y) <= np.spacing(np.maximum(abs(x), abs(y))))
            exceptions.append(dict(archive=name, count=int(np.sum(x != y)), max_difference_N=float(abs(x-y).max())))
before = read(old / 'results.json')['episodes']
after = read(new / 'results.json')['episodes']
assert [b['env'] for a, b in zip(before, after) if a['success'] != b['success']] == [13]
assert sum(r['success'] for r in before) == 9 and sum(r['success'] for r in after) == 10
assert after[13]['hold_ticks'] == 150
for a, b in zip(before, after):
    assert all(a[k] == b[k] for k in a if k not in ('success', 'hold_s'))
    assert a['success'] == b['legacy_float_success'] and a['hold_s'] == b['legacy_float_hold_s']

report = read(STUDY / 'results/results.json')
parity = []
for bank in BANKS:
    initial = (STUDY / 'unseen_initial' if bank == 'unseen' else
               STORE / 'support_reward_study_01/novel_initial' if bank == 'novel' else
               STORE / f'support_split_{bank}_trained_01')
    for name, rows in report['banks'][bank]['cases'].items():
        path = initial if name == 'initial' else STUDY / f'{bank}_{name}'
        raw_rows = read(path / 'results.json')['episodes']
        assert len(rows) == len(raw_rows) == 16
        assert [r['env'] for r in rows] == list(range(16))
        assert all(r['steps'] == 1000 and r['duration_s'] == 20 for r in raw_rows)
        for raw, row in zip(raw_rows, rows):
            ticks = raw.get('hold_ticks', round(raw['hold_s'] / .02))
            success = bool(raw['standing'] and raw['standing_checked'] and raw['startup_ready'] and
                           not any(raw['failures'].values()) and ticks >= 150)
            assert row['success'] == success and row['hold_ticks'] == ticks
        assert read(path / 'geometry_bank.json') == read(initial / 'geometry_bank.json')
        with np.load(path / 'startup.npz') as a, np.load(initial / 'startup.npz') as b:
            assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
        parity.append(dict(bank=bank, policy=name, startup_bitwise_equal=True, full_horizon=True))

def paired(a, b):
    common = [(x, y) for x, y in zip(a, b) if x['success'] and y['success']]
    result = dict(common=len(common), gained=sum(not x['success'] and y['success'] for x, y in zip(a,b)),
                  lost=sum(x['success'] and not y['success'] for x, y in zip(a,b)))
    for key in ('peak_force_N', 'seated_leg_rms_Nm', 'seated_foot_load_fraction'):
        values = [float(np.mean([pair[i][key] for pair in common])) for i in (0,1)] if common else [None,None]
        result[key] = dict(reference=values[0], candidate=values[1], percent_change=100*(values[1]/values[0]-1) if values[0] else None)
    return result

comparisons = []
for seed in (*SEEDS, 'pooled'):
    for bank in (*BANKS, 'all'):
        include = BANKS if bank == 'all' else (bank,)
        seeds = SEEDS if seed == 'pooled' else (seed,)
        rows = {arm:sum([report['banks'][b]['cases'][f'{s}_{arm}'] for s in seeds for b in include], []) for arm in ('legacy','capped')}
        initial = sum([report['banks'][b]['cases']['initial'] for s in seeds for b in include], [])
        comparisons.append(dict(seed=seed, bank=bank, capped_vs_old=paired(rows['legacy'],rows['capped']),
                                capped_vs_parent=paired(initial,rows['capped']),
                                old_refused=sum(not r['startup_ready'] for r in rows['legacy']),
                                capped_refused=sum(not r['startup_ready'] for r in rows['capped'])))
validation = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
                  budgets=budgets, total_transitions=sum(r['transitions'] for r in budgets),
                  parent_sha256=sha(parent_path), frozen_source_files_verified=40, recovery_files_verified=2,
                  timer_changed_cases=[13], timer_force_roundoff_exceptions=exceptions, evaluation_checks=parity,
                  original_automatic_report_sha256=sha(STUDY/'results/REPORT.md'),
                  original_results_sha256=sha(STUDY/'results/results.json'), promoted=False)
OUT.mkdir(exist_ok=False)
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(OUT/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')

fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
colors = ['#437ea4','#23816c']
for ax, key, title in zip(axes, ('peak_force_N','seated_leg_rms_Nm'), ('Peak seat force on common successes','Leg torque RMS on common successes')):
    for j,seed in enumerate(SEEDS):
        values = [next(r for r in comparisons if r['seed']==seed and r['bank']==b)['capped_vs_old'][key]['percent_change'] for b in ('all','unseen')]
        bars = ax.bar(np.arange(2)+(j-.5)*.32,values,.32,label=f'Seed {seed}',color=colors[j])
        ax.bar_label(bars,labels=[f'{v:+.1f}%' for v in values],padding=4)
    ax.axhline(0,color='#4b5563',lw=.8); ax.set_xticks([0,1],['All five banks','New holdout only'])
    ax.set_title(title); ax.set_ylabel('Capped vs old reward (%)'); ax.set_ylim(-18,12)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
axes[0].legend(loc='lower left')
fig.suptitle('Lower is better • paired cases successful under both rewards\nAll banks: 32 / 51 shared successes; new holdout: 6 / 6 (by seed)')
fig.savefig(OUT/'impact_effort.png',dpi=170); plt.close(fig)
print(json.dumps(dict(output=str(OUT),validation_passed=True,total_transitions=validation['total_transitions'],
    aggregate=[r for r in comparisons if r['bank'] in ('all','unseen')]),indent=2))
