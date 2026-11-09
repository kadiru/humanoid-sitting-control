"""Verify unchanged replays, attribute contact failures, and audit zero-action retention."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STORE=Path('/data/humanoid-sitting-perception')
OLD=STORE/'resting_ppo_01'
DIAG=STORE/'resting_diagnosis_02'
VALID=STORE/'resting_bounded_validation_02'
OUT=DIAG/'verified_report'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]

assert read(VALID/'status.json')['phase']=='complete'
frozen=read(OLD/'frozen/manifest.json')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
replays=[]
for seed,arm in [(52911,'stability'),(52917,'effort')]:
    new=DIAG/f'replay_{seed}_{arm}';old=OLD/f'starts_{seed}_{arm}'
    assert read(new/'results.json')==read(old/'results.json')
    for file in ('startup.npz','trace.npz'):
        with np.load(new/file) as a,np.load(old/file) as b:
            assert set(a.files)==set(b.files)
            assert all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
    failures=lines(new/'first_failures.jsonl')
    bodies=collections.Counter(body['name'] for row in failures for body in row['body_contacts']
                               if not body['allowed_patch_body'] and body['peak_normal_N']>10)
    joints=collections.Counter(row['nearest_joint_limits'][0]['name'] for row in failures
                               if row['nearest_joint_limits'][0]['margin_rad']<.005)
    replays.append(dict(seed=seed,arm=arm,bitwise_identical_results_startup_trace=True,
                        forbidden_bodies=dict(bodies),joint_margin_failures=dict(joints),first_failures=failures))
pairs=lines(VALID/'contact_pairs/contact_pairs.jsonl')
assert len(pairs)==16
assert all(max(x['seat_peak_N'] for x in row['pairs'].values())>10 for row in pairs)
assert all(x['floor_peak_N']==0 for row in pairs for x in row['pairs'].values())
retention=[]
for bank,n in [('starts',16),('heights',12),('impulse',12),('hold',16)]:
    path=VALID/f'zero_{bank}';result=read(path/'results.json')
    assert result['successes']==n
    assert all(r['steps']==1000 and r['duration_s']==20 and not any(r['failures'].values()) for r in result['episodes'])
    with np.load(path/'trace.npz') as a:
        assert np.array_equal(a['command'],np.broadcast_to(a['command'][0],a['command'].shape))
        with np.load(OLD/f"fixed_{'starts' if bank=='hold' else bank}/trace.npz") as b:
            errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ('body','q','torque','command')}
            assert errors['body']<=1.e-6 and errors['q']<=1.e-6 and errors['command']==0
    retention.append(dict(bank=bank,success=n,total=n,constant_commands=True,errors=errors))
probe=read(VALID/'reset_probe/reset_probe.json');assert probe['passed']
report=dict(replays=replays,contact_pairs=pairs,retention=retention,reset_probe=probe,
            original_frozen_sources_preserved=len(frozen),probe_recovery=read(VALID/'recovery.json'),
            conclusion='Old accumulated corrections move the right hip-roll joint toward its limit and load hip-pitch housings against the seat. Bounded offsets preserve the stable reference when action is zero; learned-policy stability and effort savings remain to be tested.')
OUT.mkdir(exist_ok=False)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
old=DIAG/'replay_52911_stability';result=read(old/'results.json')
j=result['joint_names'].index('right_hip_roll_joint')
failure=next(r for r in replays[0]['first_failures'] if r['nearest_joint_limits'][0]['margin_rad']<.005)
i=failure['case']
with np.load(old/'trace.npz') as z:
    axes[0].plot(z['t'],z['command'][:,i,j],label='Old policy target')
    axes[0].plot(z['t'],z['q'][:,i,j],label='Actual joint')
axes[0].axhline(.255,color='#af4b43',linestyle='--',label='Failure margin')
axes[0].axvline(failure['t_s'],color='#777',linestyle=':')
axes[0].set(title=f'Right hip roll, case {i}',xlabel='Seconds',ylabel='Joint angle (rad)');axes[0].legend(fontsize=8)
peak=[max(v['seat_peak_N'] for v in row['pairs'].values()) for row in pairs]
axes[1].bar(range(16),peak,color='#b46146')
axes[1].set(title='Hip housing touches the seat',xlabel='Replay trial',ylabel='Seat normal force at first failure (N)')
bars=axes[2].bar(['25–40 cm','Intermediate','Impulse','Hold enabled'],[16,12,12,16],color='#388470')
axes[2].bar_label(bars,labels=['16/16','12/12','12/12','16/16'],padding=3)
axes[2].set(title='New zero-action interface retains rest',ylabel='Successful 20-second trials',ylim=(0,19))
axes[2].tick_params(axis='x',labelsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(OUT/'diagnosis_and_retention.png',dpi=170)
print(json.dumps(dict(retention=retention,replays=[{k:v for k,v in r.items() if k!='first_failures'} for r in replays]),indent=2))
