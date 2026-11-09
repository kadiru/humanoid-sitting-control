"""Serial persistent experiment, preserving every prior artifact."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import numpy as np

ROOT=Path('/home/kadir/humanoid-sitting'); STORE=Path('/data/humanoid-sitting-perception')
OUT=STORE/'support_capped_impact_01'; os.chdir(ROOT); OUT.mkdir(exist_ok=False)
os.environ.update(OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
completed=[]
def read(p):return json.loads(p.read_text())
def status(phase,**kwargs):
    row=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=completed,**kwargs)
    temp=OUT/'status.tmp';temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(OUT/'status.json')
    print(json.dumps(row),flush=True)
def run(name,arguments):
    command=['bash','scripts/run.sh','python','../scripts/train_support_capped_impact.py',
        '--output',str(OUT/name),'--parent',str(STORE/'support_rl_independent_train_04/final.pt'),
        '--resume151',str(STORE/'support_split_ppo_train_01/final.pt'),'--interface','split',
        '--num-envs','16','--device','cuda:0',
        '--kit_args=--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false',*arguments]
    (OUT/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n');status('running',job=name)
    with (OUT/f'{name}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    completed.append(name)

try:
    status('preflight')
    with (OUT/'unit_tests.log').open('w') as log:
        subprocess.run(['IsaacLab/.venv/bin/python','scripts/test_support_capped_impact.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
    old=read(STORE/'support_reward_study_01/frozen/manifest.json')
    for rel,h in old.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    sources=[ROOT/p for p in old]
    sources += [p for folder in ('scripts','sitting_task/src/humanoid_sitting') for p in (ROOT/folder).glob('*support_capped_impact*') if p.is_file()]
    sources += [ROOT/'docs/SUPPORT_CAPPED_IMPACT_PROTOCOL.md']
    frozen=OUT/'frozen';frozen.mkdir();hashes={}
    for p in sorted(set(sources)):
        rel=p.relative_to(ROOT);hashes[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.suffix in ('.py','.sh','.md'):
            dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    (frozen/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    weight=sum(read(ROOT/'runs/g1_kinematics_geometry_com_01/model.json')['body_mass'])*9.81
    rows=read(STORE/'support_reward_study_01/train_gentle/results.json')['episodes']
    excess=np.array([max(r['peak_force_N']/weight-1,0)**2 for r in rows if r['startup_ready']])
    (OUT/'archived_cost_audit.json').write_text(json.dumps(dict(
        admitted_completed_attempts=len(excess),previous_uncapped_total=float((12*excess).sum()),
        proposed_capped_total=float((5*np.minimum(excess,1)).sum()),
        maximum_capped_attempt_cost=float((5*np.minimum(excess,1)).max()),
        note='Counterfactual cost on fixed archived attempts, not a learned-policy result.'),indent=2)+'\n')
    banks={
      'regression':['--bank-profile','legacy','--geometry-count','8','--geometry-seed','51403','--seed','51407'],
      'fresh':['--geometry-count','8','--geometry-seed','52641','--seed','52643'],
      'starts':['--bank-from',str(STORE/'support_rl_pace_mixed_train_02/geometry_bank.json'),'--geometry-count','16','--geometry-seed','51621','--seed','52623'],
      'novel':['--geometry-count','8','--geometry-seed','52741','--seed','52743'],
      'unseen':['--geometry-count','8','--geometry-seed','52841','--seed','52843']}
    run('timer_regression',['--mode','evaluate','--cap-arm','legacy','--checkpoint',
        str(STORE/'support_reward_study_01/train_gentle/final.pt'),*banks['novel']])
    previous=STORE/'support_reward_study_01/novel_gentle';current=OUT/'timer_regression'
    parity={}
    for name in ('controller.npz','rollout.npz','startup.npz'):
        with np.load(previous/name) as a,np.load(current/name) as b:
            assert set(a.files)==set(b.files)
            parity[name]=all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
            assert parity[name],name
    before=read(previous/'results.json')['episodes'];after=read(current/'results.json')['episodes']
    changed=[y['env'] for x,y in zip(before,after) if x['success']!=y['success']]
    assert changed==[13] and after[13]['hold_ticks']==150 and after[13]['success']
    assert all(y['legacy_float_success']==x['success'] for x,y in zip(before,after))
    (OUT/'timer_validation.json').write_text(json.dumps(dict(passed=True,bitwise_parity=parity,
        changed_cases=changed,prior_successes=9,corrected_successes=10,exact_boundary_ticks=150),indent=2)+'\n')
    for seed,arms in ((52813,('legacy','capped')),(52817,('capped','legacy'))):
        for arm in arms:
            run(f'train_{seed}_{arm}',['--mode','train','--cap-arm',arm,'--iterations','32',
                '--bank-from',str(STORE/'support_rl_pace_mixed_train_02/geometry_bank.json'),
                '--geometry-count','16','--geometry-seed','51621','--seed',str(seed)])
    run('unseen_initial',['--mode','evaluate','--cap-arm','legacy',*banks['unseen']])
    for seed in (52813,52817):
        for arm in ('legacy','capped'):
            for bank,arguments in banks.items():
                run(f'{bank}_{seed}_{arm}',['--mode','evaluate','--cap-arm','legacy','--checkpoint',
                    str(OUT/f'train_{seed}_{arm}/final.pt'),*arguments])
    status('reporting')
    with (OUT/'report.log').open('w') as log:
        subprocess.run(['IsaacLab/.venv/bin/python','scripts/report_support_capped_impact.py','--study',str(OUT)],stdout=log,stderr=subprocess.STDOUT,check=True)
    status('complete',report=str(OUT/'results/REPORT.md'))
except BaseException:
    status('failed',traceback=traceback.format_exc());raise
