"""Resume after documented one-ULP force-label parity difference; no reruns."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import numpy as np

ROOT=Path('/home/kadir/humanoid-sitting');STORE=Path('/data/humanoid-sitting-perception')
OUT=STORE/'support_capped_impact_01';os.chdir(ROOT)
os.environ.update(OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
def read(p):return json.loads(p.read_text())
assert read(OUT/'status.json')['phase']=='failed'
assert read(OUT/'status.json')['completed']==['timer_regression']
assert not list(OUT.glob('train_*'))
shutil.copy2(OUT/'status.json',OUT/'original_parity_failure.json')
completed=['timer_regression']
def status(phase,**kwargs):
    row=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=completed,
        recovery='one ULP tolerance for force labels only; all control/pose data bitwise equal',**kwargs)
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
    old=STORE/'support_reward_study_01/novel_gentle';new=OUT/'timer_regression';parity={};roundoff=[]
    for name in ('controller.npz','rollout.npz','startup.npz','evaluation_labels.npz'):
        parity[name]={}
        with np.load(old/name) as a,np.load(new/name) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:
                exact=np.array_equal(a[k],b[k],equal_nan=True);parity[name][k]=exact
                if exact:continue
                assert k=='force' and name in ('rollout.npz','evaluation_labels.npz')
                x,y=a[k],b[k];assert x.dtype==y.dtype==np.float32
                assert np.isfinite(x).all() and np.isfinite(y).all()
                difference=abs(x-y);ulp=np.spacing(np.maximum(abs(x),abs(y)))
                assert np.all(difference<=ulp)
                roundoff.append(dict(archive=name,field=k,different_values=int(np.sum(x!=y)),
                    max_difference_N=float(difference.max()),within_one_float32_ulp=True))
    before=read(old/'results.json')['episodes'];after=read(new/'results.json')['episodes']
    changed=[n['env'] for o,n in zip(before,after) if o['success']!=n['success']]
    assert changed==[13] and after[13]['hold_ticks']==150 and after[13]['success']
    for o,n in zip(before,after):
        assert n['legacy_float_success']==o['success'] and n['legacy_float_hold_s']==o['hold_s']
        assert all(o[k]==n[k] for k in o if k not in ('success','hold_s'))
    (OUT/'timer_validation.json').write_text(json.dumps(dict(passed=True,bitwise_parity=parity,
        roundoff_exceptions=roundoff,changed_cases=changed,prior_successes=9,corrected_successes=10,
        exact_boundary_ticks=150,all_other_episode_metrics_identical=True,
        original_failure_preserved='original_parity_failure.json'),indent=2)+'\n')
    recovery=OUT/'recovery';recovery.mkdir(exist_ok=False);hashes={}
    for p in (Path(__file__).resolve(),ROOT/'docs/SUPPORT_CAPPED_IMPACT_PARITY_AMENDMENT.md'):
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,recovery/p.name)
    (recovery/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for seed,arms in ((52813,('legacy','capped')),(52817,('capped','legacy'))):
        for arm in arms:
            run(f'train_{seed}_{arm}',['--mode','train','--cap-arm',arm,'--iterations','32',
                '--bank-from',str(STORE/'support_rl_pace_mixed_train_02/geometry_bank.json'),
                '--geometry-count','16','--geometry-seed','51621','--seed',str(seed)])
    banks={
      'regression':['--bank-profile','legacy','--geometry-count','8','--geometry-seed','51403','--seed','51407'],
      'fresh':['--geometry-count','8','--geometry-seed','52641','--seed','52643'],
      'starts':['--bank-from',str(STORE/'support_rl_pace_mixed_train_02/geometry_bank.json'),'--geometry-count','16','--geometry-seed','51621','--seed','52623'],
      'novel':['--geometry-count','8','--geometry-seed','52741','--seed','52743'],
      'unseen':['--geometry-count','8','--geometry-seed','52841','--seed','52843']}
    run('unseen_initial',['--mode','evaluate','--cap-arm','legacy',*banks['unseen']])
    for seed in (52813,52817):
        for arm in ('legacy','capped'):
            for bank,args in banks.items():
                run(f'{bank}_{seed}_{arm}',['--mode','evaluate','--cap-arm','legacy','--checkpoint',
                    str(OUT/f'train_{seed}_{arm}/final.pt'),*args])
    status('reporting')
    with (OUT/'report.log').open('w') as log:
        subprocess.run(['IsaacLab/.venv/bin/python','scripts/report_support_capped_impact.py','--study',str(OUT)],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h for rel,h in hashes.items())
    shutil.copytree(recovery,OUT/'results/recovery')
    status('complete',report=str(OUT/'results/REPORT.md'))
except BaseException:
    status('failed',traceback=traceback.format_exc());raise
