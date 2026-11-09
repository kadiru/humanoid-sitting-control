"""Two bounded-control PPO seeds; fixed versus PPO versus measured-effort hold."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import numpy as np

ROOT=Path('/home/kadir/humanoid-sitting')
STORE=Path('/data/humanoid-sitting-perception')
VALID=STORE/'resting_bounded_validation_02'
OLD=STORE/'resting_ppo_01'
OUT=STORE/'resting_bounded_ppo_02'
PYTHON=ROOT/'IsaacLab/.venv/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(VALID/'status.json')['phase']=='complete'
assert read(VALID/'reset_probe/reset_probe.json')['passed']
retention=[]
for bank,n in [('starts',16),('heights',12),('impulse',12)]:
    path=VALID/f'zero_{bank}'
    assert read(path/'results.json')['successes']==n
    with np.load(path/'trace.npz') as a,np.load(OLD/f'fixed_{bank}/trace.npz') as b:
        assert np.array_equal(a['command'],b['command'])
        errors={k:float(np.max(abs(a[k]-b[k]))) for k in ('body','q','torque')}
        assert errors['body']<=1.e-6 and errors['q']<=1.e-6,errors
    retention.append(dict(bank=bank,success=n,total=n,constant_command_matches_prior=True,errors=errors))
assert read(VALID/'zero_hold/results.json')['successes']==16
OUT.mkdir(exist_ok=False);(OUT/'frozen').mkdir()
completed=[]
def status(phase,**extra):
    p=OUT/'status.tmp';p.write_text(json.dumps(dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=completed,**extra),indent=2)+'\n');p.replace(OUT/'status.json')
protocol=dict(seeds=[53111,53117],training_objective='unchanged calibrated whole-body-effort reward from resting_ppo_01',
    updates=64,steps_per_update=64,num_envs=16,transitions_per_run=65536,total_transitions=131072,
    control='bounded offsets from the known starting command; .025rad hip/ankle roll, .05rad others; alpha .2; .002rad/tick slew; physical command margin .03rad',
    observations='393 sensor-only channels: three original108-channel frames plus23 remembered reference joint commands per frame',
    runtime_variants=['ppo','ppo_hold'],hold_goal=dict(effort_ratio=.9,dwell_ticks=100,release_ratio=1.05,
        sensors='all measured joint torque; controlled-joint velocities; IMU gyro/gravity; measured local depth plane clearance, validity, age',
        note='evaluation-only intervention on the same trained checkpoint; maintains PD targets, not motor shutdown'),
    training_heights_m=[.25,.30,.35,.40],known_intermediate_heights_m=[.275,.325,.375],
    new_holdout_heights_m=[.2625,.3125,.3625,.3875],new_holdout_seed=53203,
    conditions_per_policy=56,unique_states=44,repeated_impulse_states=12,seconds=20,
    success='unchanged: no physical failure,>=95% qualifying ticks,final2s continuously qualifying',
    effort_goal='at least10% lower normalized squared all-joint torque than the matched fixed baseline while retaining full-duration success; report Nm and load distribution separately',
    primary_effort_comparison='common successful full trials; do not claim savings from pre-failure fragments',
    effort_reference=read(OLD/'protocol.json')['effort_reference'],startup_tolerance=1.e-6,
    learning_rate=1.e-4,epochs=3,minibatches=4,entropy=.001,initial_std=.15,
    known_banks_are_regressions=True,no_automatic_promotion=True,
    limitations=['Two seeds reuse44 unique states;12 repeated with velocity impulses','Rigid level slabs; ideal proprioception','Measured hold gate is not proof of physical support; unchanged label-side evaluation remains decisive'])
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(OUT/'retention_validation.json').write_text(json.dumps(retention,indent=2)+'\n')
paths=list(map(Path,read(OLD/'frozen/manifest.json')))
paths+=[Path(__file__),ROOT/'scripts/run_resting_bounded.py',ROOT/'scripts/test_resting_bounded_control.py',
        ROOT/'docs/RESTING_BOUNDED_PROTOCOL.md',
        ROOT/'scripts/report_resting_bounded_02.py',ROOT/'sitting_task/src/humanoid_sitting/resting_bounded_control.py',
        OUT/'protocol.json',VALID/'reset_probe/reset_probe.json']
paths+=list(VALID.glob('zero_*/results.json'))
paths=list(dict.fromkeys(p.resolve() for p in paths))
frozen={str(p):sha(p) for p in paths}
for i,p in enumerate(paths):shutil.copy2(p,OUT/'frozen'/f'{i:02}_{p.name}')
(OUT/'frozen/manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
def run(name,command):
    assert all(sha(Path(p))==digest for p,digest in frozen.items())
    status('running',job=name);(OUT/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (OUT/f'{name}.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    completed.append(name)
def sim(name,mode,bank=None,n=16,seed=53011,entry='run_resting_bounded.py',**extra):
    cmd=['bash','scripts/run_resting_gpu1.sh','python','-u','../scripts/'+entry,'--mode',mode,'--output',str(OUT/name),
         '--num-envs',str(n),'--seed',str(seed),'--seconds','24' if mode=='capture' else '20',
         '--effort-reference',str(protocol['effort_reference']),'--device','cuda:0',
         '--kit_args=--/renderer/activeGpu=1,--/renderer/multiGpu/enabled=false']
    if bank:cmd+=['--bank',str(bank)]
    for key,value in extra.items():
        cmd.append('--'+key.replace('_','-'))
        if value is not True:cmd+=list(map(str,value)) if isinstance(value,list) else [str(value)]
    run(name,cmd)
try:
    run('unit_tests',[str(PYTHON),'scripts/test_resting_bounded_control.py','-v'])
    sim('novel_bank','capture',seed=53203,entry='run_resting_posture.py',heights=protocol['new_holdout_heights_m'])
    assert read(OUT/'novel_bank/capture.json')['valid_count']==16
    banks={'starts':(OLD/'heldout_starts_bank',16,0.),'heights':(OLD/'heldout_heights_bank',12,0.),
           'impulse':(OLD/'heldout_heights_bank',12,.06),'novel':(OUT/'novel_bank',16,0.)}
    for name,(bank,n,kick) in banks.items():
        sim(f'fixed_{name}','fixed',bank,n,impulse_vx=kick)
        assert read(OUT/f'fixed_{name}/results.json')['successes']==n
    for seed in protocol['seeds']:
        name=f'train_{seed}'
        sim(name,'train',STORE/'resting_seated_capture_dev02',seed=seed,arm='effort',iterations=64)
        v=read(OUT/name/'verification.json')
        assert v['updates']==64 and v['transitions']==65536 and v['finite']
        assert v['active_transitions']+v['warmup_transitions']==65536
        for variant in protocol['runtime_variants']:
            for label,(bank,n,kick) in banks.items():
                flags={'hold_goal':True} if variant=='ppo_hold' else {}
                sim(f'{label}_{seed}_{variant}','evaluate',bank,n,checkpoint=OUT/name/'final.pt',impulse_vx=kick,**flags)
    run('report',[str(PYTHON),'scripts/report_resting_bounded_02.py',str(OUT)])
    status('complete',report=str(OUT/'results/report.html'))
except BaseException:
    status('failed',traceback=traceback.format_exc());raise
