"""Persistent, gated four-height resting PPO study. Run directly, never in an agent."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data/humanoid-sitting-perception/resting_ppo_01')
CAPTURE=Path('/data/humanoid-sitting-perception/resting_seated_capture_dev02')
FIXED=Path('/data/humanoid-sitting-perception/resting_fixed_dev01')
PYTHON=ROOT/'IsaacLab/.venv/bin/python'
OUT.mkdir(parents=True,exist_ok=False); (OUT/'frozen').mkdir()
completed=[]; current=None


def status(phase,**extra):
    row=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),job=current,completed=completed,**extra)
    temp=OUT/'status.tmp'; temp.write_text(json.dumps(row,indent=2)+'\n'); temp.replace(OUT/'status.json')
    print(json.dumps(row),flush=True)


def run(name,command):
    global current
    current=name; status('running')
    for path,digest in frozen.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (OUT/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (OUT/f'{name}.log').open('w') as f:
        result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; inspect preserved log')
    completed.append(name)


def sim(name,mode,bank=None,num=16,seed=52911,seconds=20.,**extra):
    command=['bash',str(ROOT/'scripts/run_resting_gpu1.sh'),'python','-u','../scripts/run_resting_posture.py',
        '--mode',mode,'--output',str(OUT/name),'--num-envs',str(num),'--seconds',str(seconds),'--seed',str(seed),
        '--device','cuda:0','--kit_args=--/renderer/activeGpu=1,--/renderer/multiGpu/enabled=false']
    if bank:command+=['--bank',str(bank)]
    for key,value in extra.items():
        command.append('--'+key.replace('_','-'))
        if value is not True:command+=([str(v) for v in value] if isinstance(value,list) else [str(value)])
    run(name,command)


paths=[Path(__file__).resolve(),ROOT/'scripts/run_resting_posture.py',ROOT/'scripts/run_resting_gpu1.sh',
       ROOT/'scripts/report_resting_ppo_01.py',ROOT/'scripts/plot_resting_posture.py',
       ROOT/'scripts/test_resting_objective.py',ROOT/'scripts/test_resting_reward.py',
       ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py',ROOT/'sitting_task/src/humanoid_sitting/resting_reward.py',
       ROOT/'runs/g1_kinematics_geometry_04/model.json',CAPTURE/'candidates.npz',CAPTURE/'capture.json',FIXED/'results.json']
paths+=list((ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
paths+=[ROOT/f'sitting_task/src/humanoid_sitting/{n}.py' for n in ('shared_frame','support_landing_geometry','support_camera','deployment')]
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for i,p in enumerate(paths):shutil.copy2(p,OUT/'frozen'/f'{i:02}_{p.name}')
(OUT/'frozen/manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
try:
    baseline=json.loads((FIXED/'results.json').read_text())
    assert baseline['successes']==16 and len(baseline['episodes'])==16
    reference=float(np.mean([r['metrics_on_good']['normalized_torque_squared'] for r in baseline['episodes']]))
    assert 0<reference<1
    protocol=dict(scope='Maintain supported rest on level rigid surfaces; no approach/sit transition or leaning claim',
        seeds=[52911,52917],arms=['stability','effort'],updates=128,steps_per_update=64,num_envs=16,
        transitions_per_run=131072,total_transitions=524288,training_heights_m=[.25,.30,.35,.40],
        heldout_heights_m=[.275,.325,.375],effort_reference=reference,calibration=str(FIXED/'results.json'),
        reward='2 per qualifying second; effort arm subtracts E/(E+reference) per qualifying second; -5 once on physical failure',
        E='mean over ALL joints of (applied torque / existing peak effort limit)^2; not electrical energy',
        gamma=.998,lam=.97,learning_rate=.0001,epochs=3,minibatches=4,entropy=.001,initial_std=.15,
        action='23 joint-target increments capped 0.01 rad per 20ms; joint/effort limits retained; no timed nominal controller',
        reset='Full root/joint position and velocity plus held command; .5s ordinary-tick settling with zero reward during resets; warmup transitions separately counted',
        runtime='324 sensor-only channels: three frames of q,qd,torque,gravity,gyro,previous command and observed local plane; same critic inputs',
        evaluation='28 unique held-out starts:16 at training heights and12 at intermediate heights; plus same12 with +/-0.06m/s base-velocity impulse at5s;20s per trial; all four final models and fixed PD',
        success='no latched failure,>=95% qualifying rest ticks, final2s continuously qualifying; integer counters',
        startup_parity_absolute_tolerance=1e-6,
        limitations=['Two training seeds; perturbation cases repeat the intermediate-height placements',
            'Ideal simulated encoder/IMU/torque equivalents; depth is measured, no runtime simulator geometry',
            'Finite rigid floor blocks permit measured friction; top z=0, materials .8/.6; no inclined or compliant surfaces',
            'Highest initial posture has light seat support and substantial foot load; report load alongside effort'],
        no_automatic_promotion=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    run('unit_tests',[str(PYTHON),str(ROOT/'scripts/test_resting_reward.py'),'-v'])
    sim('reset_probe','fixed',CAPTURE,reset_probe=True)
    assert json.loads((OUT/'reset_probe/reset_probe.json').read_text())['passed']
    # Independent physical trajectories yield held-out starts, not arbitrary
    # joint noise that could create penetrating or unsupported initial states.
    sim('heldout_starts_bank','capture',seed=53003,seconds=24.)
    sim('heldout_heights_bank','capture',num=12,seed=53007,seconds=24.,heights=[.275,.325,.375])
    banks={'starts':(OUT/'heldout_starts_bank',16,0.),
           'heights':(OUT/'heldout_heights_bank',12,0.),
           'impulse':(OUT/'heldout_heights_bank',12,.06)}
    for bank,num,impulse in banks.values():
        info=json.loads((bank/'capture.json').read_text()); assert info['valid_count']==num
    for label,(bank,num,impulse) in banks.items():
        sim(f'fixed_{label}','fixed',bank,num=num,seed=53011,impulse_vx=impulse,effort_reference=reference)
    for seed,arm in ((52911,'stability'),(52911,'effort'),(52917,'effort'),(52917,'stability')):
        name=f'train_{seed}_{arm}'
        sim(name,'train',CAPTURE,seed=seed,arm=arm,iterations=128,effort_reference=reference)
        check=json.loads((OUT/name/'verification.json').read_text())
        assert check['updates']==128 and check['transitions']==131072 and check['finite']
        assert check['active_transitions']+check['warmup_transitions']==131072
        for label,(bank,num,impulse) in banks.items():
            sim(f'{label}_{seed}_{arm}','evaluate',bank,num=num,seed=53011,
                checkpoint=OUT/name/'final.pt',arm=arm,impulse_vx=impulse,effort_reference=reference)
    run('report',[str(PYTHON),str(ROOT/'scripts/report_resting_ppo_01.py'),str(OUT)])
    status('completed',report=str(OUT/'results/report.html'))
except BaseException as e:
    status('failed',error=repr(e)); raise
