"""Matched development retention only; never opens fresh holdout or trains."""
import json,hashlib,subprocess,traceback,os
from pathlib import Path
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
ROOT=Path('/home/kadir/humanoid-sitting');PY=ROOT/'IsaacLab/.venv/bin/python'
OLD=D.parent/'low_positive_transfer_01/ppo_01/seed_62111'
import time
while True:
    previous=json.loads((D/'current_retention_status_09.json').read_text())
    if previous['phase']=='complete':break
    assert previous['phase']!='failed',previous
    time.sleep(20)
# Admission is evaluated from completed and independently audited physics.
while True:
    probe_file=D/'status_normal_counterbalance_13.json'
    probe=json.loads(probe_file.read_text()) if probe_file.exists() else {}
    assert probe.get('phase')!='failed',probe
    if probe.get('phase')=='complete':break
    time.sleep(20)
from audit_low_seat_contact_probe_v2 import clean
probe_root=D/'normal_counterbalance_13'
assert json.loads((probe_root/'counterbalance_independent_audit_13.json').read_text())['passed']
rows=json.loads((probe_root/'candidate/results.json').read_text())['episodes']
best=json.loads((D/'normal_landing3s_06/candidate/results.json').read_text())['episodes']
targets=[0,1,2,3,6,7,10,11]
lost_best=[i for i in range(24) if clean(best[i]) and not clean(rows[i])]
failed_target=[i for i in targets if not clean(rows[i])]
physical=[i for i,r in enumerate(rows) if any(r['failures'].values())]
if lost_best or failed_target or physical:
    status(D/'current_retention_status_14.json','not_admitted',lost_best_probe_cases=lost_best,
        remaining_target_cases=failed_target,physical_cases=physical,
        reason='The physical counterbalance prerequisite failed; no full retention, PPO or holdout launched.',
        main_training_started=False,fresh_holdout_started=False)
    raise SystemExit(0)
OUT=D/'current_retention_14';OUT.mkdir(exist_ok=False)
STATE=D/'current_retention_status_14.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src')
def read(p):return json.loads(Path(p).read_text())
def clean(r):return r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks']
try:
    targets=read(D/'retention_targets.json')['banks'];reports={}
    source_names=['run_low_seat_current_counterbalance_probe.py','launch_low_seat_counterbalance_retention_14.py','low_seat_counterbalance_controller.py','low_seat_normal_timing_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in source_names}
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2))
    (OUT/'revision.json').write_text(json.dumps(dict(
        scope='Frozen early61111 at6-7s plus frozen59117 at11/15s; guarded best normal landing plus bounded arm counterbalance on low steep rows, following successful24-case physical admission. All three banks are development.',
        physical_gpu=0,episodes=72,active_ticks=72000,initial_settling_ticks_excluded=1800,
        optimizer_updates=0,fresh_holdout_opened=False,baseline_reused=True,
        physical_prerequisite='Retain all64 successful cases of verified retention09, including all60 prior union successes; remove the eight remaining failures before PPO.'),indent=2))
    for bank,local in [('development_old','known_old'),('development_transfer','known_transfer'),('holdout','known_recent')]:
        # The old study's holdout has explicitly become development for this study.
        label=bank+'_current_61111';base=OLD/label;folder=OUT/local;folder.mkdir()
        (folder/'baseline').symlink_to(base,target_is_directory=True)
        (folder/'baseline_reuse.json').write_text(json.dumps(dict(source=str(base),physical_gpu=0,
            simulated_again=False,hashes={p.name:sha(p) for p in base.iterdir() if p.is_file()}),indent=2))
        cmd=read(OLD/(label+'.command.json'));cmd[2]='../scripts/run_low_seat_current_counterbalance_probe.py'
        oldbank=Path(cmd[cmd.index('--bank')+1]);newbank=D/'banks'/local
        for n in ['geometry.json']:
            assert read(oldbank/n)==read(newbank/n),(bank,n)
        cmd[cmd.index('--bank')+1]=str(newbank);cmd[cmd.index('--output')+1]=str(folder/'candidate')
        cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
        status(STATE,'evaluating',bank=local,completed=list(reports),main_training_started=False,fresh_holdout_started=False)
        run(folder,'candidate',cmd)
        for script,args in [('audit_low_seat_contact_probe_v2.py',['--folder',str(folder)]),
                            ('audit_low_seat_sensor_control.py',['--folder',str(folder/'candidate')])]:
            with (folder/(script+'.log')).open('x') as f:
                subprocess.run([str(PY),'scripts/'+script]+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        import numpy as np
        with np.load(base/'episode_decisions.npz') as b,np.load(folder/'candidate/episode_decisions.npz') as c:
            assert np.array_equal(b['early_observation'],c['early_observation'])
            assert np.array_equal(b['early_action'],c['early_action'])
        audit=read(folder/'independent_audit_v2.json')
        assert all(all(v.values()) for v in audit['exact_startups'].values())
        assert all(v['exact'] for v in audit['locked_first10'].values())
        rows=read(folder/'candidate/results.json')['episodes'];expected=targets[bank]['cases']
        previous_rows=read(D/'current_retention_09'/local/'candidate/results.json')['episodes']
        required=[i for i,r in enumerate(previous_rows) if clean(r)]
        retained=[i for i in required if clean(rows[i])];lost=[i for i in required if not clean(rows[i])]
        reports[local]=dict(summary=audit['candidate'],required=len(required),retained=len(retained),lost=lost,
            gained_beyond_union=[int(i) for i,x in expected.items() if not x['successful_controllers'] and clean(rows[int(i)])],
            early414_inputs_and_actions_exact=True,exact_first10=True)
        (OUT/'retention.json').write_text(json.dumps(reports,indent=2))
    assert all(sha(p)==v for p,v in sources.items())
    status(STATE,'complete',reports=reports,retained=sum(r['retained'] for r in reports.values()),required=64,
        repair_complete=all(r['summary']['clean']==24 for r in reports.values()),
        main_training_started=False,fresh_holdout_started=False)
except Exception as e:
    status(STATE,'failed',error=repr(e),traceback=traceback.format_exc(),main_training_started=False,fresh_holdout_started=False)
    raise
