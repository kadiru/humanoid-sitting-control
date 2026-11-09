"""Matched development retention only; never opens fresh holdout or trains."""
import json,hashlib,subprocess,traceback,os
from pathlib import Path
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
ROOT=Path('/home/kadir/humanoid-sitting');PY=ROOT/'IsaacLab/.venv/bin/python'
OLD=D.parent/'low_positive_transfer_01/ppo_01/seed_62111'
import time
while True:
    previous=json.loads((D/'current_retention_status_07.json').read_text())
    if previous['phase']=='complete':break
    assert previous['phase']!='failed',previous
    time.sleep(20)
OUT=D/'current_retention_08';OUT.mkdir(exist_ok=False)
STATE=D/'current_retention_status_08.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src')
def read(p):return json.loads(Path(p).read_text())
def clean(r):return r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks']
try:
    targets=read(D/'retention_targets.json')['banks'];reports={}
    source_names=['run_low_seat_current_timing_probe.py','launch_low_seat_current_retention_v2.py','low_seat_normal_timing_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in source_names}
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2))
    (OUT/'revision.json').write_text(json.dumps(dict(
        scope='Frozen early61111 at6-7s plus frozen59117 at11/15s; same best normal landing with documented measured-slope selector guard (>10.5deg). All three banks are development.',
        physical_gpu=0,episodes=72,active_ticks=72000,initial_settling_ticks_excluded=1800,
        optimizer_updates=0,fresh_holdout_opened=False,baseline_reused=True,
        physical_prerequisite='All60 union successes must survive; unresolved targeted slips and housing contacts must also be removed before PPO.'),indent=2))
    for bank,local in [('development_old','known_old'),('development_transfer','known_transfer'),('holdout','known_recent')]:
        # The old study's holdout has explicitly become development for this study.
        label=bank+'_current_61111';base=OLD/label;folder=OUT/local;folder.mkdir()
        (folder/'baseline').symlink_to(base,target_is_directory=True)
        (folder/'baseline_reuse.json').write_text(json.dumps(dict(source=str(base),physical_gpu=0,
            simulated_again=False,hashes={p.name:sha(p) for p in base.iterdir() if p.is_file()}),indent=2))
        cmd=read(OLD/(label+'.command.json'));cmd[2]='../scripts/run_low_seat_current_timing_probe.py'
        oldbank=Path(cmd[cmd.index('--bank')+1]);newbank=D/'banks'/local
        for n in ['geometry.json']:
            assert read(oldbank/n)==read(newbank/n),(bank,n)
        cmd[cmd.index('--bank')+1]=str(newbank);cmd[cmd.index('--output')+1]=str(folder/'candidate')
        cmd+=['--landing-config',str(D/'normal_landing3s_guarded_08.json')]
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
        required=[int(i) for i,x in expected.items() if x['successful_controllers']]
        retained=[i for i in required if clean(rows[i])];lost=[i for i in required if not clean(rows[i])]
        reports[local]=dict(summary=audit['candidate'],required=len(required),retained=len(retained),lost=lost,
            gained_beyond_union=[int(i) for i,x in expected.items() if not x['successful_controllers'] and clean(rows[int(i)])],
            early414_inputs_and_actions_exact=True,exact_first10=True)
        (OUT/'retention.json').write_text(json.dumps(reports,indent=2))
    assert all(sha(p)==v for p,v in sources.items())
    status(STATE,'complete',reports=reports,retained=sum(r['retained'] for r in reports.values()),required=60,
        repair_complete=all(r['summary']['clean']==24 for r in reports.values()),
        main_training_started=False,fresh_holdout_started=False)
except Exception as e:
    status(STATE,'failed',error=repr(e),traceback=traceback.format_exc(),main_training_started=False,fresh_holdout_started=False)
    raise
