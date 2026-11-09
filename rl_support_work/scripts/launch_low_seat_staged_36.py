"""One isolated24-case development probe; never starts retention/PPO/holdout."""
import json,hashlib,subprocess,traceback,os,datetime,shutil
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
OUT=D/'staged_approach_36';STATE=D/'status_staged_approach_36.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    prep=json.loads((D/'staged_preparation_36/preparation.json').read_text());assert prep['passed'] and len(prep['rows'])==8
    assert all(sha(p)==h for p,h in prep['source_input_hashes'].items())
    OUT.mkdir(exist_ok=False);(OUT/'frozen_source').mkdir()
    names=['low_seat_staged_planner_36.py','low_seat_staged_controller_36.py','low_seat_normal_duration_36.py','low_seat_load_duration_36.py','low_seat_contact_duration_36.py','run_low_seat_staged_36.py','launch_low_seat_staged_36.py','audit_low_seat_staged_36.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in names}
    for n in names:shutil.copy2(ROOT/'scripts'/n,OUT/'frozen_source'/n)
    revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='PRE-TRAINING bounded staged-controller revision. Selected measuredheight<38cm and14<slope<21deg uses actual1s sensors, fixed feet, same previously tested-.35rad shoulder/elbow offset and0.9rad pelvis target. A model solve chooses rootx at5.5mm rear CoP margin. Approach a20mm hover, pause0.583s, descend normally1.75s, hold0.583s by15s. All8target rows included; no early61111 pulse in isolating probe.',
      hypothesis='31 reduced tracking lag but touched housing during ongoing approach.35 showed earlier staged landing fits the time budget with tested rate headroom. This trial tests stopping and normal landing, not larger arm/lean sweeps.33/34 did NOT find a7mm supported envelope for the two harder geometries. Existing3mm model screen is retained explicitly; minimum3.589mm is NOT a robust safety certificate.',
      changes='Selected prefix becomes1s explicitly; applied cap0.20rad/s only1<=t<15, nominal<=0.175, then0.15. Filter0.1 and all feedback gains unchanged. Foot anchors use the exact planning timestamp sensors rather than later callback. Missing/aged current geometry pauses the shared reference/landing/load path clock before and during descent; physical/outcome clocks continue and no old plane is relabeled fresh. Frozen59117 remains11/15s absolute.',
      fixed='All physical/material/slip/forbidden-body-or-illegal-patch criteria,14finger commands,reference-assisted sensor-only contract and frozen resting parent unchanged.',
      first_exact_seconds=1.,unselected_full20_exactness='Reported honestly; existing renderer coupling may break bitwise equality. Old17/31 failed gates remain failed. All64 current successes and all8target failures must be resolved in full72matched tests before PPO.',
      physical_gpu=1,trials=24,active_ticks=24000,initial_settling_excluded=600,indexed_warmup=0,optimizer_updates=0,fresh_holdout=False,automatic_followup=False,sources=sources,preparation_sha256=sha(D/'staged_preparation_36/preparation.json'),config_sha256=sha(D/'normal_landing3s_guarded_09.json'))
    (OUT/'revision.json').write_text(json.dumps(revision,indent=2))
    try:
        baseline=D/'clearance_damped_01/baseline';(OUT/'baseline').symlink_to(baseline,target_is_directory=True)
        (OUT/'baseline_reuse.json').write_text(json.dumps(dict(source=str(baseline),physical_gpu=1,simulated_again=False,hashes={p.name:sha(p) for p in baseline.iterdir() if p.is_file()}),indent=2))
        protocol=ROOT/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,OUT/'protocol_before_36.md')
        with protocol.open('a') as f:f.write('\n\nPre-training revision36: '+revision['scope']+' '+revision['hypothesis']+' '+revision['changes']+' One24-case development probe,24000active+600settling,zerooptimizer/holdout; no automatic followup. See staged_approach_36/revision.json.\n')
        cmd=feedback_command(OUT,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_staged_36.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(OUT,'candidate',cmd)
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for script,folder in [('audit_low_seat_sensor_control.py',OUT/'candidate'),('audit_low_seat_staged_36.py',OUT)]:
            with (OUT/(script+'.log')).open('x') as f:subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/'+script,'--folder',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert all(sha(p)==h for p,h in sources.items())
        a=json.loads((OUT/'independent_staged_audit_36.json').read_text())
        status(STATE,'complete',summary=a['candidate'],lost_best_clean=a['lost_best_clean_cases'],gained_best_clean=a['gained_best_clean_cases'],remaining_target_cases=a['remaining_target_cases'],added_physical=a['added_physical_cases'],software_audit_passed=a['software_audit_passed'],exact_preservation_gate_passed=a['exact_preservation_gate_passed'],repair_complete=False,main_training_started=False,fresh_holdout_started=False)
    except Exception as e:status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
