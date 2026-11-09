"""One bounded physical development probe, with independent offline audits."""
import json,hashlib,subprocess,traceback,os
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
OUT=D/'normal_slow_orientation_08';STATE=D/'status_normal_slow_orientation_08.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    screen=json.loads((D/'slow_orientation_screen_08.json').read_text())
    assert len(screen['rows'])==24 and all(x['first_command_exact'] and x['ik_error']<.003 for x in screen['rows'])
    OUT.mkdir(exist_ok=False)
    source_names=['low_seat_slow_orientation_controller.py','run_low_seat_slow_orientation_probe.py',
        'launch_low_seat_slow_orientation.py','low_seat_normal_timing_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in source_names}
    (OUT/'revision.json').write_text(json.dumps(dict(
        scope='Pre-training physical hypothesis, not execution repair or admission. Modest0.55rad pitch,3s orientation then2s normal landing only measuredheight<39cm/slope>14deg; otherwise guarded best3s landing.',
        rationale='Recorded backward tipping near touchdown and delayed tracking during previous large fast rotation. Rate-lag is a supported hypothesis, not a sole-cause proof.',
        initial_approach_seconds=10,unchanged_command_slew_rad_s=.15,
        max_selected_planned_leg_target_rate_rad_s=max(max(x['max_leg_command_rate_rad_s']) for x in screen['rows'] if x['active']),
        scope_limit='Targets can still outpace applied slew slightly. Model housing clearance stays narrow; only physics decides success.',
        physical_gpu=1,episodes=24,active_ticks=24000,initial_settling_excluded=600,optimizer_updates=0,fresh_holdout=False,
        sources=sources,screen_sha256=sha(D/'slow_orientation_screen_08.json'),config_sha256=sha(D/'normal_landing3s_guarded_08.json')),indent=2))
    try:
        original=D/'clearance_damped_01/baseline';(OUT/'baseline').symlink_to(original,target_is_directory=True)
        (OUT/'baseline_reuse.json').write_text(json.dumps(dict(source=str(original),physical_gpu=1,simulated_again=False,
            hashes={p.name:sha(p) for p in original.iterdir() if p.is_file()}),indent=2))
        cmd=feedback_command(OUT,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_slow_orientation_probe.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_08.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(OUT,'candidate',cmd)
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src')
        for script,folder in [('audit_low_seat_contact_probe_v2.py',OUT),('audit_low_seat_sensor_control.py',OUT/'candidate')]:
            with (OUT/(script+'.log')).open('x') as f:
                subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/'+script,'--folder',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert all(sha(p)==h for p,h in sources.items())
        audit=json.loads((OUT/'independent_audit_v2.json').read_text())
        status(STATE,'complete',summary=audit['candidate'],regression_gate_passed=audit['regression_gate_passed'],
            added_physical=audit['added_physical_failure_cases'],lost_clean=audit['lost_clean_cases'],
            repair_complete=audit['repair_complete'],main_training_started=False,fresh_holdout_started=False)
    except Exception as e:
        status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
