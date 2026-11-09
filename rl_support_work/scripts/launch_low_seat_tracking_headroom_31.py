"""One development-only physical probe; never starts PPO or holdout."""
import json,hashlib,subprocess,traceback,os
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
OUT=D/'tracking_headroom_31';STATE=D/'status_tracking_headroom_31.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    prior=json.loads((D/'joint_path_17/independent_joint_path_audit_v4_17.json').read_text())
    assert prior['input_action_reward_and_planner_checks_passed'] and not prior['physical_admission_passed']
    assert not prior['exact_preservation_gate_passed']
    assert all(x['screen']['peak_nominal_rate_rad_s']<=.15 for x in prior['planner_replay'])
    OUT.mkdir(exist_ok=False)
    names=['low_seat_tracking_headroom_31.py','run_low_seat_tracking_headroom_31.py','audit_low_seat_tracking_headroom_31.py','launch_low_seat_tracking_headroom_31.py','low_seat_joint_path_controller_17.py','low_seat_normal_duration_17.py','low_seat_load_duration_17.py','low_seat_contact_duration_17.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in names}
    (OUT/'revision.json').write_text(json.dumps(dict(scope='PRE-TRAINING CONTROLLER RESEARCH: isolate the actuator command-rate limitation identified in recorded joint17 failures. Same8–15s nominal joint17 path and planner, filter0.1, feedback gains, frozen resting59117, materials and physical criteria. Applied rate is explicitly increased from0.15 to0.20rad/s ONLY on the six measured-selected approach rows while8<=t<15; original0.15 resumes at15s. This is NOT an execution repair or physical-threshold relaxation.',
        rationale='All six new joint17 housing events had zero sensor-based unloading in preceding2s.0.1s samples show near-cap commands in80–100percent of intervals and0.052–0.063rad reference-command lag. Existing nominal path<=0.15 leaves limited tracking headroom. New0.20 cap tests this specific hypothesis, not a guarantee of repair. Offline staged approach28 failed timing under existing cap; early-anchor29 still did not supply7mm reserve on deepest case.',
        prior_joint17_gates_remain_failed=True,first8_exact_required=True,unselected_control_source_and_cap_unchanged_required=True,
        unselected_full20_trace_exactness_reported_separately='Known renderer coupling is not repaired; do not assume unchanged trajectories or effort. Any exactness failure remains reported. Physical retention of all64 successes across72 cases remains mandatory before PPO.',
        deepest36cm20deg='Kept at best normal3s, remains unresolved; no full repair claim possible from only six targeted fixes.',
        physical_gpu=1,trials=24,active_ticks=24000,initial_settling_excluded=600,indexed_warmup=0,optimizer_updates=0,fresh_holdout=False,
        sources=sources,prior_planner_audit_sha256=sha(D/'joint_path_17/independent_joint_path_audit_v4_17.json'),config_sha256=sha(D/'normal_landing3s_guarded_09.json')),indent=2))
    try:
        baseline=D/'clearance_damped_01/baseline';(OUT/'baseline').symlink_to(baseline,target_is_directory=True)
        (OUT/'baseline_reuse.json').write_text(json.dumps(dict(source=str(baseline),physical_gpu=1,simulated_again=False,hashes={p.name:sha(p) for p in baseline.iterdir() if p.is_file()}),indent=2))
        cmd=feedback_command(OUT,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_tracking_headroom_31.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(OUT,'candidate',cmd)
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for script,folder in [('audit_low_seat_sensor_control.py',OUT/'candidate'),('audit_low_seat_tracking_headroom_31.py',OUT)]:
            with (OUT/(script+'.log')).open('x') as f:subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/'+script,'--folder',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert all(sha(p)==h for p,h in sources.items())
        a=json.loads((OUT/'independent_tracking_audit_31.json').read_text())
        status(STATE,'complete',summary=a['candidate'],lost_best_clean=a['lost_best_clean_cases'],gained_best_clean=a['gained_best_clean_cases'],
            remaining_target_cases=a['remaining_target_cases'],added_physical=a['added_physical_cases'],repair_complete=False,software_audit_passed=a['software_audit_passed'],exact_preservation_gate_passed=a['exact_preservation_gate_passed'],
            main_training_started=False,fresh_holdout_started=False)
    except Exception as e:status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
