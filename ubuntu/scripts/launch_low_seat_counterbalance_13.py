"""One bounded physical development probe, with independent offline audits."""
import json,hashlib,subprocess,traceback,os
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
OUT=D/'normal_counterbalance_13';STATE=D/'status_normal_counterbalance_13.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    screen=json.loads((D/'counterbalance_screen_13.json').read_text())
    assert screen['passed'] and len(screen['rows'])==24 and all(x['first_command_exact'] and x['ik_error']<.003 for x in screen['rows'])
    OUT.mkdir(exist_ok=False)
    source_names=['low_seat_counterbalance_controller.py','run_low_seat_counterbalance_probe.py',
        'launch_low_seat_counterbalance_13.py','low_seat_normal_timing_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in source_names}
    (OUT/'revision.json').write_text(json.dumps(dict(
        scope='One pre-training physical hypothesis: guarded normal3s landing, plus symmetric shoulder/elbow pitch -0.35rad over3.5s only measuredheight<38cm and slope>14deg. Model leg feedforward recomputed for the changed arm mass distribution. No pelvis-path or feedback-gain change.',
        rationale='The failed housing events coincide with inward root error and backward pitch drift. Model arm counterbalance moves COM forward without rotating pelvis or requiring a larger leg-path excursion. This is a hypothesis, not causal proof; narrow housing clearance remains.',
        initial_approach_seconds=10,unchanged_command_slew_rad_s=.15,
        max_selected_arm_target_rate_rad_s=max(x['arm_peak_target_rate_rad_s'] for x in screen['rows']),
        scope_limit='Model housing clearance stays narrow. New reference arm commands are distinct from the zero learned body residual. All14 fingers stay fixed; physical outcomes decide admission.',
        physical_gpu=1,episodes=24,active_ticks=24000,initial_settling_excluded=600,optimizer_updates=0,fresh_holdout=False,
        sources=sources,screen_sha256=sha(D/'counterbalance_screen_13.json'),config_sha256=sha(D/'normal_landing3s_guarded_09.json')),indent=2))
    try:
        original=D/'clearance_damped_01/baseline';(OUT/'baseline').symlink_to(original,target_is_directory=True)
        (OUT/'baseline_reuse.json').write_text(json.dumps(dict(source=str(original),physical_gpu=1,simulated_again=False,
            hashes={p.name:sha(p) for p in original.iterdir() if p.is_file()}),indent=2))
        cmd=feedback_command(OUT,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_counterbalance_probe.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(OUT,'candidate',cmd)
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for script,folder in [('audit_low_seat_contact_probe_v2.py',OUT),('audit_low_seat_sensor_control.py',OUT/'candidate')]:
            with (OUT/(script+'.log')).open('x') as f:
                subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/'+script,'--folder',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        import torch,numpy as np
        torch.set_num_threads(2)
        from validate_low_positive_feedback_v3 import validate
        v=validate(OUT/'candidate')
        checks=[];paths=np.load(OUT/'candidate/last_rolling_paths.npz')
        for a in json.loads((OUT/'candidate/rolling_plan_audit.json').read_text()):
            info=a['counterbalance_probe']
            if not info['applied']:continue
            ids=info['arm_joint_ids'];actual=paths['command'][a['case']][:,ids]
            before=np.asarray(info['reference_arm_before'])
            delta=np.asarray([r['arm_delta_rad'] for r in info['model_path']])
            assert np.allclose(actual-before,delta[:,None],atol=2e-7,rtol=0.)
            assert abs(actual-before).max()<=.350001
            checks.append(dict(case=a['case'],maximum_reference_arm_offset_rad=float(abs(actual-before).max())))
        trace=np.load(OUT/'candidate/trace.npz');rate=float(abs(np.diff(trace['command'],axis=0)).max())
        assert rate<=.015001
        (OUT/'counterbalance_independent_audit_13.json').write_text(json.dumps(dict(passed=True,
            actual_sensor_parent_reward_accounting=v,arm_reference_checks=checks,
            sampled_command_step_rad_per_0p1s=rate,physical_admission_not_implied=True),indent=2))
        assert all(sha(p)==h for p,h in sources.items())
        audit=json.loads((OUT/'independent_audit_v2.json').read_text())
        status(STATE,'complete',summary=audit['candidate'],regression_gate_passed=audit['regression_gate_passed'],
            added_physical=audit['added_physical_failure_cases'],lost_clean=audit['lost_clean_cases'],
            repair_complete=audit['repair_complete'],main_training_started=False,fresh_holdout_started=False)
    except Exception as e:
        status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
