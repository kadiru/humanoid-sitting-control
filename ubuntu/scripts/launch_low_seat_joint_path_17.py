"""One development-only physical probe; never starts PPO or holdout."""
import json,hashlib,subprocess,traceback,os
from pathlib import Path
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
OUT=D/'joint_path_17';STATE=D/'status_joint_path_17.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    prep=json.loads((D/'joint_path_preparation_17.json').read_text());assert prep['passed']
    OUT.mkdir(exist_ok=False)
    names=['low_seat_joint_path_controller_17.py','run_low_seat_joint_path_probe_17.py','launch_low_seat_joint_path_17.py','prepare_low_seat_joint_path_17.py','audit_low_seat_joint_path_17.py','low_seat_normal_duration_17.py','low_seat_load_duration_17.py','low_seat_contact_duration_17.py']
    sources={str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in names}
    (OUT/'revision.json').write_text(json.dumps(dict(scope='Pre-training research revision: selected measured low positive slopes14–18deg use an8–15s joint-space path to a0.9rad pelvis pitch,-0.17m rootx,nominal1mm pelvis gap. Command-arc timing keeps nominal rate<=0.15rad/s; applied slew and physical criteria unchanged. Original first10s lock explicitly becomes8s for six selected trials. All other episodes retain best guarded normal3s reference.',
        rationale='Recorded inward/pitch tracking errors and narrow housing clearance motivated a rate-feasible path. Earlier Cartesian path failed its rate screen. Joint path passed only three geometries; deepest36cm/+20deg stays unchanged and unresolved. Model checks are not dynamic proof.',
        retains_parent=True,first_prefix_s=8.,unselected_first20_unchanged_required=True,
        physical_gpu=1,trials=24,active_ticks=24000,initial_settling_excluded=600,indexed_warmup=0,optimizer_updates=0,fresh_holdout=False,
        sources=sources,preparation_sha256=sha(D/'joint_path_preparation_17.json'),config_sha256=sha(D/'normal_landing3s_guarded_09.json')),indent=2))
    try:
        baseline=D/'clearance_damped_01/baseline';(OUT/'baseline').symlink_to(baseline,target_is_directory=True)
        (OUT/'baseline_reuse.json').write_text(json.dumps(dict(source=str(baseline),physical_gpu=1,simulated_again=False,hashes={p.name:sha(p) for p in baseline.iterdir() if p.is_file()}),indent=2))
        cmd=feedback_command(OUT,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
        cmd[2]='../scripts/run_low_seat_joint_path_probe_17.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
        status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(OUT,'candidate',cmd)
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for script,folder in [('audit_low_seat_sensor_control.py',OUT/'candidate'),('audit_low_seat_joint_path_17.py',OUT)]:
            with (OUT/(script+'.log')).open('x') as f:subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/'+script,'--folder',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert all(sha(p)==h for p,h in sources.items())
        a=json.loads((OUT/'independent_joint_path_audit_17.json').read_text())
        status(STATE,'complete',summary=a['candidate'],lost_best_clean=a['lost_best_clean_cases'],gained_best_clean=a['gained_best_clean_cases'],
            remaining_target_cases=a['remaining_target_cases'],added_physical=a['added_physical_cases'],repair_complete=False,
            main_training_started=False,fresh_holdout_started=False)
    except Exception as e:status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
