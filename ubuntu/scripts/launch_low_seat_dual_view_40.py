"""One bounded24-case camera-fallback development experiment. No PPO/holdout."""
from pathlib import Path
import json,hashlib,shutil,subprocess,os,traceback,datetime
from low_positive_common import feedback_command
from run_transfer_assessment import run,status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting');OUT=D/'dual_view_approach_40';STATE=D/'status_dual_view_approach_40.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 prep=json.loads((D/'dual_view_preparation_40/preparation.json').read_text());assert prep['passed'] and all(sha(p)==h for p,h in prep['source_input_hashes'].items())
 assert not OUT.exists();OUT.mkdir();(OUT/'frozen_source').mkdir()
 names=['low_seat_dual_view_40.py','run_low_seat_dual_view_40.py','prepare_low_seat_dual_view_40.py','audit_low_seat_dual_view_40.py','launch_low_seat_dual_view_40.py','low_seat_staged_planner_36.py','low_seat_staged_controller_36.py','low_seat_normal_duration_36.py','low_seat_load_duration_36.py','low_seat_contact_duration_36.py','audit_low_seat_staged_36.py']
 paths=[ROOT/'scripts'/n for n in names]+[ROOT/'scripts/run_supported_transition.py',ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',ROOT/'sitting_task/src/humanoid_sitting/support_camera.py']
 sources={str(p):sha(p) for p in paths}
 for p in paths:shutil.copy2(p,OUT/'frozen_source'/p.name)
 revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='PRE-TRAINING sensor-coverage research revision. Add the already code-defined downward camera at the mount verified in diagnostic39. Original rear measurement remains preferred. Only measured-selected staged36 rows with no current rear fit can use a CURRENT down fit, under unchanged model self-mask, plane conditioning, floor exclusion, largest-patch and prior-association checks. Other rows never consume down geometry.',evidence='Diagnostic39:15600active+600settling,13s recording only, no complete20s episode/optimizer. All original startup/full13s physical,command,reference,geometry,torque arrays EXACT. Down fits all132target frames with unavailable rear tracker; maxheight1.565mm/slope0.132deg there. Both camera mounts verified; actual parent414/CPU inference passed. This does not prove motion safety.',fixed='The staged36 1–15s path, planner, fixed foot anchors, -.35rad arm offsets, .9rad pelvis target, nominal.175/applied.20 cap, filter.1, existing feedback and invalid-geometry pause remain unchanged. Frozen59117 acts11/15s absolute. No early61111 pulse in isolated probe. All physical/material/slip/contact/support criteria and14fingers unchanged. No fit thresholds lowered or truth/old baseline sensors substituted.',first_exact_seconds=1.,unselected_full20_exactness='Measure/report without tolerance change; no exact-preservation claim if rendered measurements diverge.',physical_gpu=1,environments=24,episode_seconds=20.,active_ticks=24000,initial_settling_excluded=600,indexed_warmup=0,optimizer_updates=0,main_training=False,fresh_holdout=False,automatic_followup=False,sources=sources,preparation_sha256=sha(D/'dual_view_preparation_40/preparation.json'))
 (OUT/'revision.json').write_text(json.dumps(revision,indent=2));baseline=D/'clearance_damped_01/baseline';(OUT/'baseline').symlink_to(baseline,target_is_directory=True)
 (OUT/'baseline_reuse.json').write_text(json.dumps(dict(source=str(baseline),simulated_again=False,physical_gpu=1,hashes={p.name:sha(p) for p in baseline.iterdir() if p.is_file()}),indent=2))
 protocol=ROOT/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,OUT/'protocol_before_40.md')
 with protocol.open('a') as f:f.write('\n\nPre-training revision40: '+revision['scope']+' '+revision['evidence']+' '+revision['fixed']+' One24-case full20s development probe,24000active+600settling,zerooptimizer/holdout. No automatic retention or PPO. See dual_view_approach_40/revision.json.\n')
 try:
  cmd=feedback_command(OUT,1,'candidate','zero',D/'banks/probe',D.parent/'low_positive_transfer_01/learning_reference.json',kind='original',seed=63101,start=10.)
  cmd[2]='../scripts/run_low_seat_dual_view_40.py';cmd+=['--landing-config',str(D/'normal_landing3s_guarded_09.json')]
  status(STATE,'running',main_training_started=False,fresh_holdout_started=False);run(OUT,'candidate',cmd)
  status(STATE,'auditing',main_training_started=False,fresh_holdout_started=False)
  env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
  for script,folder in [('audit_low_seat_sensor_control.py',OUT/'candidate'),('audit_low_seat_staged_36.py',OUT),('audit_low_seat_dual_view_40.py',OUT)]:
   with (OUT/(script+'.log')).open('x') as f:subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/'+script,'--folder',str(folder)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert all(sha(p)==h for p,h in sources.items())
  a=json.loads((OUT/'independent_dual_view_audit_40.json').read_text());s=a['staged_audit']
  status(STATE,'complete',summary=s['candidate'],lost_best_clean=s['lost_best_clean_cases'],gained_best_clean=s['gained_best_clean_cases'],remaining_target_cases=s['remaining_target_cases'],added_physical=s['added_physical_cases'],software_audit_passed=True,exact_preservation_gate_passed=s['exact_preservation_gate_passed'],repair_complete=False,main_training_started=False,fresh_holdout_started=False)
 except Exception as e:status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
