"""Audit-only numbered-finger-name correction; never reruns physical simulation."""
from pathlib import Path
import json,shutil,hashlib,subprocess,os,time,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');OUT=D/'tracking_headroom_audit_recovery_31';STATE=D/'tracking_headroom_audit_recovery_status_31.json'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(phase,**kw):STATE.write_text(json.dumps(dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
def main():
 status('waiting_for_original_audit',simulations=0)
 for _ in range(60):
  s=read(D/'status_tracking_headroom_31.json')
  if s['phase']!='running':break
  time.sleep(15)
 assert s['phase']=='failed',s
 folder=D/'tracking_headroom_31';log=folder/'audit_low_seat_tracking_headroom_31.py.log';text=log.read_text();assert 'len(finger)==14' in text and 'AssertionError' in text,text[-2000:]
 exit_record=read(folder/'candidate.exit.json');assert exit_record['returncode']==0,exit_record
 assert (folder/'candidate/command_headroom_ticks_31.npz').exists()
 OUT.mkdir(exist_ok=False)
 originals=[R/'scripts/audit_low_seat_tracking_headroom_31.py',D/'status_tracking_headroom_31.json',log,D/'tracking_headroom_31.log']
 for p in originals:shutil.copy2(p,OUT/p.name)
 recovery=dict(scope='Offline audit-only repair: original auditor looked for thumb/index/middle names absent from this model. Use the exact numbered zero/one/two/three/four/five/six finger names, as the existing sensor auditor does.14finger count and bit-exact fixed-command requirement unchanged. No simulation, runtime, bound, reward or physical-criterion changes.',originals={str(p):sha(p) for p in originals},corrected_source=str(R/'scripts/audit_low_seat_tracking_headroom_v2_31.py'),corrected_source_sha256=sha(R/'scripts/audit_low_seat_tracking_headroom_v2_31.py'),physical_candidate_exit=exit_record,simulations=0)
 (OUT/'recovery.json').write_text(json.dumps(recovery,indent=2));status('auditing',simulations=0)
 env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 with (OUT/'audit_v2.log').open('x') as f:subprocess.run([str(R/'IsaacLab/.venv/bin/python'),'scripts/audit_low_seat_tracking_headroom_v2_31.py','--folder',str(folder)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 a=read(folder/'independent_tracking_audit_v2_31.json');assert a['software_audit_passed']
 revision=read(folder/'revision.json');assert all(sha(p)==h for p,h in revision['sources'].items())
 status('complete',candidate=a['candidate'],lost_best_clean=a['lost_best_clean_cases'],gained_best_clean=a['gained_best_clean_cases'],added_physical=a['added_physical_cases'],remaining_target_cases=a['remaining_target_cases'],software_audit_passed=True,exact_preservation_gate_passed=a['exact_preservation_gate_passed'],repair_complete=False,main_training_started=False,fresh_holdout_started=False)
if __name__=='__main__':main()
