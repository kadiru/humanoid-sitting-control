"""Record only a verified active424 worker, preserving completed422/423 evidence."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def worker():
 s=subprocess.run(['ps','-p','1330771','-o','pid,etime,pcpu,args'],check=True,capture_output=True,text=True).stdout
 assert 'IsaacLab/.venv/bin/python scripts/build_low_seat_budget_path_424.py' in s and not (D/'budget_path_424/results.json').exists();return s

def main():
 assert not (D/'active_run_424.json').exists();worker();prior=[]
 for name in ['prior_hashes.json','completed_probes_preservation_421.json','supplemental_preservation_421.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  prior.append(dict(manifest=name,files=len(mf),passed=True))
 for name in ['budget_preparation_422','budget_preparation_audit_423']:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'budget_path_launch_424.json')['source_input_sha256'].items():assert sha(p)==h,p
 proc=worker();now=datetime.datetime.now(datetime.timezone.utc).isoformat();latest=[]
 for p in sorted((D/'budget_path_424').glob('node_*/history.json')):
  h=read(p);latest.append(dict(node=p.parent.name,callbacks=len(h),last=h[-1] if h else None))
 completed=read(D/'budget_path_424/nodes.json') if (D/'budget_path_424/nodes.json').exists() else []
 for p,name in [(D/'status.json','status_before_active_budget_424.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_active_budget_424.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 q=D/'prior_recheck_active_budget_424.json';q.write_text(json.dumps(dict(utc=now,checks=prior,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_421.json');mf.update(read(D/'supplemental_preservation_421.json'))
 for folder in ['budget_preparation_422','budget_preparation_audit_423']:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['budget_preparation_422.log','budget_preparation_audit_423.log','budget_path_launch_424.json','budget_path_launch_424.log','handoff_running_424.txt','status_before_active_budget_424.json','handoff_before_active_budget_424.md',q.name]:mf[str(D/name)]=sha(D/name)
 for name in ['low_seat_budget_domain_422.py','prepare_low_seat_budget_422.py','audit_low_seat_budget_preparation_423.py','build_low_seat_budget_path_424.py','freeze_low_seat_budget_launch_424.py','audit_low_seat_budget_path_425.py','audit_low_seat_budget_domain_425.py','report_low_seat_budget_426.py','record_low_seat_active_budget_424.py']:mf[str(R/'scripts'/name)]=sha(R/'scripts'/name)
 target=D/'budget_evidence_preservation_424.json';assert not target.exists();target.write_text(json.dumps(mf,indent=2))
 active=dict(utc=now,unified_exec_session=44383,worker_pid=1330771,worker=proc.strip(),actual_exit_status='PENDING',launched_once=True,maximum_nonlinear_nodes=5,maximum_iterations_per_node=30,maximum_total_iterations=150,latest_nodes=latest,completed_node_summaries=[{k:v for k,v in r.items() if k!='history'} for r in completed],audit_425_executed=False,report_426_executed=False,closure_427_executed=False,completed_manifest=str(target),completed_files=len(mf),manifest_sha256=sha(target),changing_main_log_and_outputs_excluded=True,prior_checks=prior,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,repair_complete=False,task_complete=False,prior_sealed_closure='closure_record_421.json')
 (D/'active_run_424.json').write_text(json.dumps(active,indent=2));notes=(D/'handoff_running_424.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_joint_event_budget_424_running',audit_jobs_running=False,model_jobs_running=True,physical_jobs_running=False,latest_active_run=str(D/'active_run_424.json'),latest_active_snapshot=None,latest_handoff=str(D/'handoff_running_424.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'budget_path_424'),latest_offline_audit=str(D/'budget_preparation_audit_423/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Preparation423 arithmetic complete; main424 active, independent425 unexecuted. No physical admission.',latest_offline_gate_scope='Bounded mixed-integer nonlinear whole-path domain; fractional numerical relaxations are not integer schedules.',latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_nlp_calls=None,latest_round_offline_milp_calls=0,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text());print(json.dumps(active,indent=2),flush=True)
if __name__=='__main__':main()
