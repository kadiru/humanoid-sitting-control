"""Seal completed416–420 evidence, preserving421 evidence. Guarded by actual exit, audit and local review records; no task completion."""
from pathlib import Path
import hashlib,json,datetime,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_421.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_415.json','supplemental_preservation_415.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['direct_preparation_416','direct_preparation_audit_417','direct_path_418','direct_path_audit_419','direct_report_draft_revision_420','progress_report_420'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 assert read(D/'direct_path_exit_418.json')['observed_exit_code']==0
 assert read(D/'direct_path_audit_419/independent_audit.json')['passed']
 post=read(D/'direct_postprocessing_420.json');assert post['audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_420_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 for p,h in read(D/'direct_path_418/results.json')['source_input_sha256'].items():assert sha(p)==h,p
 procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'close_low_seat_direct_421.py' not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_direct_421.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_direct_421.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 priorpath=D/'prior_recheck_421.json';priorpath.write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_415.json');mf.update(read(D/'supplemental_preservation_415.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['direct_preparation_416.log','direct_preparation_audit_417.log','direct_path_418.log','direct_path_audit_419.log','progress_report_420.log','direct_path_launch_418.json','direct_path_exit_418.json','direct_saved_constraint_groups_421.json','direct_postprocessing_420.json','progress_report_420_local_review.json','status_before_direct_421.json','handoff_before_direct_421.md','prior_recheck_421.json']:
  p=D/name;mf[str(p)]=sha(p)
 for name in ['low_seat_direct_constraints_416.py','prepare_low_seat_direct_416.py','audit_low_seat_direct_preparation_417.py','freeze_low_seat_direct_launch_418.py','build_low_seat_direct_path_418.py','audit_low_seat_direct_path_419.py','audit_low_seat_direct_domain_419.py','report_low_seat_direct_420.py','index_low_seat_direct_421.py','close_low_seat_direct_421.py']:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 target=D/'completed_probes_preservation_421.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_421.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_direct_path_418_complete_repair_unresolved',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_421.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'direct_path_418'),latest_offline_audit=str(D/'direct_path_audit_419/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Independent419 arithmetic complete. Candidate gates and signed values are retained in418/419; no physical admission. No workers remain.',latest_offline_gate_scope='Direct nonlinear command-hard wholepath problem from406, fixed existing budgets; original physical criteria remain. No callback selection or prior-solve extension.',latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=1,latest_round_offline_nonlinear_trial_evaluations=None,latest_round_offline_accepted_steps=None,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_progress_report=str(D/'progress_report_420/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='416–420 complete once; candidate outcome retained, physical admission false',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=6,original_manifest_entries=entries,supplemental_readable_entries=0,local_review=review,workers=workers,main_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_421.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
