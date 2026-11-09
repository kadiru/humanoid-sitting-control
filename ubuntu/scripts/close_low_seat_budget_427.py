"""Seal completed422–426 evidence, preserving421 evidence. Guarded by actual exit, audit and local review records; no task completion."""
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
 assert not (D/'closure_record_427.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_421.json','supplemental_preservation_421.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['budget_preparation_422','budget_preparation_audit_423','budget_path_424','budget_path_audit_425','budget_report_draft_revision_426','progress_report_426'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 assert read(D/'budget_path_exit_424.json')['observed_exit_code']==0
 assert read(D/'budget_path_audit_425/independent_audit.json')['passed']
 post=read(D/'budget_postprocessing_426.json');assert post['audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_426_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 for p,h in read(D/'budget_path_424/results.json')['source_input_sha256'].items():assert sha(p)==h,p
 procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'close_low_seat_budget_427.py' not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_budget_427.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_budget_427.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 priorpath=D/'prior_recheck_427.json';priorpath.write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_421.json');mf.update(read(D/'supplemental_preservation_421.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['active_run_424.log','active_record_attempt_424.json','handoff_running_424.txt','budget_path_launch_424.log','budget_preparation_422.log','budget_preparation_audit_423.log','budget_path_424.log','budget_path_audit_425.log','progress_report_426.log','budget_path_launch_424.json','budget_path_exit_424.json','budget_saved_constraint_groups_427.json','budget_postprocessing_426.json','progress_report_426_local_review.json','status_before_budget_427.json','handoff_before_budget_427.md','prior_recheck_427.json']:
  p=D/name;mf[str(p)]=sha(p)
 for name in ['record_low_seat_active_budget_424.py','low_seat_budget_domain_422.py','prepare_low_seat_budget_422.py','audit_low_seat_budget_preparation_423.py','freeze_low_seat_budget_launch_424.py','build_low_seat_budget_path_424.py','audit_low_seat_budget_path_425.py','audit_low_seat_budget_domain_425.py','report_low_seat_budget_426.py','index_low_seat_budget_427.py','close_low_seat_budget_427.py']:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 target=D/'completed_probes_preservation_427.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_427.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_budget_path_424_tree_complete_integer_alignment_unresolved',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_427.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'budget_path_424'),latest_offline_audit=str(D/'budget_path_audit_425/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Independent425 arithmetic complete. Five fractional numerical relaxations, no exact integer model point; six branches unvisited at declared cap. No physical admission or workers.',latest_offline_gate_scope='Bounded five-node integer-event nonlinear tree from406. Continuous research box unchanged; original/reserved criteria retained; fractional relaxations not executable. Cap is not infeasibility.',latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=5,latest_round_offline_nonlinear_trial_evaluations=None,latest_round_offline_accepted_steps=None,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_report=str(D/'progress_report_426/findings.html'),latest_progress_report=str(D/'progress_report_426/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='422–426 complete once; candidate outcome retained, physical admission false',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=6,original_manifest_entries=entries,supplemental_readable_entries=0,local_review=review,workers=workers,main_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_427.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
