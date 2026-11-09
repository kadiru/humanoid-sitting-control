"""Seal completed410–414 evidence, preserving 409 evidence; no scientific or physical recovery this round. No task completion."""
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
 assert not (D/'closure_record_415.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_409.json','supplemental_preservation_409.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['monotone_preparation_410','monotone_preparation_audit_411','monotone_path_412','monotone_path_audit_413','monotone_report_draft_revision_414','progress_report_414'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 assert read(D/'monotone_path_exit_412.json')['observed_exit_code']==0
 assert read(D/'monotone_path_audit_413/independent_audit.json')['passed']
 post=read(D/'monotone_postprocessing_414.json');assert post['audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_414_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 for p,h in read(D/'monotone_path_412/results.json')['source_input_sha256'].items():assert sha(p)==h,p
 procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'close_low_seat_monotone_415.py' not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_monotone_415.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_monotone_415.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 priorpath=D/'prior_recheck_415.json';priorpath.write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_409.json');mf.update(read(D/'supplemental_preservation_409.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['monotone_preparation_410.log','monotone_preparation_audit_411.log','monotone_path_412.log','monotone_path_audit_413.log','progress_report_414.log','monotone_path_launch_412.json','monotone_path_exit_412.json','monotone_saved_rejections_415.json','monotone_postprocessing_414.json','progress_report_414_local_review.json','status_before_monotone_415.json','handoff_before_monotone_415.md','prior_recheck_415.json']:
  p=D/name;mf[str(p)]=sha(p)
 for name in ['low_seat_monotone_phase_410.py','prepare_low_seat_monotone_410.py','audit_low_seat_monotone_preparation_411.py','build_low_seat_monotone_path_412.py','audit_low_seat_monotone_path_413.py','audit_low_seat_monotone_steps_413.py','report_low_seat_monotone_414.py','close_low_seat_monotone_415.py']:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 target=D/'completed_probes_preservation_415.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_415.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_monotone_path_412_complete_repair_unresolved',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_415.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'monotone_path_412'),latest_offline_audit=str(D/'monotone_path_audit_413/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Independent413 arithmetic complete; sampled geometry/nonnegative support/hover/hard constraints pass, command margin/rates/headroom and NLP/window/clock gates fail. No workers remain.',latest_offline_gate_scope='412 rejected; five nonlinear-feasible half steps retain hard geometry-construction rows, but command margin/rates/headroom fail. No trust-floor extension.',latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=8,latest_round_offline_nonlinear_trial_evaluations=13,latest_round_offline_accepted_steps=5,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_progress_report=str(D/'progress_report_414/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='410–414 complete once;412 rejected',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=6,original_manifest_entries=entries,supplemental_readable_entries=0,local_review=review,workers=workers,main_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_415.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
