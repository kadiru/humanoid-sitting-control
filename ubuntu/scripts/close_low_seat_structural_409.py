"""Seal completed404–408 evidence, preserving 403 evidence and preserved405 audit failure. No task completion."""
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
 assert not (D/'closure_record_409.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_403.json','supplemental_preservation_403.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['structural_preparation_404','structural_preparation_audit_v2_405','structural_audit_scalar_recovery_405','structural_path_406','structural_path_audit_407','structural_report_draft_revision_408','progress_report_408'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 assert read(D/'structural_path_exit_406.json')['observed_exit_code']==0
 assert read(D/'structural_path_audit_407/independent_audit.json')['passed']
 post=read(D/'structural_postprocessing_408.json');assert post['audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_408_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 for p,h in read(D/'structural_path_406/results.json')['source_input_sha256'].items():assert sha(p)==h,p
 procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'close_low_seat_structural_409.py' not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_structural_409.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_structural_409.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 priorpath=D/'prior_recheck_409.json';priorpath.write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_403.json');mf.update(read(D/'supplemental_preservation_403.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['structural_preparation_404.log','structural_preparation_audit_405.log','structural_preparation_audit_v2_405.log','structural_path_406.log','structural_path_audit_407.log','progress_report_408.log','structural_path_launch_406.json','structural_path_exit_406.json','structural_hard_rejection_409.json','structural_hard_original_409.json','structural_postprocessing_408.json','progress_report_408_local_review.json','status_before_structural_409.json','handoff_before_structural_409.md','prior_recheck_409.json']:
  p=D/name;mf[str(p)]=sha(p)
 for name in ['low_seat_structural_coordinates_404.py','prepare_low_seat_structural_404.py','audit_low_seat_structural_preparation_405.py','audit_low_seat_structural_preparation_v2_405.py','build_low_seat_structural_path_406.py','audit_low_seat_structural_path_407.py','audit_low_seat_structural_steps_407.py','report_low_seat_structural_408.py','close_low_seat_structural_409.py']:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 target=D/'completed_probes_preservation_409.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_409.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_structural_path_406_complete_repair_unresolved',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_409.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'structural_path_406'),latest_offline_audit=str(D/'structural_path_audit_407/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Independent407 arithmetic complete; sampled geometry/static commands/support/hover pass, original rates/headroom and NLP/window/clock gates fail. No workers remain.',latest_offline_gate_scope='406 rejected; hard nonlinear properties retained but original rates/headroom fail. Signed original and inward phase deficits distinguished in saved409 localization.',latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=10,latest_round_offline_nonlinear_trial_evaluations=10,latest_round_offline_accepted_steps=1,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_progress_report=str(D/'progress_report_408/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='404–408 complete once;406 rejected',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=7,original_manifest_entries=entries,supplemental_readable_entries=0,local_review=review,workers=workers,main_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_409.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
