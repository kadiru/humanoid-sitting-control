"""Seal completed398–402 evidence, preserving both397 supplemental chains. No task completion."""
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
 assert not (D/'closure_record_403.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_397.json','supplemental_preservation_397.json','status_label_supplemental_397.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['window_preparation_398','window_preparation_audit_399','window_path_400','window_path_audit_401','progress_report_402'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 for p,h in read(D/'progress_report_402/readable_SHA256.json').items():assert sha(D/'progress_report_402'/p)==h,p
 assert read(D/'window_path_exit_400.json')['observed_exit_code']==0
 assert read(D/'window_path_audit_401/independent_audit.json')['passed']
 post=read(D/'window_postprocessing_402.json');assert post['audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_402_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 for p,h in read(D/'window_path_400/results.json')['source_input_sha256'].items():assert sha(p)==h,p
 procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'close_low_seat_window_403.py' not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_window_403.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_window_403.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 priorpath=D/'prior_recheck_403.json';priorpath.write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_397.json');mf.update(read(D/'supplemental_preservation_397.json'));mf.update(read(D/'status_label_supplemental_397.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['window_preparation_398.log','window_preparation_audit_399.log','window_path_400.log','window_path_audit_401.log','progress_report_402.log','window_path_launch_400.json','window_path_exit_400.json','window_zero_step_witness_403.json','window_zero_step_witness_403.npy','window_postprocessing_402.json','progress_report_402_local_review.json','status_before_window_403.json','handoff_before_window_403.md','prior_recheck_403.json']:
  p=D/name;mf[str(p)]=sha(p)
 for name in ['low_seat_window_envelope_398.py','prepare_low_seat_window_398.py','audit_low_seat_window_preparation_399.py','build_low_seat_window_path_400.py','audit_low_seat_window_path_401.py','report_low_seat_window_402.py','close_low_seat_window_403.py']:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 target=D/'completed_probes_preservation_403.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_403.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_window_path_400_complete_repair_unresolved',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_403.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'window_path_400'),latest_offline_audit=str(D/'window_path_audit_401/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Independent401 arithmetic complete; strict static/nonlinear/window/clock gates failed. No worker remains.',latest_offline_gate_scope='400 rejected; geometry passed, commands/support/hover/rates/headroom failed. Saved zero-step witness prevents treating final linear solver status as certified infeasibility.',latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=12,latest_round_offline_nonlinear_trial_evaluations=11,latest_round_offline_accepted_steps=3,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_progress_report=str(D/'progress_report_402/readable.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='398–402 complete once;400 rejected',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=5,original_manifest_entries=entries,supplemental_readable_entries=2,local_review=review,workers=workers,main_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_403.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
