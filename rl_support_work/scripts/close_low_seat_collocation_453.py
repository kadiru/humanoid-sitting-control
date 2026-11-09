"""Seal completed448–452 evidence; user repair/PPO task remains unfinished."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_453.json').exists();checks=[]
 for name in ['prior_hashes.json','clock_transcription_evidence_preservation_447.json','clock_transcription_supplemental_447.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['collocation_solver_preparation_448','collocation_solver_audit_449','collocation_path_450','collocation_path_audit_451','collocation_index_recovery_453','progress_report_452']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 post=read(D/'collocation_postprocessing_452.json');assert post['main_actual_exit']==post['independent_audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_452_local_review.json');assert review['visual_review_completed']
 workers=[]
 for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
  a=l.split()
  if len(a)>2 and a[1].endswith('IsaacLab/.venv/bin/python') and a[2].startswith('scripts/') and Path(a[2]).name!=Path(__file__).name:workers.append(l)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_collocation_453.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_collocation_453.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'clock_transcription_evidence_preservation_447.json');mf.update(read(D/'clock_transcription_supplemental_447.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 sources=['low_seat_collocation_solver_448.py','prepare_low_seat_collocation_solver_448.py','audit_low_seat_collocation_solver_449.py','freeze_low_seat_collocation_450.py','build_low_seat_collocation_path_450.py','audit_low_seat_collocation_steps_451.py','audit_low_seat_collocation_path_451.py','index_low_seat_collocation_453.py','index_low_seat_collocation_v2_453.py','report_low_seat_collocation_452.py','close_low_seat_collocation_453.py']
 for n in sources:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 names=['collocation_solver_preparation_448.log','collocation_solver_audit_449.log','collocation_preparation_exit_448.json','collocation_preparation_audit_exit_449.json','collocation_freeze_450.log','collocation_path_launch_450.json','collocation_path_450.log','collocation_path_exit_450.json','collocation_audit_launch_451.json','collocation_path_audit_451.log','collocation_index_453.log','collocation_index_v2_453.log','collocation_saved_localization_453.json','collocation_gate_scope_453.json','collocation_postprocessing_before_report_452.json','collocation_postprocessing_452.json','collocation_report_452.log','collocation_local_hashes_453.json','progress_report_452_local_review.json','report_low_seat_collocation_452_unexecuted_draft.py','status_before_collocation_453.json','handoff_before_collocation_453.md','handoff_update_453.txt']
 for n in names:mf[str(D/n)]=sha(D/n)
 manifest=D/'completed_probes_preservation_453.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_453.txt').read_text();status=read(D/'status.json');result=read(D/'collocation_path_450/results.json');audit=read(D/'collocation_path_audit_451/independent_audit.json')
 status.update(utc=now,updated_utc=now,phase='offline_complete_clock_path_450_rejected_rates_targets',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_453.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_453.json'),latest_preparation=str(D/'collocation_solver_preparation_448'),latest_preparation_audit=str(D/'collocation_solver_audit_449/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='Saved-array wrapper preparation and audit448/449 complete; bounded450 and full451 audit complete.',latest_constructed_candidate=str(D/'collocation_path_450/results.json'),latest_candidate_status='Rejected: original rates and ideal targets fail; sampled geometry/monotone phases/hard static constraints pass.',latest_audit=str(D/'collocation_path_audit_451/independent_audit.json'),latest_audit_status='451 complete arithmetic pass; candidate physical admission false',latest_audit_passed=True,latest_report=str(D/'progress_report_452/findings.html'),latest_report_local_review=str(D/'progress_report_452_local_review.json'),latest_round_constructed_model_records=1467,latest_round_saved_primitive_replays=0,latest_round_new_derivative_fixture_columns=0,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=21840,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=14,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=6,latest_round_offline_nonlinear_trial_evaluations=14,latest_round_offline_accepted_steps=4,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits={'448':'ef98bc/session79037/exit0','449':'98b9be/immediate/exit0','450':'e6897b/session28514/exit0','451':'4e06b0/session81143/exit0','452':'a8b042/session60996/exit0','453_index_original':'5d58d7/exit1','453_index_v2':'b8d34e/exit0'},local_review=review,solver_summary=result,independent_audit_summary={k:v for k,v in audit.items() if k!='step_audit'},original_failed_index_preserved=True,no_scientific_recovery=True,trajectory_lp_calls=14,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,video307_unchanged=True,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_453.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['local_review','solver_summary','independent_audit_summary']},indent=2),flush=True)
if __name__=='__main__':main()
