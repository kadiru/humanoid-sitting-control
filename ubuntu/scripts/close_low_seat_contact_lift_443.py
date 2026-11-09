"""Seal completed438..442 evidence; repair, PPO and user task remain unfinished."""
from pathlib import Path
import json,hashlib,datetime,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_443.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_437.json','supplemental_preservation_437.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['contact_lift_preparation_438','contact_lift_preparation_audit_439','contact_lift_440','contact_lift_audit_441','progress_report_442'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 post=read(D/'contact_lift_postprocessing_442.json');assert post['construction_actual_exit']==post['independent_audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_442_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 assert read(D/'contact_lift_audit_441/independent_audit.json')['passed'];result=read(D/'contact_lift_440/results.json');assert result['minimum_new_clock_rate_rad']<0 and result['minimum_sampled_descent_increment']<0 and not result['physical_admission']
 for p,h in read(D/'contact_lift_launch_440.json')['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'contact_lift_audit_launch_441.json')['source_sha256'].items():assert sha(p)==h,p
 workers=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if 'IsaacLab/.venv/bin/python scripts/' in s and Path(__file__).name not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_contact_lift_443.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_contact_lift_443.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 (D/'prior_recheck_443.json').write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_437.json');mf.update(read(D/'supplemental_preservation_437.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 names=['contact_lift_preparation_launch_438.json','contact_lift_preparation_438.log','contact_lift_preparation_audit_439.log','contact_lift_launch_440.json','contact_lift_440.log','contact_lift_exit_440.json','contact_lift_audit_launch_441.json','contact_lift_audit_441.log','contact_lift_postprocessing_442.json','progress_report_442.log','progress_report_442_local_review.json','contact_lift_localization_443.json','contact_lift_gate_scope_443.json','status_before_contact_lift_443.json','handoff_before_contact_lift_443.md','prior_recheck_443.json']
 for n in names:mf[str(D/n)]=sha(D/n)
 for n in ['low_seat_contact_lift_438.py','prepare_low_seat_contact_lift_438.py','audit_low_seat_contact_preparation_439.py','build_low_seat_contact_lift_440.py','audit_low_seat_contact_lift_441.py','audit_low_seat_contact_rows_441.py','report_low_seat_contact_lift_442.py','close_low_seat_contact_lift_443.py']:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 target=D/'completed_probes_preservation_443.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_443.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_persistent_contact_lift_440_geometry_pass_rates_phase_fail',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_443.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_constructed_path=str(D/'contact_lift_440'),latest_clock_lift_assessment=str(D/'contact_lift_440'),latest_clock_lift_compatible=False,latest_clock_lift_sampled_geometry_passed=True,latest_offline_audit=str(D/'contact_lift_audit_441/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='441 complete arithmetic audit PASS; new440 rate and between-knot phase gates FAIL. 20848 new finite-work columns,163 cardinal-equivalent inherited431 records,one exact new duplicate. No435work repeated.',latest_offline_gate_scope='NEW440 is a complete non-optimized persistent-contact kinematic construction. All1467 sampled geometries/static limits pass; rates,between-knot phase order and added-buffer gates fail. Source430 optimized path remains unchanged: its prior model/integer gates pass, added buffer fails.',latest_offline_screen_scope='Persistent local corners/moment-derived COM/one-sided left support/constrained FK at1467 records. Declared330 frames drift during approach; not a stationary-foot,no-slip,continuous/live-collision or actual support certificate.',latest_offline_geometry_passed=True,latest_offline_static_commands_passed=True,latest_offline_nlp_passed=False,latest_offline_time_gate_passed=True,latest_offline_integer_event_gate_passed=True,latest_offline_extra_buffer_numerical_passed=False,latest_offline_extra_buffer_strict_passed=False,latest_physical_admission_passed=False,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=0,latest_round_offline_milp_nodes=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_objective_evaluations=0,latest_round_offline_jacobian_evaluations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,latest_round_constructed_model_records=1467,latest_round_actual_forward_filter_ticks=0,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_report=str(D/'progress_report_442/findings.html'),latest_progress_report=str(D/'progress_report_442/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='438preparation/439saved-arrayaudit/440completeconstruction/441new-stateaudit/442report complete. New sampled geometry/static/target gates pass, rates and between-knot phase fail. No optimizer/filter/physics.',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=len(folders),original_manifest_entries=entries,local_review=review,workers=workers,construction_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,new_finite_work_columns=20848,inherited431_cardinal_records=163,duplicate_new_state_records=1,original435_finite_work_repeated=0,optimizer_calls=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,actual_forward_filter_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_443.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
