"""Preserve completed434–436 evidence; this is not repair or task completion."""
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
 assert not (D/'closure_record_437.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_433.json','supplemental_preservation_433.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['clock_lift_input_recovery_434','clock_lift_v2_434','clock_lift_geometry_audit_recovery_435','clock_lift_audit_v3_435','clock_lift_report_draft_revision_436','progress_report_436'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 post=read(D/'clock_lift_postprocessing_436.json');assert post['assessment_actual_exit']==post['audit_corrected_actual_exit']==post['report_actual_exit']==0;assert post['assessment_original_actual_exit']==post['audit_original_actual_exit']==1
 review=read(D/'progress_report_436_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 assert read(D/'clock_lift_audit_v3_435/independent_audit.json')['passed'];assert not read(D/'clock_lift_v2_434/results.json')['nonlinear_lift_compatible']
 for p,h in read(D/'clock_lift_launch_v2_434.json')['source_input_sha256'].items():assert sha(p)==h,p
 workers=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if 'IsaacLab/.venv/bin/python scripts/' in s and Path(__file__).name not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_clock_lift_437.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_clock_lift_437.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 (D/'prior_recheck_437.json').write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_433.json');mf.update(read(D/'supplemental_preservation_433.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 names=['clock_lift_launch_434.json','clock_lift_launch_v2_434.json','clock_lift_434.log','clock_lift_v2_434.log','clock_lift_exit_434.json','clock_lift_audit_v2_435.log','clock_lift_audit_v3_435.log','clock_lift_postprocessing_436.json','progress_report_436.log','progress_report_436_local_review.json','clock_lift_localization_437.json','clock_corner_identity_437.json','status_before_clock_lift_437.json','handoff_before_clock_lift_437.md','prior_recheck_437.json']
 for name in names:mf[str(D/name)]=sha(D/name)
 for name in ['assess_low_seat_clock_lift_434.py','assess_low_seat_clock_lift_v2_434.py','audit_low_seat_clock_lift_435.py','audit_low_seat_clock_lift_v2_435.py','audit_low_seat_clock_lift_v3_435.py','report_low_seat_clock_lift_436.py','close_low_seat_clock_lift_437.py']:mf[str(R/'scripts'/name)]=sha(R/'scripts'/name)
 target=D/'completed_probes_preservation_437.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_437.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_clock_lift_434_rejected_source430_prior_gates_pass_buffer_unmet',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_437.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_clock_lift_assessment=str(D/'clock_lift_v2_434'),latest_clock_lift_compatible=False,latest_clock_lift_sampled_geometry_passed=False,latest_offline_audit=str(D/'clock_lift_audit_v3_435/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Corrected435v3 staged arithmetic audit complete. Original435v2 geometry failure retained; its completed finite-work aggregates inherited without repeating columns, detailed array unavailable. New434 interpolant rejected; source430 prior gates still pass and extra buffer remains unmet.',latest_offline_gate_scope='Source430 remains the latest optimized path: integer/prior model gates pass, added2^-23 buffer fails. NEW434 is a separate non-optimized between-knot compatibility assessment, not a replacement optimized path.',latest_offline_screen_scope='434 explicit linear/geodesic pose and componentwise force-slot interpolant fails single-foot transition support and equilibrium/command consistency. Finite samples only, no actual forward filter, current quiet, runtime, or physics.',latest_offline_geometry_passed=True,latest_offline_static_commands_passed=True,latest_offline_nlp_passed=True,latest_offline_time_gate_passed=True,latest_offline_integer_event_gate_passed=True,latest_offline_extra_buffer_numerical_passed=False,latest_offline_extra_buffer_strict_passed=False,latest_physical_admission_passed=False,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=0,latest_round_offline_milp_nodes=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_objective_evaluations=0,latest_round_offline_jacobian_evaluations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,latest_round_interpolated_model_records=1467,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_report=str(D/'progress_report_436/findings.html'),latest_progress_report=str(D/'progress_report_436/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='434v2 compatibility assessment rejected; staged435v3 arithmetic audit and436report complete. Original434input/435geometry failures retained. No optimizer/filter/physics.',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=len(folders),original_manifest_entries=entries,local_review=review,workers=workers,assessment_actual_exit=0,original_assessment_actual_exit=1,original_audit_actual_exit=1,corrected_audit_actual_exit=0,report_actual_exit=0,finite_work_columns_completed_once_in_original_audit=23472,finite_work_columns_repeated=0,detailed_finite_work_array_available=False,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_437.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
