"""Seal completed428–432 evidence after actual exits and local review. Not task completion."""
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
 assert not (D/'closure_record_433.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_427.json','supplemental_preservation_427.json']:
  mf=read(D/name)
  for p,h in mf.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(mf),passed=True))
 folders=['lattice_preparation_428','lattice_preparation_audit_429','lattice_launch_metadata_recovery_430','lattice_path_430','lattice_path_audit_431','lattice_report_draft_revision_432','progress_report_432'];entries=0
 for name in folders:
  mf=read(D/name/'SHA256.json')
  for p,h in mf.items():assert sha(D/name/p)==h,p
  entries+=len(mf)
 assert read(D/'lattice_path_exit_430.json')['observed_exit_code']==0 and read(D/'lattice_path_audit_431/independent_audit.json')['passed']
 post=read(D/'lattice_postprocessing_432.json');assert post['audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_432_local_review.json');assert review['visual_review_complete'] and review['original_entries']==entries
 result=read(D/'lattice_path_430/results.json')
 for p,h in result['source_input_sha256'].items():assert sha(p)==h,p
 procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and Path(__file__).name not in s and 'ssh ' not in s and 'sh -c' not in s];assert not workers,workers
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for p,name in [(D/'status.json','status_before_lattice_433.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_lattice_433.md')]:assert not (D/name).exists();shutil.copy2(p,D/name)
 (D/'prior_recheck_433.json').write_text(json.dumps(dict(utc=now,checks=checks,passed=True),indent=2))
 mf=read(D/'completed_probes_preservation_427.json');mf.update(read(D/'supplemental_preservation_427.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['lattice_preparation_428.log','lattice_preparation_audit_429.log','lattice_path_launch_430.log','lattice_path_launch_v2_430.log','lattice_path_launch_430.json','lattice_path_430.log','lattice_path_exit_430.json','lattice_path_audit_431.log','progress_report_432.log','lattice_saved_constraint_groups_433.log','lattice_saved_constraint_groups_433.json','lattice_postprocessing_432.json','progress_report_432_local_review.json','status_before_lattice_433.json','handoff_before_lattice_433.md','prior_recheck_433.json']:
  p=D/name;mf[str(p)]=sha(p)
 for name in ['low_seat_lattice_projection_428.py','prepare_low_seat_lattice_428.py','audit_low_seat_lattice_preparation_429.py','freeze_low_seat_lattice_launch_430.py','freeze_low_seat_lattice_launch_v2_430.py','build_low_seat_lattice_path_430.py','export_low_seat_lattice_430.py','audit_low_seat_lattice_path_431.py','audit_low_seat_lattice_domain_431.py','report_low_seat_lattice_432.py','index_low_seat_lattice_433.py','close_low_seat_lattice_433.py']:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 target=D/'completed_probes_preservation_433.json';target.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 notes=(D/'handoff_update_433.txt').read_text();s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='offline_lattice_path_430_integer_and_prior_gates_pass_extra_buffer_unmet',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_433.txt'),latest_preservation=str(target),completed_evidence_manifest=str(target),completed_evidence_count=len(mf),latest_offline_path=str(D/'lattice_path_430'),latest_offline_paths=[str(D/'lattice_path_430')],latest_offline_audit=str(D/'lattice_path_audit_431/independent_audit.json'),latest_offline_audit_passed=True,latest_audit_status='Independent431 arithmetic complete. Exact integer events and all prior model/inward/clock/window gates pass; NEW2^-23 buffer gate fails. No float32/runtime/physics admission or workers.',latest_offline_gate_scope='One integer-exact all-hard closest-path construction, derived428 allocation. All894 continuous coordinates in unchanged416 box. Original/prior inward requirements pass; added numerical buffer fails, not original-rate failure.',latest_offline_screen_scope='Sampled offline cooked50 geometry/static forces/real-arithmetic nominal clock only. No intermediate/live-shape, actual float32 forward filter, realized support or CURRENT quiet certificate.',latest_offline_geometry_passed=True,latest_offline_static_commands_passed=True,latest_offline_nlp_passed=True,latest_offline_time_gate_passed=True,latest_offline_integer_event_gate_passed=True,latest_offline_extra_buffer_numerical_passed=False,latest_offline_extra_buffer_strict_passed=False,latest_physical_admission_passed=False,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_milp_calls=0,latest_round_offline_milp_nodes=0,latest_round_offline_nlp_calls=1,latest_round_offline_nonlinear_iterations=30,latest_round_offline_objective_evaluations=73,latest_round_offline_jacobian_evaluations=31,latest_round_offline_nonlinear_trial_evaluations=None,latest_round_offline_accepted_steps=None,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,latest_report=str(D/'progress_report_432/findings.html'),latest_progress_report=str(D/'progress_report_432/findings.html'),notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 closure=dict(utc=now,completed_work='428–432 complete once; original/integer gates pass and added buffer fails; physical admission false',prior_checks=checks,protected_files=len(mf),manifest=str(target),manifest_sha256=sha(target),original_report_manifests=len(folders),original_manifest_entries=entries,supplemental_readable_entries=0,local_review=review,workers=workers,main_actual_exit=0,independent_audit_actual_exit=0,report_actual_exit=0,physics_ticks=0,actual_candidate_command_schedule_ticks=0,policy_optimizer_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,main_ppo_started=False,fresh_holdout_started=False,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_433.json').write_text(json.dumps(closure,indent=2));print(json.dumps(closure,indent=2),flush=True)
if __name__=='__main__':main()
