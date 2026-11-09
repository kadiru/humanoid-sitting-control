"""Preserve completed496/497 preparation; last sealed closure495 unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_497.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_495.json','supplemental_preservation_495.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'residual_preparation_launch_496.json')['source_input_sha256'].items():assert sha(p)==h,p
 for n in ['residual_compile_exit_496.json','residual_freeze_exit_496.json','residual_preparation_exit_496.json','residual_audit_exit_497.json']:assert read(D/n)['actual_exit']==0
 prep=read(D/'residual_coordinate_preparation_496/preparation.json');audit=read(D/'residual_coordinate_audit_497/independent_audit.json');review=read(D/'residual_local_verification_497.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_residual_497.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_residual_497.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_495.json');mf.update(read(D/'supplemental_preservation_495.json'))
 for name in ['residual_coordinate_preparation_496','residual_coordinate_audit_497']:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['low_seat_residual_coordinates_496.py','prepare_low_seat_residual_coordinates_496.py','audit_low_seat_residual_coordinates_497.py',Path(__file__).name]:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 for pattern in ['residual*.json','residual*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='residual_preservation_497.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_497.txt','status_before_residual_497.json','handoff_before_residual_497.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'residual_evidence_preservation_497.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_497.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_residual_coordinate496_preparation_audited_no_new_solver',latest_handoff=str(D/'handoff_prepared_497.txt'),latest_preparation=str(D/'residual_coordinate_preparation_496'),latest_preparation_audit=str(D/'residual_coordinate_audit_497/independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_497.json'),last_sealed_closure=str(D/'closure_record_495.json'),latest_closure=str(D/'closure_record_495.json'),latest_attempt=str(D/'lifted_path_492'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='497 saved-matrix residual-coordinate preparation audit complete; arithmetic only/no new actual path. Actual-state conditional493 remains complete with sealed473 inheritance.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
  latest_round_constructed_model_records=0,latest_round_new_complete_trial_paths=0,latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_finite_work_columns=0,latest_round_new_root_fixture_states=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_joint_arithmetic_fixtures=0,latest_round_new_certificate_arithmetic_cases=0,latest_round_repeated_scalar_arithmetic_regression_checks=0,latest_round_new_assembly_fixtures=2,latest_round_new_local_saved_matrix_residual_solves=1365,latest_round_new_algebraic_right_hand_sides=68250,latest_round_new_affine_vector_checks=2,latest_round_new_directed_inverse_certificates=1365,latest_round_inherited_geometry_records=0,latest_round_inherited_finite_work_records=0,latest_round_duplicate_new_finite_work_records=0,
  latest_round_offline_lp_calls=0,latest_round_offline_assembled_subproblems=0,latest_round_offline_empty_correction_prechecks=0,latest_round_offline_map_failures=0,latest_round_offline_certificate_failures=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,
  latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,
  new_solver_exists=False,new_solver_scope='No498 run wrapper or candidate.492 COMPLETE/read-only;496/497 saved-matrix preparation only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope='No changed actualX/Y/J or auxiliary state; only affine substitution and independent directed-rounding nonsingularity proof.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,repair_complete=False,task_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_495.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,preparation=prep,independent_audit=audit,local_verification=review,actual_exits={n:read(D/n) for n in ['residual_compile_exit_496.json','residual_freeze_exit_496.json','residual_preparation_exit_496.json','residual_audit_exit_497.json']},new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physical_admission=False,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_497.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','independent_audit','local_verification']},indent=2),flush=True)
if __name__=='__main__':main()
