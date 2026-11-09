"""Preserve completed502/503 preparation; last sealed closure501 unchanged."""
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
 assert not (D/'preparation_handoff_record_503.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_501.json','supplemental_preservation_501.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'sparse_gap_preparation_launch_502.json')['source_input_sha256'].items():assert sha(p)==h,p
 for n in ['sparse_gap_compile_exit_502.json','sparse_gap_freeze_exit_502.json','sparse_gap_preparation_exit_502.json','sparse_gap_audit_exit_503.json']:assert read(D/n)['actual_exit']==0
 prep=read(D/'sparse_gap_residual_preparation_502/preparation.json');audit=read(D/'sparse_gap_residual_audit_503/independent_audit.json');review=read(D/'sparse_gap_local_verification_503.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_sparse_gap_503.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_sparse_gap_503.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_501.json');mf.update(read(D/'supplemental_preservation_501.json'))
 for name in ['sparse_gap_residual_preparation_502','sparse_gap_residual_audit_503']:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['low_seat_sparse_gap_residual_502.py','prepare_low_seat_sparse_gap_residual_502.py','audit_low_seat_sparse_gap_residual_503.py',Path(__file__).name]:
  p=R/'scripts'/name;mf[str(p)]=sha(p)
 for pattern in ['sparse_gap*.json','sparse_gap*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='sparse_gap_preservation_503.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_503.txt','status_before_sparse_gap_503.json','handoff_before_sparse_gap_503.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'sparse_gap_evidence_preservation_503.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_503.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_sparse_gap502_preparation_audited_no_new_solver',latest_handoff=str(D/'handoff_prepared_503.txt'),latest_preparation=str(D/'sparse_gap_residual_preparation_502'),latest_preparation_audit=str(D/'sparse_gap_residual_audit_503/independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_503.json'),last_sealed_closure=str(D/'closure_record_501.json'),latest_closure=str(D/'closure_record_501.json'),latest_attempt=str(D/'residual_path_498'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='503 integer gap-incidence and whole-row preparation audit complete; arithmetic only/no new actual path. Actual-state conditional499 remains complete with sealed473 inheritance.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
  latest_round_constructed_model_records=0,latest_round_new_complete_trial_paths=0,latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_finite_work_columns=0,latest_round_new_root_fixture_states=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_joint_arithmetic_fixtures=0,latest_round_new_certificate_arithmetic_cases=0,latest_round_repeated_scalar_arithmetic_regression_checks=0,latest_round_new_assembly_fixtures=2,latest_round_new_local_saved_matrix_residual_solves=0,latest_round_new_algebraic_right_hand_sides=0,latest_round_new_affine_vector_checks=0,latest_round_new_directed_inverse_certificates=0,latest_round_new_integer_normal_gap_incidence_coordinates=205,latest_round_inherited_geometry_records=0,latest_round_inherited_finite_work_records=0,latest_round_duplicate_new_finite_work_records=0,
  latest_round_offline_lp_calls=0,latest_round_offline_assembled_subproblems=0,latest_round_offline_empty_correction_prechecks=0,latest_round_offline_map_failures=0,latest_round_offline_certificate_failures=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,
  latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,
  new_solver_exists=False,new_solver_scope='No504 run wrapper or candidate.498 COMPLETE/read-only;502/503 saved-matrix normal-gap incidence preparation only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='502/503 continuous interior-normal-gap coordinate incidence prepared/audited; all rows retained,23.7Mprimarynnz, no LP/maps/new actualpath or speed claim.',latest_preparation_handoff=str(D/'preparation_handoff_record_503.json'),latest_new_path_scope='No changed actualX/Y/J, canonicalauxiliary or dropstate; new integer gap incidence, storedS composition and all-row assembly audited. PriorQ proof inherited, no new proof/solve.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,repair_complete=False,task_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_501.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,preparation=prep,independent_audit=audit,local_verification=review,actual_exits={n:read(D/n) for n in ['sparse_gap_compile_exit_502.json','sparse_gap_freeze_exit_502.json','sparse_gap_preparation_exit_502.json','sparse_gap_audit_exit_503.json']},new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physical_admission=False,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_503.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','independent_audit','local_verification']},indent=2),flush=True)
if __name__=='__main__':main()
