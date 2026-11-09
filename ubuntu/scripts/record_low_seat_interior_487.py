"""Seal completed484/485/486 evidence; physical repair and PPO remain unfinished."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_487.json').exists();checks=[]
 for name in ['prior_hashes.json','interior_evidence_preservation_483.json','interior_supplemental_483.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['interior_path_484','interior_path_audit_485','progress_report_486']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'interior_path_launch_484.json')['source_input_sha256'].items():assert sha(p)==h,p
 ex=read(D/'interior_postprocessing_486.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 main=read(D/'interior_path_484/results.json');audit=read(D/'interior_path_audit_485/independent_audit.json');scope=read(D/'interior_saved_scope_487.json');review=read(D/'progress_report_486_local_review.json')
 assert audit['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert main['actual_lp_calls']==scope['actual_scipy_linprog_calls'] and main['accepted_steps']==scope['accepted_steps']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_interior_487.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_interior_487.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'interior_evidence_preservation_483.json');mf.update(read(D/'interior_supplemental_483.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for name in ['low_seat_interior_execution_484.py','build_low_seat_interior_path_484.py','freeze_low_seat_interior_path_484.py','audit_low_seat_interior_steps_485.py','audit_low_seat_interior_path_485.py','audit_low_seat_interior_result_485.py','report_low_seat_interior_486.py','report_low_seat_interior_486_unexecuted_draft.py','index_low_seat_interior_487.py','scope_low_seat_interior_487.py','record_low_seat_interior_487.py']:
  mf[str(R/'scripts'/name)]=sha(R/'scripts'/name)
 for pattern in ['interior*.json','interior*.log','progress_report_486_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file():mf[str(p)]=sha(p)
 mf.pop(str(D/'interior_preservation_487.log'),None)
 for name in ['handoff_update_487.txt','status_before_interior_487.json','handoff_before_interior_487.md','active_interior_484.json']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_487.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_487.txt').read_text();status=read(D/'status.json');inherited=bool(audit.get('all_geometry_and_work_inherited',False));newwork=int(audit['new_finite_work_columns'])
 status.update(utc=now,updated_utc=now,phase='offline_interior_witness484_complete_repair_gates_unmet',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_487.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_487.json'),latest_closure=str(D/'closure_record_487.json'),latest_attempt=str(D/'interior_path_484'),latest_complete_construction=str(D/'interior_path_484'),latest_independent_audit=str(D/'interior_path_audit_485/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_486/findings.html'),latest_report_local_review=str(D/'progress_report_486_local_review.json'),latest_round_constructed_model_records=1467,latest_round_new_complete_trial_paths=main['nonlinear_trial_paths'],latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_joint_arithmetic_fixtures=0,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=newwork,latest_round_inherited_geometry_records=1467 if inherited else 0,latest_round_inherited_finite_work_records=sum(int(audit.get(k,0)) for k in ['inherited479_exact_records','inherited473_exact_records','inherited441_exact_records']),latest_round_duplicate_new_finite_work_records=int(audit.get('duplicate_new_records',0)),latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=main['actual_lp_calls'],latest_round_offline_assembled_subproblems=main['assembled_subproblems'],latest_round_offline_empty_correction_prechecks=sum(not s['actual_lp_call'] for s in scope['trial_summary']),latest_round_offline_map_failures=main['coordinate_map_failures'],latest_round_offline_certificate_failures=main['certificate_failures'],latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=main['wrapper_counts']['root_updates'],latest_round_offline_nonlinear_trial_evaluations=main['nonlinear_trial_paths'],latest_round_offline_accepted_steps=main['accepted_steps'],new_solver_exists=True,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=bool(main['accepted_steps']),new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,accepted_new_path_steps=main['accepted_steps'],new_initial_lift=False,inherited_initial_records=1467,all_final_geometry_inherited=inherited,new_finite_work_columns=newwork,actual_scipy_lp_calls=main['actual_lp_calls'],assembled_subproblems=main['assembled_subproblems'],finite_maps=main['finite_maps'],certificate_failures=main['certificate_failures'],finite_map_failures=main['coordinate_map_failures'],nonlinear_trials=main['nonlinear_trial_paths'],actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_487.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['main_result','independent_audit','saved_scope','local_review']},indent=2),flush=True)
if __name__=='__main__':main()
