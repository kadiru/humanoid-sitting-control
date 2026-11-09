"""Seal new476/477/478/479/480 evidence; repair/PPO remain unfinished and active."""
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
 assert not (D/'closure_record_481.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_475.json','supplemental_preservation_475.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['joint_ray_preparation_476','joint_ray_audit_477','joint_ray_path_478','joint_ray_path_audit_479','progress_report_480']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'joint_ray_report_supplemental_481.json').items():assert sha(p)==h,p
 for p,h in read(D/'joint_ray_path_launch_478.json')['source_input_sha256'].items():assert sha(p)==h,p
 ex=read(D/'joint_ray_postprocessing_480.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 main=read(D/'joint_ray_path_478/results.json');audit=read(D/'joint_ray_path_audit_479/independent_audit.json');scope=read(D/'joint_ray_saved_scope_481.json');review=read(D/'progress_report_480_local_review.json')
 assert audit['passed'] and review['all_hashes_verified'] and review['visual_review'];assert main['accepted_steps']==1 and scope['actual_scipy_linprog_calls']==13
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_joint_ray_481.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_joint_ray_481.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'completed_probes_preservation_475.json');mf.update(read(D/'supplemental_preservation_475.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 names=['low_seat_joint_ray_476.py','prepare_low_seat_joint_ray_476.py','audit_low_seat_joint_ray_477.py','low_seat_joint_ray_execution_478.py','build_low_seat_joint_ray_path_478.py','freeze_low_seat_joint_ray_path_478.py','audit_low_seat_joint_ray_steps_479.py','audit_low_seat_joint_ray_path_479.py','audit_low_seat_joint_ray_result_479.py','index_low_seat_joint_ray_481.py','report_low_seat_joint_ray_480.py','report_low_seat_joint_ray_480_unexecuted_draft.py','record_low_seat_joint_ray_481.py']
 for name in names:mf[str(R/'scripts'/name)]=sha(R/'scripts'/name)
 for pattern in ['joint_ray*.json','joint_ray*.log','progress_report_480_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file():mf[str(p)]=sha(p)
 mf.pop(str(D/'joint_ray_preservation_481.log'),None)
 for name in ['handoff_update_481.txt','status_before_joint_ray_481.json','handoff_before_joint_ray_481.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_481.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_481.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_joint_ray478_one_hard_feasible_step_original_rates_targets_fail',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_481.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_481.json'),latest_closure=str(D/'closure_record_481.json'),latest_attempt=str(D/'joint_ray_path_478'),latest_complete_construction=str(D/'joint_ray_path_478'),latest_independent_audit=str(D/'joint_ray_path_audit_479/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_480/readable.html'),latest_report_local_review=str(D/'progress_report_480_local_review.json'),latest_round_constructed_model_records=1467,latest_round_new_complete_trial_paths=6,latest_round_new_initial_complete_lifts=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=10540,latest_round_new_joint_arithmetic_fixtures=9,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=21840,latest_round_inherited_geometry_records=0,latest_round_inherited_finite_work_records=48,latest_round_duplicate_new_finite_work_records=54,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=13,latest_round_offline_assembled_subproblems=13,latest_round_offline_empty_correction_prechecks=0,latest_round_offline_map_failures=7,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=13,latest_round_offline_nonlinear_trial_evaluations=6,latest_round_offline_accepted_steps=1,new_solver_exists=True,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=True,new_candidate_exists=False,latest_new_path_scope='One accepted newsharedray step atlinear10; finalhard/phase/normal/gap/fullboxpass butoriginalrates/targetsfail. Sealed472INITIALinherited,zeroinitialrootrepeats.13actualLPs/6newrootedpaths/7exactmapfailures. All1467geometryrechecked;21840newworkcolumns,48sealed473inheritance,54newduplicates. Noactualfilter/physics/PPO',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,accepted_new_path_steps=1,new_initial_lift=False,inherited_initial_records=1467,new_final_changed_records=1419,inherited473_finite_work_records=48,duplicate_new_records=54,actual_scipy_lp_calls=13,assembled_subproblems=13,finite_map_attempts=13,finite_map_failures=7,nonlinear_trials=6,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_481.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['main_result','independent_audit','saved_scope','local_review']},indent=2),flush=True)
if __name__=='__main__':main()
