"""Seal new472/473/474 evidence; repair/PPO remain unfinished and active."""
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
 assert not (D/'closure_record_475.json').exists();checks=[]
 for name in ['prior_hashes.json','free_gap_evidence_preservation_471.json','free_gap_supplemental_471.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['free_gap_path_472','free_gap_path_audit_473','progress_report_v2_474','free_gap_report_input_recovery_474']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'free_gap_report_supplemental_475.json').items():assert sha(p)==h,p
 for p,h in read(D/'free_gap_path_launch_472.json')['source_input_sha256'].items():assert sha(p)==h,p
 ex=read(D/'free_gap_postprocessing_474.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0 and ex['original_report_actual_exit']==143
 main=read(D/'free_gap_path_472/results.json');audit=read(D/'free_gap_path_audit_473/independent_audit.json');scope=read(D/'free_gap_saved_scope_475.json');review=read(D/'progress_report_474_local_review.json')
 assert audit['passed'] and review['all_hashes_verified'] and review['visual_review'];assert main['accepted_steps']==0 and scope['actual_scipy_linprog_calls']==6
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_free_gap_475.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_free_gap_475.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'free_gap_evidence_preservation_471.json');mf.update(read(D/'free_gap_supplemental_471.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 names=['low_seat_free_gap_execution_472.py','build_low_seat_free_gap_path_472.py','freeze_low_seat_free_gap_path_472.py','audit_low_seat_free_gap_steps_473.py','audit_low_seat_free_gap_path_473.py','index_low_seat_free_gap_475.py','report_low_seat_free_gap_474.py','report_low_seat_free_gap_474_unexecuted_draft.py','report_low_seat_free_gap_v2_474.py','record_low_seat_free_gap_475.py']
 for name in names:mf[str(R/'scripts'/name)]=sha(R/'scripts'/name)
 for pattern in ['free_gap*472.json','free_gap*472.log','free_gap*473.log','free_gap*474.json','free_gap*474.log','free_gap*475.json','free_gap*475.log','progress_report_474_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file():mf[str(p)]=sha(p)
 mf.pop(str(D/'free_gap_preservation_475.log'),None)
 for name in ['handoff_update_475.txt','status_before_free_gap_475.json','handoff_before_free_gap_475.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_475.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_475.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_free_gap472_rejected_normal_invariants_pass_phase_and_correction_boxes_fail',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_475.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_475.json'),latest_closure=str(D/'closure_record_475.json'),latest_attempt=str(D/'free_gap_path_472'),latest_complete_construction=str(D/'free_gap_path_472'),latest_independent_audit=str(D/'free_gap_path_audit_473/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_v2_474/readable.html'),latest_report_local_review=str(D/'progress_report_474_local_review.json'),latest_round_constructed_model_records=1467,latest_round_new_complete_trial_paths=6,latest_round_new_initial_complete_lifts=1,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=3280,latest_round_inherited_geometry_records=1262,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=6,latest_round_offline_assembled_subproblems=12,latest_round_offline_empty_correction_prechecks=6,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=13,latest_round_offline_nonlinear_trial_evaluations=6,latest_round_offline_accepted_steps=0,new_solver_exists=True,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=True,new_candidate_exists=False,latest_new_path_scope='New initial15-equation lift exported because zero optimization trials accepted.205statesnew/1262exact441. Sixnewtrialsnormal/gap/rootpass,phasefail. SixactualLPs plus sixprecheckedemptydependent-f correctionboxes. No actualfilter/physics/PPO.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,accepted_new_path_steps=0,new_initial_lift=True,new_initial_state_records=205,inherited441_records=1262,actual_scipy_lp_calls=6,assembled_subproblems=12,nonlinear_trials=6,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_475.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['main_result','independent_audit','saved_scope','local_review']},indent=2),flush=True)
if __name__=='__main__':main()
