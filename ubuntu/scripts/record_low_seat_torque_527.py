"""Seal completed524/525/526 after one staged pre-LP recovery; physical task remains unfinished."""
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
 assert not (D/'closure_record_527.json').exists();checks=[]
 oldnames=['prior_hashes.json','torque_projection_evidence_preservation_523.json','torque_projection_supplemental_523.json']
 for name in oldnames:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['torque_projection_preparation_522','torque_projection_audit_523','torque_path_524','torque_path_audit_525','progress_report_526']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'torque_path_launch_524.json')['source_input_sha256'].items():assert sha(p)==h,p
 for name in ['torque_compile_exit_524.json','torque_freeze_exit_524.json','torque_path_exit_524.json','torque_audit_exit_525.json','torque_report_exit_526.json']:
  assert read(D/name)['actual_exit']==0,name
 ex=read(D/'torque_postprocessing_526.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 m=read(D/'torque_path_524/results.json');a=read(D/'torque_path_audit_525/independent_audit.json');scope=read(D/'torque_saved_scope_527.json');review=read(D/'progress_report_526_local_review.json')
 assert a['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0
 assert scope['actual_scipy_linprog_calls']==m['actual_lp_calls']==1 and scope['original_direction_reconstructions']==scope['returned_reduced_vectors']==scope['backsubstitution_attempts']==0
 assert m['inherited_projection_systems']==1 and m['new_projection_systems']==m['new_quotient_proof_rows']==m['new_projection_pairs']==0
 assert not m['final_actual_state_changed'] and a['all_geometry_and_work_inherited'] and m['finite_maps']==0
 assert scope['actual_X_exact472'] and scope['actual_Y_exact472'] and scope['actual_J_exact472'] and not scope['auxiliary_Y_exact_actual_Y']
 assert read(D/'torque_path_initial_exit_524.json')['actual_exit']==1
 assert read(D/'torque_path_staged_launch_524.json')['saved_files']==12
 for p,h in read(D/'torque_path_staged_launch_524.json')['source_and_saved_sha256'].items():assert sha(p)==h,p
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_torque_527.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_torque_527.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders[2:]:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*torque*524*.py','*torque*525*.py','*torque*526*.py','*torque*527*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['torque_*.json','torque_*.log','progress_report_526_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='torque_preservation_527.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_527.txt','status_before_torque_527.json','handoff_before_torque_527.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_527.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_527.txt').read_text();status=read(D/'status.json')
 for k in list(status):
  if k.startswith('latest_round_'):
   if isinstance(status[k],bool):status[k]=False
   elif isinstance(status[k],(int,float)):status[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 status.update(utc=now,updated_utc=now,phase='offline_torque524_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_527.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_527.json'),latest_closure=str(D/'closure_record_527.json'),latest_attempt=str(D/'torque_path_524'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;524 returned no LP witness or new actual state.',latest_independent_audit=str(D/'torque_path_audit_525/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_526/findings.html'),latest_report_local_review=str(D/'progress_report_526_local_review.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=1,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_projection_systems=1,latest_round_new_projection_systems=0,latest_round_new_exact_quotient_proof_rows=0,latest_round_new_projection_pairs=0,latest_round_eliminated_auxiliary_torque_directions=21840,latest_round_new_root_fixture_scope='ZERO new model/root/AD/geometry/finite-work/finite-map or old row grouping/projection/proof/fixture calls. One new bounded torque-projected primary LP timed out without a witness.',
 new_solver_exists=True,new_solver_scope='524 complete torque-projected construction stopped after one primary IPM time limit. Original pre-LP file-lookup failure and saved-stage continuation preserved. No528 algorithm/candidate.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],latest_audit_status='525 conditional audit COMPLETE: one full original problem and inherited torque projection/subset; all1467sealed473 geometry/work inherited, no new actual states.',latest_audit=str(D/'torque_path_audit_525/independent_audit.json'),latest_audit_passed=True,latest_progress_report=str(D/'progress_report_526/findings.html'),latest_nominal_research_attempt=str(D/'torque_path_524/results.json'),latest_preparation=str(D/'torque_projection_preparation_522'),latest_preparation_audit=str(D/'torque_projection_audit_523/independent_audit.json'),latest_preparation_scope='522/523 exact auxiliary torque projection complete/read-only; initial proof inherited524 without repeated pair/example loops.',latest_preparation_handoff=str(D/'handoff_prepared_523.txt'),latest_preparation_preservation=str(D/'torque_projection_evidence_preservation_523.json'),latest_preparation_handoff_record=str(D/'preparation_handoff_record_523.json'),latest_candidate_status='Physical273 remains rejected. Actual-model484 fails original rates/targets.524 produced no complete LP witness or changed actual state.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 assert status['latest_round_new_integer_normal_gap_incidence_coordinates']==0
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=m,independent_audit=a,saved_scope=scope,local_review=review,new_closure=True,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_scipy_lp_calls=1,original_direction_reconstructions=0,full_linear_witnesses=0,inherited_projection_systems=1,new_projection_systems=0,new_quotient_proof_rows=0,prior516_failed_fixture_preserved=True,original524_actual_exit=1,staged_recovery=True,initial_assembly_repeated=False,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_527.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',R/'docs/LOW_SEAT_CONTACT_STATUS.md',D/'handoff_update_527.txt',manifest,D/'closure_record_527.json',D/'torque_path_524/results.json',D/'torque_path_audit_525/independent_audit.json',D/'progress_report_526/findings.html',D/'progress_report_526_local_review.json']
 (D/'final_handoff_manifest_527.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_scipy_lp_calls','full_linear_witnesses','inherited_projection_systems','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
