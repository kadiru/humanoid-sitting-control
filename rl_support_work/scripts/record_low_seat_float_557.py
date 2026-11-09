"""Seal completed554/555/556 once; no native witness or actual-path change, physical task unfinished."""
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
 assert not (D/'closure_record_557.json').exists();checks=[]
 oldnames=['prior_hashes.json','float_hull_evidence_preservation_553.json','float_hull_supplemental_553.json']
 for name in oldnames:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['float_hull_preparation_552','float_hull_audit_553','float_path_554','float_path_audit_555','progress_report_556']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'float_path_launch_554.json')['source_input_sha256'].items():assert sha(p)==h,p
 for name in ['float_path_compile_exit_554.json','float_path_freeze_exit_554.json','float_path_exit_554.json','float_audit_exit_555.json','float_report_exit_556.json','float_index_exit_557.json','float_preservation_compile_exit_557.json']:assert read(D/name)['actual_exit']==0,name
 ex=read(D/'float_postprocessing_556.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 m=read(D/'float_path_554/results.json');a=read(D/'float_path_audit_555/independent_audit.json');scope=read(D/'float_saved_scope_557.json');review=read(D/'progress_report_556_local_review.json');assert a['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0 and scope['actual_IPM_lp_calls']==m['actual_lp_calls']==1
 assert scope['returned_native_vectors']==scope['complete_certified_witnesses']==scope['coordinate_reconstruction_attempts']==0 and scope['backsubstitution_attempts']==0
 assert m['inherited_native_systems']==1 and m['new_native_systems']==m['new_quotient_proof_rows']==m['new_projection_pairs']==0
 assert not m['final_actual_state_changed'] and a['all_geometry_and_work_inherited'] and m['finite_maps']==0
 assert scope['actual_X_exact472'] and scope['actual_Y_exact472'] and scope['actual_J_exact472'] and scope['auxiliary_features_exact490'] and not scope['auxiliary_Y_exact_actual_Y']
 assert scope['IPM_iterations']==17 and scope['crossover_iterations']==0 and scope['backend_returned_native_vectors']==0
 assert not scope['metadata_freeze_recovery'] and not scope['scientific_execution_recovery']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_float_557.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_float_557.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders[2:]:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*float*554*.py','*float*555*.py','*float*556*.py','*float*557*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['float_*554*.json','float_*554*.log','float_*555*.json','float_*555*.log','float_*556*.json','float_*556*.log','float_*557*.json','float_*557*.log','progress_report_556_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='float_preservation_557.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_557.txt','status_before_float_557.json','handoff_before_float_557.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_557.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_557.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_float554_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_557.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_557.json'),latest_closure=str(D/'closure_record_557.json'),latest_attempt=str(D/'float_path_554'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;554 returned no native LP witness or new actual state.',latest_independent_audit=str(D/'float_path_audit_555/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_556/findings.html'),latest_report_local_review=str(D/'progress_report_556_local_review.json'),latest_progress_report=str(D/'progress_report_556/findings.html'),latest_nominal_research_attempt=str(D/'float_path_554/results.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=0,latest_round_inherited_whole_problem_assemblies=1,latest_round_new_full_size_lp_calls=1,latest_round_new_full_size_IPM_iterations=17,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_native_systems=1,latest_round_new_native_systems=0,latest_round_new_root_fixture_scope='ONE NEW full-size finite-binary64-bound native IPM primary LP:45s total deadline,17 IPM/0 crossover,zero returned candidate or original dual-lift attempt. No new assembly/quotient/box/float proof/model/root/AD/geometry/finite-work/finite-map/physics. Backend factor and linear-solve counts unmeasured,not zero. No executed science/audit/report recovery in554-556; prior548/553 metadata failures remain preserved.',
 new_solver_exists=True,new_solver_scope='554 complete552 finite-bound native construction with original dual lift and fixed cold installed HiGHS-IPM stopped at45s total; no witness. No558 new formulation or candidate.552/553 exact finite-binary64 equivalence remains valid/read-only; real LP and tolerance feasible sets not equated. No performance gain demonstrated.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],latest_audit_status='555 conditional independent audit COMPLETE: inherited524/525,516/517,546/547 and552/553 finite-domain proofs. Checked full/source/box/new mixed identities and differing bound domains,provenance,backend configuration/status/resources. Original dual-lift/all-four KKT/new-state branches compiled/frozen but unexecuted. Backend internals unavailable/not replayed. All1467sealed473 geometry/work inherited.',
 latest_audit=str(D/'float_path_audit_555/independent_audit.json'),latest_audit_passed=True,latest_candidate_status='Physical273 remains rejected. Actual-model484 fails original rates/targets.554 produced no complete native LP witness or changed actual state.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=m,independent_audit=a,saved_scope=scope,local_review=review,new_closure=True,prior_closure551_unchanged=True,scientific_execution_recovery=False,metadata_freeze_recovery=False,metadata_freeze_original_actual_exit=None,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_IPM_lp_calls=1,actual_scipy_lp_calls=1,IPM_iterations=17,crossover_iterations=0,backend_factor_counts=None,backend_linear_solve_counts=None,coordinate_reconstruction_attempts=0,full_linear_witnesses=0,inherited_native_systems=1,new_native_systems=0,new_quotient_proof_rows=0,initial_assembly_repeated=False,inherited_initial_assemblies=1,new_complete_assemblies=0,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_557.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_update_557.txt',manifest,D/'closure_record_557.json',D/'float_path_554/results.json',D/'float_path_audit_555/independent_audit.json',D/'progress_report_556/findings.html',D/'progress_report_556_local_review.json']
 (D/'final_handoff_manifest_557.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_IPM_lp_calls','IPM_iterations','crossover_iterations','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
