"""Seal completed548/549/550 once; no native witness or actual-path change, physical task unfinished."""
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
 assert not (D/'closure_record_551.json').exists();checks=[]
 oldnames=['prior_hashes.json','box_support_evidence_preservation_547.json','box_support_supplemental_547.json']
 for name in oldnames:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['box_support_preparation_546','box_support_audit_547','box_path_548','box_path_audit_549','progress_report_550']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'box_path_launch_548.json')['source_input_sha256'].items():assert sha(p)==h,p
 for name in ['box_path_compile_exit_548.json','box_path_freeze_exit_548.json','box_path_exit_548.json','box_audit_exit_549.json','box_report_exit_550.json','box_index_exit_551.json','box_preservation_compile_exit_551.json']:assert read(D/name)['actual_exit']==0,name
 ex=read(D/'box_postprocessing_550.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 m=read(D/'box_path_548/results.json');a=read(D/'box_path_audit_549/independent_audit.json');scope=read(D/'box_saved_scope_551.json');review=read(D/'progress_report_550_local_review.json');assert a['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0 and scope['actual_IPM_lp_calls']==m['actual_lp_calls']==1
 assert scope['returned_native_vectors']==scope['complete_certified_witnesses']==scope['coordinate_reconstruction_attempts']==0 and scope['backsubstitution_attempts']==0
 assert m['inherited_native_systems']==1 and m['new_native_systems']==m['new_quotient_proof_rows']==m['new_projection_pairs']==0
 assert not m['final_actual_state_changed'] and a['all_geometry_and_work_inherited'] and m['finite_maps']==0
 assert scope['actual_X_exact472'] and scope['actual_Y_exact472'] and scope['actual_J_exact472'] and scope['auxiliary_features_exact490'] and not scope['auxiliary_Y_exact_actual_Y']
 assert scope['IPM_iterations']==17 and scope['crossover_iterations']==0 and scope['backend_returned_native_vectors']==0
 assert scope['metadata_freeze_recovery'] and not scope['scientific_execution_recovery']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_box_551.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_box_551.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders[2:]:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*box*548*.py','*box*549*.py','*box*550*.py','*box*551*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['box_*548*.json','box_*548*.log','box_*549*.json','box_*549*.log','box_*550*.json','box_*550*.log','box_*551*.json','box_*551*.log','progress_report_550_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='box_preservation_551.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_551.txt','status_before_box_551.json','handoff_before_box_551.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_551.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_551.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_box548_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_551.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_551.json'),latest_closure=str(D/'closure_record_551.json'),latest_attempt=str(D/'box_path_548'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;548 returned no native LP witness or new actual state.',latest_independent_audit=str(D/'box_path_audit_549/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_550/findings.html'),latest_report_local_review=str(D/'progress_report_550_local_review.json'),latest_progress_report=str(D/'progress_report_550/findings.html'),latest_nominal_research_attempt=str(D/'box_path_548/results.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=0,latest_round_inherited_whole_problem_assemblies=1,latest_round_new_full_size_lp_calls=1,latest_round_new_full_size_IPM_iterations=17,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_native_systems=1,latest_round_new_native_systems=0,latest_round_new_root_fixture_scope='ONE NEW full-size exact-box-reduced native IPM primary LP:45s total deadline,17 IPM/0 crossover,zero returned candidate. No new source assembly/quotient/box proof/model/root/AD/geometry/finite-work/finite-map/physics. Backend internal factor and linear-solve counts unmeasured,not zero. Original metadata-freeze exit1 preserved; scientific run/audit/report each once.',
 new_solver_exists=True,new_solver_scope='548 complete exact546 box-reduced native construction with fixed cold installed HiGHS-IPM stopped at45s total; no witness. No552 new formulation or candidate. Structural reduction546/547 remains valid/read-only and supplies no demonstrated performance gain.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],latest_audit_status='549 conditional independent audit COMPLETE: inherited initial524 assembly/525 proof,516/517 original implications and546/547 exact box support. Checks full/source/reduced/domain copies,proof provenance,backend configuration/status/resources. No backend internal iterates/factors available or replayed. All1467sealed473 geometry/work inherited. New-state/full-witness/finite-map branches compiled but unexecuted.',
 latest_audit=str(D/'box_path_audit_549/independent_audit.json'),latest_audit_passed=True,latest_candidate_status='Physical273 remains rejected. Actual-model484 fails original rates/targets.548 produced no complete native LP witness or changed actual state.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=m,independent_audit=a,saved_scope=scope,local_review=review,new_closure=True,prior_closure545_unchanged=True,scientific_execution_recovery=False,metadata_freeze_recovery=True,metadata_freeze_original_actual_exit=1,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_IPM_lp_calls=1,actual_scipy_lp_calls=1,IPM_iterations=17,crossover_iterations=0,backend_factor_counts=None,backend_linear_solve_counts=None,coordinate_reconstruction_attempts=0,full_linear_witnesses=0,inherited_native_systems=1,new_native_systems=0,new_quotient_proof_rows=0,initial_assembly_repeated=False,inherited_initial_assemblies=1,new_complete_assemblies=0,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_551.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_update_551.txt',manifest,D/'closure_record_551.json',D/'box_path_548/results.json',D/'box_path_audit_549/independent_audit.json',D/'progress_report_550/findings.html',D/'progress_report_550_local_review.json']
 (D/'final_handoff_manifest_551.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_IPM_lp_calls','IPM_iterations','crossover_iterations','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
