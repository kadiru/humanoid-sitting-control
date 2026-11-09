"""Seal completed566/567/568 saved evidence; actual repair/PPO/video task remains unfinished."""
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
 assert not (D/'closure_record_569.json').exists();checks=[]
 oldnames=['prior_hashes.json','dual_balance_evidence_preservation_565.json','dual_balance_supplemental_565.json']
 for name in oldnames:
  old=read(D/name)
  for p,h in old.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(old),passed=True))
 folders=['dual_balance_preparation_564','dual_balance_audit_565','dual_path_566','dual_path_audit_567','progress_report_568']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'dual_path_launch_566.json')['source_input_sha256'].items():assert sha(p)==h,p
 for name in ['dual_path_compile_exit_566.json','dual_path_freeze_exit_566.json','dual_path_exit_566.json','dual_audit_exit_567.json','dual_report_exit_568.json','dual_index_exit_569.json','dual_preservation_compile_exit_569.json']:assert read(D/name)['actual_exit']==0,name
 exits=read(D/'dual_postprocessing_568.json');assert exits['main_actual_exit']==exits['independent_audit_actual_exit']==exits['report_actual_exit']==0
 main=read(D/'dual_path_566/results.json');audit=read(D/'dual_path_audit_567/independent_audit.json');scope=read(D/'dual_saved_scope_569.json');review=read(D/'progress_report_568_local_review.json')
 assert audit['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0 and scope['actual_IPM_lp_calls']==main['actual_lp_calls']==1
 assert scope['returned_native_vectors']==scope['complete_certified_witnesses']==scope['coordinate_reconstruction_attempts']==scope['backsubstitution_attempts']==0
 assert scope['backend_returned_multiplier_vectors']==scope['backend_returned_equality_marginals']==0
 assert main['inherited_native_systems']==1 and main['new_native_systems']==main['new_quotient_proof_rows']==main['new_projection_pairs']==0
 assert not main['final_actual_state_changed'] and audit['all_geometry_and_work_inherited'] and main['finite_maps']==0
 assert all(scope[k] for k in ['actual_X_exact472','actual_Y_exact472','actual_J_exact472','auxiliary_features_exact490']) and not scope['auxiliary_Y_exact_actual_Y']
 assert scope['IPM_iterations']==17 and scope['crossover_iterations']==0 and not scope['scientific_execution_recovery']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 notes=(D/'handoff_update_569.txt').read_text();s=read(D/'status.json')
 # Validate mutable-schema handling BEFORE writing backups or the preservation manifest.
 for k,v in s.items():
  if k.startswith('latest_round_'):assert isinstance(v,(bool,int,float)) or k=='latest_round_new_root_fixture_scope',(k,type(v).__name__)
 for p,name in [(D/'status.json','status_before_dual_569.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_dual_569.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders[2:]:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*dual*566*.py','*dual*567*.py','*dual*568*.py','*dual*569*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['dual_*566*.json','dual_*566*.log','dual_*567*.json','dual_*567*.log','dual_*568*.json','dual_*568*.log','dual_*569*.json','dual_*569*.log','progress_report_568_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='dual_preservation_569.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_569.txt','status_before_dual_569.json','handoff_before_dual_569.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_569.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for k in s:
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
 s.update(utc=now,updated_utc=now,phase='offline_dual566_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_569.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_569.json'),latest_closure=str(D/'closure_record_569.json'),latest_attempt=str(D/'dual_path_566'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;566 returned no multiplier/equality-marginal/native witness or new actual state.',latest_independent_audit=str(D/'dual_path_audit_567/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_568/findings.html'),latest_report_local_review=str(D/'progress_report_568_local_review.json'),latest_progress_report=str(D/'progress_report_568/findings.html'),latest_nominal_research_attempt=str(D/'dual_path_566/results.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=0,latest_round_inherited_whole_problem_assemblies=1,latest_round_new_full_size_lp_calls=1,latest_round_new_full_size_IPM_iterations=17,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_native_systems=1,latest_round_new_native_systems=0,
 latest_round_new_root_fixture_scope='ONE NEW full-size exact564 dual-balance IPM primary LP:45sTOTAL deadline,17IPM/0crossover,no multiplier/equality-marginal/native vector or return/KKT.402489 nonnegative optimizer variables,78791 original-coordinate equality slots,2049087 nonzero E coefficients,0 inequality rows. Original546 realdomain; no552/558/522. No new assembly/quotient/box/transpose/identity/model/root/AD/geometry/finite-work/map/physics. Backend factors/solves unmeasured,notzero. No executed failure/recovery566-568; prior565 stagedaudit and563 stagedmetadata failures remain preserved.',
 new_solver_exists=True,new_solver_scope='566 complete exact564 native dualization with originalw BITWISE scipy equalitymarginals,notresult.x,allFOUR D/546/516/490 KKT. One fixed cold IPM stopped45sTOTAL no witness. No570 formulation/solver/budget/candidate exists.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],
 latest_audit_status='567 conditional independent audit COMPLETE: inherited524/525,517/547/565 proofs; full/source/box/dual identities,domains/equality-slotlayout/provenance/cold backend/status/resources checked. Equality-marginalreturn/dualembedding/allFOUR KKT/new-state branches compiledfrozenUNEXECUTED. Backendinternals unavailable/notreplayed. All1467sealed473 geometry/work inherited.',latest_audit=str(D/'dual_path_audit_567/independent_audit.json'),latest_audit_passed=True,latest_candidate_status='Physical273 rejected. Actualmodel484 rates/targets fail.566 no complete multiplier/equalitymarginal/native witness or changed actualstate.',
 main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,prior_closure563_unchanged=True,scientific_execution_recovery=False,metadata_execution_recovery=False,prior565_audit_staged_recovery_preserved=True,prior563_metadata_staged_recovery_preserved=True,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_IPM_lp_calls=1,IPM_iterations=17,crossover_iterations=0,backend_factor_counts=None,backend_linear_solve_counts=None,coordinate_reconstruction_attempts=0,full_linear_witnesses=0,inherited_native_systems=1,new_native_systems=0,new_quotient_proof_rows=0,initial_assembly_repeated=False,inherited_initial_assemblies=1,new_complete_assemblies=0,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_569.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_update_569.txt',manifest,D/'closure_record_569.json',D/'dual_path_566/results.json',D/'dual_path_audit_567/independent_audit.json',D/'progress_report_568/findings.html',D/'progress_report_568_local_review.json']
 (D/'final_handoff_manifest_569.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_IPM_lp_calls','IPM_iterations','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
