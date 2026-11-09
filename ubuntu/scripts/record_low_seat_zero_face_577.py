"""Seal completed574/575/576 saved evidence; actual repair/PPO/video task remains unfinished."""
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
 assert not (D/'closure_record_577.json').exists();checks=[]
 oldnames=['prior_hashes.json','zero_face_evidence_preservation_573.json','zero_face_supplemental_573.json']
 for name in oldnames:
  old=read(D/name)
  for p,h in old.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(old),passed=True))
 folders=['zero_face_preparation_572','zero_face_audit_573','zero_face_path_574','zero_face_path_audit_575','progress_report_576']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'zero_face_path_launch_574.json')['source_input_sha256'].items():assert sha(p)==h,p
 for name in ['zero_face_path_compile_exit_574.json','zero_face_path_freeze_exit_574.json','zero_face_path_exit_574.json','zero_face_audit_exit_575.json','zero_face_report_exit_576.json','zero_face_index_exit_577.json','zero_face_preservation_compile_exit_577.json']:assert read(D/name)['actual_exit']==0,name
 exits=read(D/'zero_face_postprocessing_576.json');assert exits['main_actual_exit']==exits['independent_audit_actual_exit']==exits['report_actual_exit']==0
 main=read(D/'zero_face_path_574/results.json');audit=read(D/'zero_face_path_audit_575/independent_audit.json');scope=read(D/'zero_face_saved_scope_577.json');review=read(D/'progress_report_576_local_review.json')
 assert audit['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0 and scope['actual_IPM_lp_calls']==main['actual_lp_calls']==1
 assert scope['returned_native_vectors']==scope['complete_certified_witnesses']==scope['coordinate_reconstruction_attempts']==scope['backsubstitution_attempts']==0
 assert scope['backend_returned_original_primal_vectors']==0
 assert main['inherited_native_systems']==1 and main['new_native_systems']==main['new_quotient_proof_rows']==main['new_projection_pairs']==0
 assert not main['final_actual_state_changed'] and audit['all_geometry_and_work_inherited'] and main['finite_maps']==0
 assert all(scope[k] for k in ['actual_X_exact472','actual_Y_exact472','actual_J_exact472','auxiliary_features_exact490']) and not scope['auxiliary_Y_exact_actual_Y']
 assert scope['IPM_iterations']==24 and scope['crossover_iterations']==0 and not scope['scientific_execution_recovery']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 notes=(D/'handoff_update_577.txt').read_text();s=read(D/'status.json')
 # Validate mutable-schema handling BEFORE writing backups or the preservation manifest.
 for k,v in s.items():
  if k.startswith('latest_round_'):assert isinstance(v,(bool,int,float)) or k=='latest_round_new_root_fixture_scope',(k,type(v).__name__)
 for p,name in [(D/'status.json','status_before_zero_face_577.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_zero_face_577.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders[2:]:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*zero_face*574*.py','*zero_face*575*.py','*zero_face*576*.py','*zero_face*577*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['zero_face_*574*.json','zero_face_*574*.log','zero_face_*575*.json','zero_face_*575*.log','zero_face_*576*.json','zero_face_*576*.log','zero_face_*577*.json','zero_face_*577*.log','progress_report_576_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='zero_face_preservation_577.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_577.txt','status_before_zero_face_577.json','handoff_before_zero_face_577.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_577.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for k in s:
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
 s.update(utc=now,updated_utc=now,phase='offline_zero_face574_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_577.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_577.json'),latest_closure=str(D/'closure_record_577.json'),latest_attempt=str(D/'zero_face_path_574'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;574 returned no original native primal witness or new actual state.',latest_independent_audit=str(D/'zero_face_path_audit_575/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_576/findings.html'),latest_report_local_review=str(D/'progress_report_576_local_review.json'),latest_progress_report=str(D/'progress_report_576/findings.html'),latest_nominal_research_attempt=str(D/'zero_face_path_574/results.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=0,latest_round_inherited_whole_problem_assemblies=1,latest_round_new_full_size_lp_calls=1,latest_round_new_full_size_IPM_iterations=24,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_native_systems=1,latest_round_new_native_systems=0,
 latest_round_new_root_fixture_scope='ONE NEW full-size primary572 objective-zero-face IPM LP:45sTOTAL configured fromMAINentry beforeimports,actualcharged50.13256082194857s,24IPM/0crossover,no returnedprimal/KKT. ALL78791nativevariables,scalar78790;A265967/414193nnz,E69617/749167nnz AFFINE. Face mayempty/proper;no wholeLP equivalence/optimum-zero claim.573 analyticdual inheritedonly,notnewlyconstructed; candidatepair checksUNEXECUTED. Backend46.11364382202737s of40.98136658105068 allowance,overshoot5.1323sreported. No new assembly/face/analyticdualproof/model/root/AD/geometry/finitework/map/physics. Backend factors/solves unmeasured,notzero. No scientific/model/solver/audit/report execution recovery574-576; old failures preserved.',
 new_solver_exists=True,new_solver_scope='574 complete primary572zero-face and actual full-original interior construction with independent input/return/actualauditor. One cold IPM returned no witness at45sTOTAL deadline(overshoot reported),no restart. No578 formulation/solver/budget/candidate exists.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],
 latest_audit_status='575 conditional independent audit COMPLETE: inherited524525/517547559571573 proofs; FULL/source/box/coupled/face identity/domain/AFFINEE/scalar/coldbackend/chargedMAIN-entry resources checked. Return/allFIVEKKT/newinput/interior/newactualstate branches frozenUNEXECUTED. Backendinternals unavailable/notreplayed. All1467sealed473 geometry/work inherited.',latest_audit=str(D/'zero_face_path_audit_575/independent_audit.json'),latest_audit_passed=True,latest_candidate_status='Physical273 rejected. Actualmodel484 rates/targets fail.574 no original primal witness or changed actualstate.',
 main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,main_result=main,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,prior_closure569_unchanged=True,scientific_execution_recovery=False,metadata_execution_recovery=False,prior565_audit_staged_recovery_preserved=True,prior563_metadata_staged_recovery_preserved=True,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_IPM_lp_calls=1,IPM_iterations=24,crossover_iterations=0,backend_factor_counts=None,backend_linear_solve_counts=None,coordinate_reconstruction_attempts=0,full_linear_witnesses=0,inherited_native_systems=1,new_native_systems=0,new_quotient_proof_rows=0,initial_assembly_repeated=False,inherited_initial_assemblies=1,new_complete_assemblies=0,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_577.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_update_577.txt',manifest,D/'closure_record_577.json',D/'zero_face_path_574/results.json',D/'zero_face_path_audit_575/independent_audit.json',D/'progress_report_576/findings.html',D/'progress_report_576_local_review.json']
 (D/'final_handoff_manifest_577.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_IPM_lp_calls','IPM_iterations','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
