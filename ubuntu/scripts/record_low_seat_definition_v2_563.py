"""Seal completed560/561/562 once; no native witness or actual-path change, physical task unfinished."""
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
 assert not (D/'closure_record_563.json').exists() and not (D/'completed_probes_preservation_v2_563.json').exists()
 assert read(D/'definition_preservation_original_exit_563.json')['actual_exit']==1
 assert read(D/'definition_preservation_v2_compile_exit_563.json')['actual_exit']==0
 stage=read(D/'definition_preservation_stage_563.json')
 for p,h in stage['source_input_sha256'].items():assert sha(p)==h,p
 mf=read(D/'completed_probes_preservation_563.json')
 for p,h in mf.items():assert sha(p)==h,p
 checks=read(D/'definition_path_launch_560.json')['prior_checks']
 ex=read(D/'definition_postprocessing_562.json');m=read(D/'definition_path_560/results.json');a=read(D/'definition_path_audit_561/independent_audit.json');scope=read(D/'definition_saved_scope_563.json');review=read(D/'progress_report_562_local_review.json')
 assert a['passed'] and review['all_hashes_verified'] and review['visual_review'] and scope['actual_IPM_lp_calls']==1 and scope['IPM_iterations']==16 and scope['backend_returned_extended_vectors']==0
 assert scope['complete_certified_witnesses']==scope['finite_maps']==scope['actual_trials']==scope['accepted']==0 and not m['final_actual_state_changed']
 assert scope['actual_X_exact472'] and scope['actual_Y_exact472'] and scope['actual_J_exact472'] and scope['auxiliary_features_exact490'] and a['all_geometry_and_work_inherited']
 assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 for p in [D/'completed_probes_preservation_563.json',D/'definition_preservation_563.log',D/'definition_preservation_original_exit_563.json',D/'definition_preservation_v2_compile_exit_563.json',D/'definition_preservation_stage_563.json',D/'handoff_update_v2_563.txt',R/'scripts/record_low_seat_definition_v2_563.py']:mf[str(p)]=sha(p)
 manifest=D/'completed_probes_preservation_v2_563.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_v2_563.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_definition560_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_v2_563.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_563.json'),latest_closure=str(D/'closure_record_563.json'),latest_attempt=str(D/'definition_path_560'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;560 returned no extended or original LP witness or new actual state.',latest_independent_audit=str(D/'definition_path_audit_561/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_562/findings.html'),latest_report_local_review=str(D/'progress_report_562_local_review.json'),latest_progress_report=str(D/'progress_report_562/findings.html'),latest_nominal_research_attempt=str(D/'definition_path_560/results.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=0,latest_round_inherited_whole_problem_assemblies=1,latest_round_new_full_size_lp_calls=1,latest_round_new_full_size_IPM_iterations=16,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_native_systems=1,latest_round_new_native_systems=0,latest_round_new_root_fixture_scope='ONE NEW full-size exact558 definition-extended IPM primary LP:45sTOTAL deadline,16IPM/0crossover,no extended/native vector or original-projection/KKT attempt.136120variables,originalscalar78790,NONZERO E58695x136120. All original78791 preserved; original546 domain,not552finitebounds. No new assembly/quotient/box/definition proof/model/root/AD/geometry/finite-work/map/physics. Backend factor/linear-solve counts unmeasured,notzero. No executed failure/recovery560-562; prior548/553metadata and554local authoring failures preserved.',
 new_solver_exists=True,new_solver_scope='560 complete558 exact sparse extended construction with original prefix projection and ALLFOUR KKT gates,one fixed cold SciPy HiGHS-IPM stopped45sTOTAL no witness. No564 formulation/solver/candidate.558/559 exact real projection/objective proof remainsvalid/read-only;no universalbinary64extension,tolerance-set or performance claim.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],latest_audit_status='561 conditional independent audit COMPLETE: inherited524/525,517/547/559 proofs. Full/source/box/extended identity,NONZERO E,scalar78790,originaldomain/cost/IDs/provenance/backend configuration/status/resources independentlychecked. Original prefix projection and allFOUR KKT/new-state branches compiledfrozenUNEXECUTED. Backendinternals unavailable/notreplayed. All1467sealed473 geometry/work inherited.',
 latest_audit=str(D/'definition_path_audit_561/independent_audit.json'),latest_audit_passed=True,latest_candidate_status='Physical273 remains rejected. Actual-model484 fails original rates/targets.560 produced no complete extended/original LP witness or changed actual state.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,main_result=m,independent_audit=a,saved_scope=scope,local_review=review,new_closure=True,prior_closure557_unchanged=True,scientific_execution_recovery=False,metadata_freeze_recovery=False,metadata_freeze_original_actual_exit=None,closure_metadata_recovery=True,original_closure_writer_actual_exit=1,original_preservation_manifest_sha256=sha(D/"completed_probes_preservation_563.json"),accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_IPM_lp_calls=1,actual_scipy_lp_calls=1,IPM_iterations=16,crossover_iterations=0,backend_factor_counts=None,backend_linear_solve_counts=None,coordinate_reconstruction_attempts=0,full_linear_witnesses=0,inherited_native_systems=1,new_native_systems=0,new_quotient_proof_rows=0,initial_assembly_repeated=False,inherited_initial_assemblies=1,new_complete_assemblies=0,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_563.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_update_v2_563.txt',manifest,D/'closure_record_563.json',D/'definition_path_560/results.json',D/'definition_path_audit_561/independent_audit.json',D/'progress_report_562/findings.html',D/'progress_report_562_local_review.json']
 (D/'final_handoff_manifest_563.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_IPM_lp_calls','IPM_iterations','crossover_iterations','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
