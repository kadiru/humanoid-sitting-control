"""Seal completed598/599/600 saved evidence; actual repair/PPO/video task remains unfinished."""
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
 assert not (D/'closure_record_601.json').exists();checks=[];oldnames=['prior_hashes.json','balance_encoding_evidence_preservation_597.json','balance_encoding_supplemental_597.json']
 for name in oldnames:
  old=read(D/name)
  for p,h in old.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(old),passed=True))
 folders=['balance_block_preparation_594','balance_block_audit_595','balance_encoding_preparation_596','balance_encoding_audit_597','balance_path_598','balance_path_audit_599','progress_report_600']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'balance_path_launch_598.json')['source_input_sha256'].items():assert sha(p)==h,p
 for name in ['balance_path_compile_exit_598.json','balance_path_freeze_exit_598.json','balance_path_exit_598.json','balance_audit_exit_599.json','balance_report_exit_600.json','balance_index_exit_601.json','balance_postprocessing_compile_exit_600.json','balance_preservation_compile_exit_601.json']:assert read(D/name)['actual_exit']==0,name
 exits=read(D/'balance_postprocessing_600.json');assert exits['main_actual_exit']==exits['independent_audit_actual_exit']==exits['report_actual_exit']==0
 m=read(D/'balance_path_598/results.json');audit=read(D/'balance_path_audit_599/independent_audit.json');scope=read(D/'balance_saved_scope_601.json');review=read(D/'progress_report_600_local_review.json');stage=scope['stages'][0]
 assert audit['passed'] and review['all_hashes_verified'] and review['visual_review']
 assert scope['worker_actual_exits']==[0] and scope['worker_wall_cap_hits']==0 and scope['actual_IPM_lp_calls']==m['actual_lp_calls']==1
 for k in ['returned_native_vectors','complete_certified_witnesses','coordinate_reconstruction_attempts','backsubstitution_attempts','backend_returned_primal_vectors','completed_rootxyf_outputs','completed_leg_outputs','completed_auxiliary_outputs','exact_rational_point_checks','finite_maps','actual_trials','accepted']:assert scope[k]==0,k
 assert scope['point_arithmetic_progress']==[{}]
 assert m['inherited_native_systems']==1 and m['new_native_systems']==m['new_quotient_proof_rows']==m['new_projection_pairs']==0
 assert not m['final_actual_state_changed'] and audit['all_geometry_and_work_inherited']
 assert all(scope[k] for k in ['actual_X_exact472','actual_Y_exact472','actual_J_exact472','auxiliary_features_exact490']) and not scope['auxiliary_Y_exact_actual_Y']
 assert scope['IPM_iterations']==6 and scope['crossover_iterations']==0 and not scope['scientific_execution_recovery'] and m['initial_inverse_inherited']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 notes=(D/'handoff_update_601.txt').read_text();s=read(D/'status.json')
 for k,v in s.items():
  if k.startswith('latest_round_'):assert isinstance(v,(bool,int,float)) or k=='latest_round_new_root_fixture_scope',(k,type(v).__name__)
 for p,name in [(D/'status.json','status_before_balance_601.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_balance_601.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders[4:]:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*balance*598*.py','*balance*599*.py','*balance*600*.py','*balance*601*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['balance_*598*.json','balance_*598*.log','balance_*599*.json','balance_*599*.log','balance_*600*.json','balance_*600*.log','balance_*601*.json','balance_*601*.log','progress_report_600_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='balance_preservation_601.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_601.txt','status_before_balance_601.json','handoff_before_balance_601.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_601.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for k in s:
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
 scope_text='ONE NEW594/596 condensed moment/gap INNER primary IPM,45sTOTAL MAINentry beforeimports/globalchecks,actual'+repr(stage['elapsed_s'])+'s;6IPM and0crossover iterations reported,NO returnedprimal/internalvector. NumericA265967x10541/6102946nnz,EXACTE1367x10541/1nnz,EXPLICIT10541finiteprojectedOUTERbounds. Complete596numericREALinput sufficientINNERsubsetof594zeroFACEinsidebox,possiblyempty/proper,NOTequivalence/finite/tolerance/primal/performanceevidence. AllNINEsystemchecks/EXACT594rootxy-f then586legs then578aux atUNROUNDEDintermediates/68250once-rounding frozenUNEXECUTED. Prior5941365exact3x3+5862730exact6x6inverses inherited,zero newinverses. No newpointchecks/graphlift/analyticdualpair/KKT/interior/map/model/root/AD/geometry/finitework/physics. Backendfactors/solves unmeasured,notzero. No scientific/model/solver/audit/report execution recovery598-600;all priorfailures including593 postsealread-onlyauthoring preserved.601 localreadonlyreview incorrectly expected7 CSVunitlabels(actual8;7 refersclosureunits) afterALL75 filehashes passed; remainingmetadata only completed, nohashloop/science replay. One staleAXscroll failed thencoordinate scroll succeeded.'
 s.update(utc=now,updated_utc=now,phase='offline_balance598_complete_no_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_601.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_601.json'),latest_closure=str(D/'closure_record_601.json'),latest_attempt=str(D/'balance_path_598'),latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest NEW actualmodel states remain484.598 returned no primal witness ornewactualstate;6IPM/0crossover are computationalwork,notfeasibility orperformancegain.',latest_independent_audit=str(D/'balance_path_audit_599/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_600/findings.html'),latest_report_local_review=str(D/'progress_report_600_local_review.json'),latest_progress_report=str(D/'progress_report_600/findings.html'),latest_nominal_research_attempt=str(D/'balance_path_598/results.json'),
 latest_round_offline_lp_calls=1,latest_round_offline_assembled_subproblems=1,latest_round_whole_problem_assemblies=0,latest_round_inherited_whole_problem_assemblies=1,latest_round_new_full_size_lp_calls=1,latest_round_new_full_size_IPM_iterations=6,latest_round_new_crossover_iterations=0,latest_round_original_direction_reconstructions=0,latest_round_full_linear_witnesses=0,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_inherited_native_systems=1,latest_round_new_native_systems=0,latest_round_new_root_fixture_scope=scope_text,
 new_solver_exists=True,new_solver_scope='598 complete594/596 condensed-inner primary andconservativeFULLoriginal interior/changedstate construction with599 return/actualstate auditor. ONEcoldIPM timedout,no witness,no restart. No602 formulation/algorithm/budget/candidate exists.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],
 latest_audit_status='599 conditional independent audit COMPLETE,arithmeticPASSphysicalFALSE. Initial524525/517547559571573579581587589595597 proofs inheritedafterhashidentity. ExactremainingE/completeINNERinput/finiteouterbounds/source/proof/options/MAINentryclock checked. Exact594A/E pointcheck,EXACTunroundedroot-then-leg-then-aux graphlift,finalroundedexact586/578/allNINEKKT/newinput/interior/newactualstate branches frozenUNEXECUTED. Emptyflags notwitness evidence. All1467sealed473geometry/work inherited.',latest_audit=str(D/'balance_path_audit_599/independent_audit.json'),latest_audit_passed=True,latest_candidate_status='Physical273 rejected;actualmodel484 originalrates/targetsFAIL;598 no reduced/native/internalvector despite reported6IPM/0crossover.',
 main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,main_result=m,independent_audit=audit,saved_scope=scope,local_review=review,new_closure=True,prior_closure593_unchanged=True,scientific_execution_recovery=False,metadata_execution_recovery=False,prior565_audit_staged_recovery_preserved=True,prior563_metadata_staged_recovery_preserved=True,prior593_postseal_authoring_failures_preserved=True,new_readonly_local_review_failure_preserved=True,remote_metadata_writer_recovery=False,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_IPM_lp_calls=1,IPM_iterations=6,crossover_iterations=0,backend_factor_counts=None,backend_linear_solve_counts=None,coordinate_reconstruction_attempts=0,completed_rootxyf_outputs=0,completed_leg_outputs=0,completed_auxiliary_outputs=0,exact_rational_point_checks=0,full_linear_witnesses=0,inherited_native_systems=1,new_native_systems=0,new_quotient_proof_rows=0,initial_assembly_repeated=False,inherited_initial_assemblies=1,new_complete_assemblies=0,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_601.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_update_601.txt',manifest,D/'closure_record_601.json',D/'balance_path_598/results.json',D/'balance_path_audit_599/independent_audit.json',D/'progress_report_600/findings.html',D/'progress_report_600_local_review.json']
 (D/'final_handoff_manifest_601.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2));print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_IPM_lp_calls','IPM_iterations','crossover_iterations','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
