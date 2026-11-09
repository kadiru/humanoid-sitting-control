"""Seal completed510/511/512/513/514 with actual exits and local review; physical task unfinished."""
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
 assert not (D/'closure_record_515.json').exists();checks=[]
 oldnames=['prior_hashes.json','completed_probes_preservation_509.json','supplemental_preservation_509.json','mixed_metadata_supplemental_509.json']
 for name in oldnames:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['row_separation_preparation_510','row_separation_audit_511','separated_path_512','separated_path_audit_513','progress_report_514']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 for p,h in read(D/'separated_path_launch_512.json')['source_input_sha256'].items():assert sha(p)==h,p
 prep_exits={name:read(D/name) for name in ['row_separation_compile_exit_510.json','row_separation_freeze_exit_510.json','row_separation_preparation_exit_510.json','row_separation_audit_exit_511.json']}
 assert all(x['actual_exit']==0 for x in prep_exits.values())
 ex=read(D/'separated_postprocessing_514.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==ex['report_actual_exit']==0
 m=read(D/'separated_path_512/results.json');a=read(D/'separated_path_audit_513/independent_audit.json');scope=read(D/'separated_saved_scope_515.json');review=read(D/'progress_report_514_local_review.json')
 assert a['passed'] and review['all_hashes_verified'] and review['visual_review'];assert scope['worker_actual_exits']==[0,0] and scope['worker_wall_cap_hits']==0
 assert scope['actual_scipy_linprog_calls']==m['actual_lp_calls']==2 and scope['fully_certified_witnesses']==0 and scope['returned_master_candidates']==1
 assert m['separation_added_rows']==15885 and sum(x['hard_rows'] for x in scope['cut_scope'][0]['groups'])==1808
 assert not m['final_actual_state_changed'] and a['all_geometry_and_work_inherited'] and m['finite_maps']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if args and 'python' in args[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in args[1:]):workers.append(args)
 assert not workers,workers
 for p,name in [(D/'status.json','status_before_separated_515.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_separated_515.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf={}
 for name in oldnames[1:]:mf.update(read(D/name))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*row_separation*510*.py','*row_separation*511*.py','*separated*512*.py','*separated*513*.py','*separated*514*.py','*separated*515*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['row_separation_*.json','row_separation_*.log','separated_*.json','separated_*.log','active_separated_512.json','progress_report_514_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='separated_preservation_515.log':mf[str(p)]=sha(p)
 for name in ['handoff_update_515.txt','status_before_separated_515.json','handoff_before_separated_515.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'completed_probes_preservation_515.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_515.txt').read_text();status=read(D/'status.json')
 # Every prior latest-round scalar here is an accounting count/flag, never an error bound.
 for k in list(status):
  if k.startswith('latest_round_'):
   if isinstance(status[k],bool):status[k]=False
   elif isinstance(status[k],(int,float)):status[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 status.update(utc=now,updated_utc=now,phase='offline_separated512_complete_no_full_witness_actual_unchanged',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_handoff=str(D/'handoff_update_515.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_515.json'),latest_closure=str(D/'closure_record_515.json'),latest_attempt=str(D/'separated_path_512'),
 latest_complete_construction=str(D/'interior_path_484'),latest_complete_construction_scope='Latest newly constructed actual-model states remain484;512 partial linear witness was rejected before mapping.',latest_independent_audit=str(D/'separated_path_audit_513/independent_audit.json'),latest_independent_audit_passed=True,latest_report=str(D/'progress_report_514/findings.html'),latest_report_local_review=str(D/'progress_report_514_local_review.json'),
 latest_round_offline_lp_calls=2,latest_round_offline_assembled_subproblems=2,latest_round_whole_problem_assemblies=1,latest_round_separation_cut_rounds=1,latest_round_separation_added_rows=15885,latest_round_original_direction_reconstructions=1,latest_round_full_linear_witnesses=0,latest_round_returned_partial_master_witnesses=1,latest_round_inherited_geometry_records=1467,latest_round_inherited_finite_work_records=1467,latest_round_new_assembly_fixtures=2,latest_round_offline_assembly_fixtures=2,latest_round_new_separation_decision_cases=10,latest_round_new_root_fixture_scope='ZERO new model/root/AD/geometry/finite-work/finite-map calls. New saved row-separation arithmetic and bounded two-master LP attempt only.',
 new_solver_exists=True,new_solver_scope='512 whole-row separation COMPLETE/read-only: one optimal partial master fails15885 deferred rows; after all cuts second master times out. No516 algorithm/candidate.',prospective_joint_solver_assembly_exists=True,new_finite_lifted_decoder_exists=True,new_executable_whole_path_wrapper_exists=True,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_new_path_scope=review['outcome_scope'],
 latest_audit_status='513 conditional audit COMPLETE: one full original/transformed problem, two subsets and full rejected-partial residual/cut replay; all1467sealed473 geometry/work inherited.',latest_audit=str(D/'separated_path_audit_513/independent_audit.json'),latest_audit_passed=True,latest_progress_report=str(D/'progress_report_514/findings.html'),latest_nominal_research_attempt=str(D/'separated_path_512/results.json'),latest_preparation=str(D/'row_separation_preparation_510'),latest_preparation_audit=str(D/'row_separation_audit_511/independent_audit.json'),latest_preparation_scope='510/511 new saved subset selection and10separation-decision cases COMPLETE; immutable504basis inherited, no inverse/incidence/model/map fixtures repeated.',latest_preparation_handoff=str(D/'closure_record_515.json'),latest_preparation_preservation=str(manifest),latest_preparation_handoff_record=str(D/'closure_record_515.json'),latest_candidate_status='Physical273 remains rejected. Latest actual-model484 fails original rates/targets.512 produced no complete LP witness or changed actual state.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 assert status['latest_round_new_integer_normal_gap_incidence_coordinates']==0
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,preparation_actual_exits=prep_exits,preparation=read(D/'row_separation_preparation_510/preparation.json'),preparation_audit=read(D/'row_separation_audit_511/independent_audit.json'),main_result=m,independent_audit=a,saved_scope=scope,local_review=review,new_closure=True,accepted_solver_iterates=0,actual_final_state_changed=False,all_final_geometry_inherited=True,new_finite_work_columns=0,actual_scipy_lp_calls=2,partial_linear_witnesses=1,full_linear_witnesses=0,separation_cut_rounds=1,separation_added_rows=15885,finite_maps=0,nonlinear_trials=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_515.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',R/'docs/LOW_SEAT_CONTACT_STATUS.md',D/'handoff_update_515.txt',manifest,D/'closure_record_515.json',D/'separated_path_512/results.json',D/'separated_path_audit_513/independent_audit.json',D/'progress_report_514/findings.html',D/'progress_report_514_local_review.json']
 (D/'final_handoff_manifest_515.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','actual_scipy_lp_calls','partial_linear_witnesses','full_linear_witnesses','separation_added_rows','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
