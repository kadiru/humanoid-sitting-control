"""Preserve new finite gap chart and LP assembly ONLY; sealed closure465 unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_471.json').exists();checks=[]
 for n in ['prior_hashes.json','gap_evidence_preservation_467.json','gap_supplemental_467.json']:
  m=read(D/n)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=n,files=len(m),passed=True))
 folders=['exact_gap_chart_preparation_468','exact_gap_chart_audit_v2_469','exact_gap_audit_association_recovery_469','free_gap_solver_preparation_470','free_gap_assembly_audit_471']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 for n in ['exact_gap_chart_launch_468.json','free_gap_preparation_launch_470.json']:
  for p,h in read(D/n)['source_input_sha256'].items():assert sha(p)==h,p
 a=read(D/'exact_gap_chart_audit_launch_v2_469.json');assert sha(a['source'])==a['source_sha256']
 ex=read(D/'free_gap_preparation_exits_471.json')
 for k in ['chart468_actual_exit','staged_audit469v2_actual_exit','assembly470_actual_exit','audit471_actual_exit']:assert ex[k]==0,k
 assert ex['original_audit469_actual_exit']==1
 prep=read(D/'exact_gap_chart_preparation_468/preparation.json');chart=read(D/'exact_gap_chart_audit_v2_469/independent_audit.json');assembly=read(D/'free_gap_solver_preparation_470/preparation.json');audit=read(D/'free_gap_assembly_audit_471/independent_audit.json')
 assert all(x['passed'] for x in [prep,chart,assembly,audit]);local=read(D/'free_gap_local_verification_471.json');assert local['all_hashes_verified'] and not local['visual_review_claimed']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_free_gap_471.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_free_gap_471.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'gap_evidence_preservation_467.json');mf.update(read(D/'gap_supplemental_467.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 names=['low_seat_exact_gap_chart_468.py','prepare_low_seat_exact_gap_chart_468.py','audit_low_seat_exact_gap_chart_469.py','audit_low_seat_exact_gap_chart_v2_469.py','low_seat_free_gap_solver_470.py','prepare_low_seat_free_gap_solver_470.py','prepare_low_seat_free_gap_solver_470_unexecuted_draft.py','audit_low_seat_free_gap_assembly_471.py','record_low_seat_free_gap_preparation_471.py']
 for n in names:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for pattern in ['exact_gap*46?.json','exact_gap*46?.log','free_gap*47?.json','free_gap*47?.log']:
  for p in D.glob(pattern):
   if p.is_file():mf[str(p)]=sha(p)
 mf.pop(str(D/'free_gap_preservation_471.log'),None)
 for n in ['handoff_prepared_471.txt','status_before_free_gap_471.json','handoff_before_free_gap_471.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'free_gap_evidence_preservation_471.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_471.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_exact_gap_chart_468_469_and_whole_row_470_471_prepared_no_candidate',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_prepared_471.txt'),latest_preparation_handoff=str(D/'preparation_handoff_record_471.json'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_465.json'),latest_preparation=str(D/'free_gap_solver_preparation_470'),latest_preparation_audit=str(D/'free_gap_assembly_audit_471/independent_audit.json'),latest_finite_gap_chart=str(D/'exact_gap_chart_preparation_468'),latest_finite_gap_chart_audit=str(D/'exact_gap_chart_audit_v2_469/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='Finite normal-drop exact residual pivot/rounding chart468 and staged469 audit complete. Whole-row prospective predictor/corrector470/471 assembly audited. No executable whole-path wrapper, complete state or optimizer run. Original469 finite prefix completed then sparse association assertion failed; staged469v2 inherits that prefix without repeating it.',latest_round_constructed_model_records=0,latest_round_interpolated_model_records=0,latest_round_new_complete_trial_paths=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=205,latest_round_new_arithmetic_finite_map_fixtures=618,latest_round_new_root_fixture_states=0,latest_round_new_root_fixture_scope='Zero new mechanics/AD/root/geometry/finite-work calls; this round uses saved matrices and exact arithmetic only.',latest_round_new_finite_work_columns=0,latest_round_inherited_geometry_records=0,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,new_solver_exists=False,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=False,new_candidate_exists=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,finite_chart_preparation=prep,staged_chart_audit=chart,whole_row_preparation=assembly,independent_assembly_audit=audit,local_hash_review=local,last_sealed_closure=str(D/'closure_record_465.json'),new_closure=False,new_solver_exists=False,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=False,new_candidate=False,model_calls=0,root_calls=0,geometry_calls=0,finite_work_columns=0,optimizer_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_471.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['finite_chart_preparation','staged_chart_audit','whole_row_preparation','independent_assembly_audit','local_hash_review']},indent=2),flush=True)
if __name__=='__main__':main()
