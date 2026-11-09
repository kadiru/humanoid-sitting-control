"""Preserve completed454–458 evidence; no accepted new path or user-task completion."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_459.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_453.json','supplemental_preservation_453.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['bounded_retraction_preparation_454','bounded_retraction_audit_455','bounded_path_456','bounded_path_audit_457','progress_report_458','bounded_report_readability_459']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 post=read(D/'bounded_postprocessing_458.json');assert post['main_actual_exit']==post['independent_audit_actual_exit']==post['report_actual_exit']==0
 review=read(D/'progress_report_458_local_review.json');assert review['visual_review_completed']
 workers=[]
 for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
  a=l.split()
  if len(a)>2 and a[1].endswith('IsaacLab/.venv/bin/python') and any(x.startswith('scripts/') and Path(x).name!=Path(__file__).name for x in a[2:]):workers.append(l)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_bounded_459.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_bounded_459.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_453.json');mf.update(read(D/'supplemental_preservation_453.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 sources=['low_seat_bounded_retraction_454.py','prepare_low_seat_bounded_retraction_454.py','audit_low_seat_bounded_retraction_455.py','freeze_low_seat_bounded_path_456.py','build_low_seat_bounded_path_456.py','audit_low_seat_bounded_steps_457.py','audit_low_seat_bounded_path_457.py','audit_low_seat_bounded_result_457.py','index_low_seat_bounded_459.py','report_low_seat_bounded_458.py','close_low_seat_bounded_459.py']
 for n in sources:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 names=['bounded_retraction_preparation_454.log','bounded_retraction_audit_455.log','bounded_preparation_exit_454.json','bounded_preparation_audit_exit_455.json','bounded_launch_456.log','bounded_path_launch_456.json','bounded_path_456.log','bounded_path_exit_456.json','bounded_audit_launch_457.json','bounded_path_audit_457.log','bounded_index_459.log','bounded_saved_localization_459.json','bounded_conservation_defect_459.json','bounded_postprocessing_before_report_458.json','bounded_postprocessing_458.json','bounded_report_458.log','bounded_local_hashes_459.json','progress_report_458_local_review.json','report_low_seat_bounded_458_unexecuted_draft.py','status_before_bounded_459.json','handoff_before_bounded_459.md','handoff_update_459.txt']
 for n in names:mf[str(D/n)]=sha(D/n)
 manifest=D/'completed_probes_preservation_459.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_459.txt').read_text();status=read(D/'status.json');result=read(D/'bounded_path_456/results.json');audit=read(D/'bounded_path_audit_457/independent_audit.json')
 assert result['accepted_steps']==0 and not result['new_final_differs_from_source'];assert audit['new_finite_work_columns']==0
 status.update(utc=now,updated_utc=now,phase='offline_bounded_path_456_rejected_no_accepted_path',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_459.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_459.json'),latest_preparation=str(D/'bounded_retraction_preparation_454'),latest_preparation_audit=str(D/'bounded_retraction_audit_455/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='New bounded-map arithmetic preparation454/455; construction456 accepted no path; audit457 inherits unchanged441 full-state evidence.',latest_nominal_research_attempt=str(D/'bounded_path_456/results.json'),latest_constructed_candidate=str(D/'collocation_path_450/results.json'),latest_candidate_status='450 accepted final remains preserved: original rates/targets fail, hard phases/static constraints pass. New456 accepted zero paths; its export is original440, not a new candidate.',latest_audit=str(D/'bounded_path_audit_457/independent_audit.json'),latest_audit_status='457 complete saved-array arithmetic pass; all new trials rejected; original440 full-state evidence inherited without rerun',latest_audit_passed=True,latest_report=str(D/'progress_report_458/readable.html'),latest_report_local_review=str(D/'progress_report_458_local_review.json'),latest_round_constructed_model_records=0,latest_round_new_complete_trial_paths=26,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=11906,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=0,latest_round_inherited_geometry_records=1467,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=27,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=6,latest_round_offline_nonlinear_trial_evaluations=26,latest_round_offline_accepted_steps=0,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits={'454':'53b5db/session8197/exit0','455':'8d7cc1/immediate/exit0','456':'146a69/session79855/exit0','457':post['independent_audit_exit_evidence'],'458':post['report_exit_evidence'],'459_index':'85e5bf/immediate/exit0'},local_review=review,solver_summary=result,independent_audit_summary={k:v for k,v in audit.items() if k!='step_audit'},source450_final_unchanged=True,source440_export_exact=True,no_geometry_or_finite_work_repeat=True,trajectory_lp_calls=27,accepted_paths=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,video307_unchanged=True,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_459.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['local_review','solver_summary','independent_audit_summary']},indent=2),flush=True)
if __name__=='__main__':main()
