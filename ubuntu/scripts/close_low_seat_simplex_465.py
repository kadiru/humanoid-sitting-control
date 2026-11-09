"""Seal completed conserved-load construction evidence; no accepted new path or user-task completion."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_465.json').exists()
 checks=[]
 for name in ['prior_hashes.json','simplex_evidence_preservation_461.json','simplex_supplemental_461.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['simplex_solver_preparation_462','simplex_solver_preparation_audit_463','simplex_path_462','simplex_path_audit_463','progress_report_464']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 post=read(D/'simplex_postprocessing_464.json');assert post['main_actual_exit']==post['independent_audit_actual_exit']==post['report_actual_exit']==0
 assert post['report_exit_evidence']=='d50495/session84829'
 review=read(D/'progress_report_464_local_review.json');assert review['visual_review_completed']
 result=read(D/'simplex_path_462/results.json');audit=read(D/'simplex_path_audit_463/independent_audit.json')
 assert result['accepted_steps']==0 and not result['new_final_differs_from_source'];assert audit['passed'] and audit['new_finite_work_columns']==0
 workers=[]
 for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
  a=l.split()
  if len(a)>2 and a[1].endswith('IsaacLab/.venv/bin/python') and any(x.startswith('scripts/') and Path(x).name!=Path(__file__).name for x in a[2:]):workers.append(l)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_simplex_close_465.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_simplex_close_465.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'simplex_evidence_preservation_461.json');mf.update(read(D/'simplex_supplemental_461.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 sources=['low_seat_simplex_solver_462.py','prepare_low_seat_simplex_solver_462.py','build_low_seat_simplex_path_462.py','freeze_low_seat_simplex_path_462.py','audit_low_seat_simplex_assembly_463.py','audit_low_seat_simplex_steps_463.py','audit_low_seat_simplex_path_463.py','audit_low_seat_simplex_result_463.py','index_low_seat_simplex_465.py','report_low_seat_simplex_464.py','report_low_seat_simplex_464_unexecuted_draft.py','close_low_seat_simplex_465.py']
 for n in sources:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 names=['active_simplex_462.json','simplex_path_audit_463.log','simplex_solver_preparation_exits_462.json','simplex_saved_index_465.log','simplex_solver_preparation_462.log','simplex_saved_localization_465.json','simplex_report_464.log','simplex_path_freeze_462.log','simplex_postprocessing_before_report_464.json','simplex_path_exit_462.json','status_before_simplex_run_462.json','simplex_postprocessing_464.json','simplex_conservation_scope_465.json','simplex_solver_preparation_audit_463.log','simplex_postprocessing_premature_exit_label_464.json','handoff_running_462.txt','simplex_path_launch_462.json','simplex_path_462.log','handoff_before_simplex_run_462.md','simplex_local_hashes_465.json','progress_report_464_local_review.json','status_before_simplex_close_465.json','handoff_before_simplex_close_465.md','handoff_update_465.txt']
 for n in names:mf[str(D/n)]=sha(D/n)
 manifest=D/'completed_probes_preservation_465.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_update_465.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_conserved_load_path_462_rejected_zero_accepted',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_update_465.txt'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_465.json'),latest_preparation=str(D/'simplex_solver_preparation_462'),latest_preparation_audit=str(D/'simplex_solver_preparation_audit_463/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='462/463 saved reduced assembly; conserved bounded-load460/461 chart retained. Main462 accepted zero paths; audit463 inherits exact441 geometry/work.',latest_nominal_research_attempt=str(D/'simplex_path_462/results.json'),latest_constructed_candidate=str(D/'collocation_path_450/results.json'),latest_candidate_status='450 accepted final remains preserved: original rates/targets fail. New462 accepted zero paths; its export is exact440 input.',latest_audit=str(D/'simplex_path_audit_463/independent_audit.json'),latest_audit_status='463 complete arithmetic pass; all26 new trials preserve bounds/normal but fail hard path rows; exact440 export inherits sealed441 without rerun.',latest_audit_passed=True,latest_report=str(D/'progress_report_464/findings.html'),latest_report_local_review=str(D/'progress_report_464_local_review.json'),latest_round_constructed_model_records=0,latest_round_new_complete_trial_paths=26,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=0,latest_round_inherited_geometry_records=1467,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=27,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=6,latest_round_offline_nonlinear_trial_evaluations=26,latest_round_offline_accepted_steps=0,new_solver_exists=True,new_candidate_exists=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits={'462_preparation':'8c8feb/session21387/exit0','463_assembly':'9a0220/session22304/exit0','462_freeze':'da136f/session57019/exit0','462_main':post['main_exit_evidence'],'463_result':post['independent_audit_exit_evidence'],'464_report':post['report_exit_evidence'],'465_index':post['saved_index_exit_evidence']},local_review=review,solver_summary=result,independent_audit_summary={k:v for k,v in audit.items() if k!='step_audit'},conservation=read(D/'simplex_conservation_scope_465.json'),source450_final_unchanged=True,source440_export_exact=True,no_geometry_or_finite_work_repeat=True,trajectory_lp_calls=27,accepted_paths=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,video307_unchanged=True,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'closure_record_465.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['local_review','solver_summary','independent_audit_summary','conservation']},indent=2),flush=True)
if __name__=='__main__':main()
