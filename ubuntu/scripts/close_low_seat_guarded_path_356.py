"""Seal completed offline351–355 evidence without claiming physical repair."""
from pathlib import Path
import json,hashlib,datetime,re,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_356.json').exists()
 folders=['guarded_path_preparation_351','guarded_path_351','guarded_path_audit_352','guarded_schedule_not_run_353','guarded_path_failure_354','progress_report_355']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 for name in ['review_SHA256.json','readable_SHA256.json']:
  for p,h in read(D/'progress_report_355'/name).items():assert sha(D/'progress_report_355'/p)==h
 a=read(D/'guarded_path_audit_352/independent_audit.json');assert a['passed'] and not a['physical_admission']
 x=read(D/'guarded_path_exit_351.json');assert x['observed_exit_code']==0 and x['model_results_complete'] and x['source_hashes_intact'] and not x['optimizer_success']
 r=read(D/'guarded_path_351/results.json');o=r['optimizer'];d=r['dense']
 assert not o['success'] and o['iterations']==40 and o['evaluations']==60 and not o['numerical_constraints_passed']
 assert all(not d[k] for k in ['model_geometry_passed','model_command_passed','guarded_timing_constraints_passed','physical_admission'])
 assert not read(D/'guarded_schedule_not_run_353/not_executed.json')['executed']
 review=read(D/'progress_report_355_local_review.json');assert all(review[k] for k in ['local_hashes_verified','visual_layout_inspected','browser_visually_inspected','embedded_video_loaded'])
 workers=[s for s in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines() if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|build|bound|audit|refine|optimize|diagnose|report|review|schedule|render|verify|finalize|launch|prepare|check|screen)_low_seat_\S+\.py',s)]
 assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_350.json','supplemental_preservation_350.json']:
  data=read(D/name);bad=[p for p,h in data.items() if not Path(p).exists() or sha(p)!=h];assert not bad,bad[:5]
  checks.append(dict(manifest=name,checked=len(data),passed=True,mismatches=[]))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks)
 priorpath=D/'prior_recheck_after_guarded_path_356.json';priorpath.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,name in [(status,'status_before_round_356.json'),(doc,'handoff_before_round_356.md'),(protocol,'protocol_before_round_356.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 notes=(D/'handoff_update_356.txt').read_text();protected=read(D/'completed_probes_preservation_350.json');protected.update(read(D/'supplemental_preservation_350.json'))
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 351<=int(m[1])<=356 or p.name in ['closure_356.log','closure_record_356.json','completed_probes_preservation_356.json']:continue
  for f in (p.rglob('*') if p.is_dir() else [p]):
   if f.is_file() and not f.is_symlink():protected[str(f)]=sha(f)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 351<=int(m[1])<=356:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_355_local_review.json')]=sha(D/'progress_report_355_local_review.json')
 mf=D/'completed_probes_preservation_356.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound356: completed one40-iteration351 guarded115-knot offline solve, process exit0 but optimizer Iteration limit reached and all geometry/static-command/NLP/timing gates FAILED.18 failedgeometryknots41–58: maxfoot5.805006mm, CoP4.172739mm; housing3.731078mm passes. Commandmargin.029999477rad fails.03. Infeasibleobjective13.558908s is neither schedule nor lowerbound; worst linear desiredtarget deficit.010116231rad. Independent352 arithmetic passed;354 localized all18319 savedconstraints.353 scheduler remainsunexecuted. No restart/extension, no actualfilter/runtime/bothplacement/physics/PPO/holdout. No scientificexecutionrecovery. Report355 reviewed; originalartifacts preserved. All64protected/all8unresolved; video307 unchanged. Next design must separate feasibility/conditioning/timing using savedfailuregroups; no tolerance/authority/material/quietguard change or globalinfeasibility claim. See handoff356.\n')
 s=read(status);s.update(updated_utc=now,utc=now,phase='offline_guarded_complete_path_351_rejected_repair_unresolved',task_complete=False,repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_diagnostic_running=False,latest_running_candidate=None,latest_running_diagnostic=None,latest_report=str(D/'progress_report_355'),latest_diagnosis=str(D/'guarded_path_failure_354'),latest_offline_path=str(D/'guarded_path_351'),latest_offline_paths=[str(D/'guarded_path_351')],latest_offline_bound=None,latest_round_force_lp_started=False,latest_offline_audit=str(D/'guarded_path_audit_352/independent_audit.json'),latest_offline_audit_passed=True,latest_offline_command_arithmetic_passed=True,latest_offline_time_gate_passed=False,latest_offline_geometry_passed=False,latest_offline_static_commands_passed=False,latest_offline_nlp_passed=False,latest_offline_gate_scope='Completed351 failed geometry/staticcommands/NLP/guardedtiming;352 arithmetic pass only.353 notexecuted; no physicaladmission.',latest_offline_screen_scope='One115-knot completepath failed18geometryknots; margin.029999477rad and infeasible13.558908s fail unchanged.03/13.5 gates. No actualfilter/quietguard/runtime/physics.',latest_handoff=str(D/'handoff_update_356.txt'),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),prior_recheck=str(priorpath),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_optimizer_updates=0,latest_round_offline_nonlinear_iterations=40,latest_round_command_ticks_per_device=0,latest_round_hover_guard_exercised=False,notes=notes);status.write_text(json.dumps(s,indent=2))
 c=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=0,new_initial_settling_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,new_command_schedule_ticks=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',best_development_clean=64,latest_physical_candidate='roll_feedback_probe_273',offline_optimizer=o,offline_geometry_passed=False,offline_static_commands_passed=False,actual_schedule_executed=False,hover_guard_exercised=False,observed_solver_process_exit=x,independent_audit_passed=True,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_355/readable.html',video='all_attempts_video_307/all_attempts.mp4',video_unchanged=True,local_review=review,next=notes)
 (D/'closure_record_356.json').write_text(json.dumps(c,indent=2));print(json.dumps(dict(utc=now,prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],supplemental=checks[2]['checked'],protected_completed_files=len(protected),manifest_sha256=sha(mf),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
