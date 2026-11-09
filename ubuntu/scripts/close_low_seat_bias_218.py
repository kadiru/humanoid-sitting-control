"""Preserve bounded-bias probe210 and its audits, without repair/PPO admission."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_218.json').exists()
 F=D/'bias_probe_210';a=read(F/'independent_bias_audit_211.json');v=read(F/'independent_dual_view_audit_212.json');vid=read(D/'all_attempts_video_audit_214/independent_audit.json');report=read(D/'progress_report_217/summary.json')
 assert read(D/'bias_finalization_status_215.json')['phase']=='complete'
 assert a['audit_completed'] and a['software_audit_passed'] and v['software_audit_passed'] and vid['passed']
 assert a['selected_filter_records']==3800 and a['settling_records']==3800 and a['balance_records']==3800
 assert a['fixed14fingers_exact'] and a['every20ms_cap_checked'] and a['every20ms_selected_filter_replay_exact']
 assert a['every20ms_settling_raw_replay_exact'] and a['every20ms_balance_raw_replay_exact']
 assert a['physical_admission_passed'] is False and not report['repair_complete']
 assert len(read(F/'candidate/results.json')['episodes'])==24
 assert vid['source_runs']==39 and vid['full_trials']==816 and vid['recorded_prefixes']==120 and vid['aborted_prefixes_disclosed']==96
 revision=read(F/'revision.json');assert revision['active_ticks']==24000 and revision['initial_settling_excluded']==600 and revision['optimizer_updates']==0 and not revision['fresh_holdout']
 source_sets=[revision['sources'],read(D/'bias_finalization_sources_215.json'),read(D/'bias_postflight_sources_217.json')]
 for source in source_sets:
  for p,h in source.items():assert sha(p)==h,p
 folders=['bias_feedback_checks_v3_207','bias_control_audit_v3_209','bias_runtime_preparation_v3_209','bias_preparation_recovery_207','bias_antiwindup_recovery_207','bias_fixture_recovery_209','bias_launch_recovery_210','bias_diagnosis_216','all_attempts_video_213','all_attempts_video_audit_214','progress_report_217']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 for p,h in read(D/'progress_report_217/REVIEW_SHA256.json').items():assert sha(D/'progress_report_217'/p)==h,p
 review=read(D/'progress_report_217_local_review.json');assert review['local_hashes_verified'] and review['browser_visually_inspected'] and review['video_frames_visually_inspected'] and review['embedded_video_loaded'] and not review['physical_admission']
 processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
 workers=[s for s in processes if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen|render|verify|finalize|report|launch)_low_seat_\S+\.py',s)]
 assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_206.json','supplemental_preservation_206.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
  checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks)
 priorpath=D/'prior_recheck_after_bias_218.json';priorpath.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_218.json'),(doc,'handoff_before_round_218.md'),(protocol,'protocol_before_round_218.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 closed=dict(utc=now,phase='complete_repair_not_admitted',candidate=a['candidate'],software_audit_passed=True,exact_preservation_gate_passed=a['exact_preservation_gate_passed'],physical_admission_passed=False,remaining_target_cases=a['remaining_target_cases'],main_training_started=False,fresh_holdout_started=False,workers_running=False)
 (D/'bias_round_status_218.json').write_text(json.dumps(closed,indent=2))
 protected=dict(read(D/'completed_probes_preservation_206.json'));protected.update(read(D/'supplemental_preservation_206.json'))
 excluded={'closure_218.log','completed_probes_preservation_218.json','closure_record_218.json'}
 for p in D.iterdir():
  match=re.search(r'_(\d+)(?:\.py)?(?:\.[^.]*)?$',p.name)
  if not match or not 207<=int(match[1])<=218 or p.name in excluded:continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 207<=int(m[1])<=218:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_217_local_review.json')]=sha(D/'progress_report_217_local_review.json')
 mf=D/'completed_probes_preservation_218.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 notes=(D/'handoff_update_218.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound218 closure: one bounded integral hover-bias210 full24case20s probe,24000active+600settling,zerooptimizer/holdout. Software inputs/guard/integral/filter/parent/view audits completed; exact/physical outcomes retained separately. No full72retention or PPO admission. Report217/video213 locally verified. Original offline failures preserved. See closure218.\n')
 st=read(status);st.update(new_physical_ticks_since43=st['new_physical_ticks_since43']+24000,new_initial_settling_ticks_since43=st['new_initial_settling_ticks_since43']+600,local_supplemental_hashes_verified=3,utc=now,updated_utc=now,phase='bounded_bias_probe_audits_complete_repair_unresolved',task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,latest_candidate='bias_probe_210',latest_physical_candidate='bias_probe_210',latest_candidate_outcome=a['candidate'],latest_physical_outcomes=a['candidate'],latest_software_audit_passed=True,latest_independent_audit_passed=a['passed'],latest_exact_preservation_gate_passed=a['exact_preservation_gate_passed'],latest_physical_exact_preservation_passed=a['exact_preservation_gate_passed'],latest_physical_admission_passed=False,latest_report=str(D/'progress_report_217'),latest_diagnosis=str(D/'bias_diagnosis_216'),latest_independent_audit=str(F/'independent_bias_audit_211.json'),latest_audit_status=str(D/'bias_round_status_218.json'),prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_218.txt'),latest_video=str(D/'all_attempts_video_213/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_214/independent_audit.json'),video_ready_locally=True,video_duration_s=58.4,video_complete_trials=vid['full_trials'],video_successes=vid['successes'],video_failures=vid['failures'],video_recorded_prefixes=120,video_aborted_case_attempts=96,local_report_original_hashes_verified=len(read(D/'progress_report_217/SHA256.json')),local_report_visually_inspected=True,latest_round_new_full20s_trials=24,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=24000,latest_round_initial_settling_ticks=600,latest_round_optimizer_updates=0,best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,notes=notes)
 status.write_text(json.dumps(st,indent=2))
 closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=24000,new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=24,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,holdout_started=False,candidate='bias_probe_210',candidate_outcomes=a['candidate'],software_audit_passed=True,exact_preservation_gate_passed=a['exact_preservation_gate_passed'],physical_admission_passed=False,remaining_target_cases=a['remaining_target_cases'],frozen_source_counts=[len(x) for x in source_sets],prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_217',video='all_attempts_video_213/all_attempts.mp4',video_summary={k:vid[k] for k in ['full_trials','successes','failures','recorded_prefixes','aborted_prefixes_disclosed','event_frames_checked','source_runs']},local_review=review,handoff_sha256=sha(D/'handoff_update_218.txt'),next=notes)
 (D/'closure_record_218.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
