"""Close verified private-buffer diagnostic198 without physical/PPO admission."""
from pathlib import Path
import datetime, hashlib, json, re, shutil, subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_206.json').exists()
 audit=read(D/'private_audit_v2_199/independent_audit.json')
 diagnosis=read(D/'private_diagnosis_200/results.json')
 video=read(D/'all_attempts_video_audit_202/independent_audit.json')
 assert read(D/'private_finalization_status_v2_204.json')['phase']=='complete'
 assert audit['diagnostic_checks_passed'] and audit['exact_prefix_replay']
 assert not audit['physical_admission'] and not audit['repair_complete']
 assert audit['all_substep_raw_exact'] and audit['substep_raw_checks']==2008
 assert audit['fixed_fingers']==14 and audit['source_hashes_unchanged']
 assert audit['postflight_completion_checked_from_saved_arrays']
 assert audit['post_context_completion_marker_present'] is False
 assert diagnosis['exact_prefix_replay'] and video['passed']
 assert [video[k] for k in ['source_runs','full_trials','successes','failures','recorded_prefixes','aborted_prefixes_disclosed','event_frames_checked']]==[38,792,461,331,120,96,554]
 meta=audit['recorder_metadata']
 assert meta['private_query_destinations'] and meta['records']==1001
 assert meta['every_read_public_cache_bytes_and_timestamps_exact'] and meta['every_backend_query_nonoverlapping_public_storage']
 assert meta['nonidentity_joint_ordering'] is False
 accounting=audit['accounting']
 assert accounting['active_ticks']==15600 and accounting['physics_substeps']==2600
 assert accounting['initial_settling_excluded']==600 and accounting['complete20s_episodes']==0
 assert accounting['optimizer_updates']==0 and not accounting['fresh_holdout']
 source_counts={}
 for name,key in [('private_revision_198.json','sources'),('passive_revision_196.json','sources'),('private_finalization_sources_v2_204.json',None)]:
  m=read(D/name);m=m[key] if key else m
  for p,h in m.items(): assert sha(p)==h,p
  source_counts[name]=len(m)
 assert source_counts['private_revision_198.json']==49 and source_counts['passive_revision_196.json']==47
 folders=['progress_report_203','private_audit_v2_199','private_diagnosis_200','private_recorder_preparation_198','passive_recorder_preparation_v2_196','passive_preparation_recovery_196','passive_storage_recovery_196','private_postflight_recovery_199','solver_context_205','all_attempts_video_201','all_attempts_video_audit_202']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items(): assert sha(D/folder/p)==h,(folder,p)
 for p,h in read(D/'progress_report_203/REVIEW_SHA256.json').items(): assert sha(D/'progress_report_203'/p)==h,p
 review=read(D/'progress_report_203_local_review.json')
 assert review['local_hashes_verified'] and review['total_hash_entries']==56
 assert review['browser_visually_inspected'] and review['video_frames_visually_inspected'] and review['embedded_video_loaded']
 assert not review['physical_admission']
 processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
 workers=[s for s in processes if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen|render|verify|finalize|report|launch)_low_seat_\S+\.py',s)]
 assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_195.json','supplemental_preservation_195.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h]
  checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 prior=dict(utc=now,passed=True,checks=checks)
 priorpath=D/'prior_recheck_after_private_206.json';priorpath.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,name in [(status,'status_before_round_206.json'),(doc,'handoff_before_round_206.md'),(protocol,'protocol_before_round_206.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 protected=dict(read(D/'completed_probes_preservation_195.json'));protected.update(read(D/'supplemental_preservation_195.json'))
 excluded={'closure_206.log','completed_probes_preservation_206.json','closure_record_206.json'}
 for p in D.iterdir():
  match=re.search(r'_(\d+)(?:\.py)?(?:\.[^.]*)?$',p.name)
  if not match or not 196<=int(match[1])<=206 or p.name in excluded: continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  match=re.search(r'_(\d+)\.py$',p.name)
  if match and 196<=int(match[1])<=206:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_203_local_review.json')]=sha(D/'progress_report_203_local_review.json')
 mf=D/'completed_probes_preservation_206.json';mf.write_text(json.dumps(protected,indent=2))
 assert all(sha(p)==h for p,h in protected.items())
 notes=(D/'handoff_update_206.txt').read_text()
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:
  f.write('\n\nRound206: private-destination recorder198 passed exact full13s original replay, including actual414 inputs, events and torque; all1001 public-cache before/after bytes/timestamps unchanged. 196 stopped0active+600settling;198 used15600active+600settling,zero full20s/optimizer/holdout. Diagnostic passes are not physical admission. Original189 global replay remains FAILED. Solver205 source/docs context changes no physics, guard or velocity. Report203/video201 audited and locally reviewed; see closure206.\n')
 st=read(status)
 st.update(utc=now,updated_utc=now,phase='private_diagnostic_complete_exact_replay_passed',task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_diagnostic_running=False,latest_running_diagnostic=None,latest_running_candidate=None,model_job=None,latest_candidate='touchdown_probe_178',latest_physical_candidate='touchdown_probe_178',latest_physical_admission_passed=False,latest_report=str(D/'progress_report_203'),latest_diagnostic=str(D/'private_diagnostic_198'),latest_diagnosis=str(D/'private_diagnosis_200'),latest_diagnostic_audit=str(D/'private_audit_v2_199/independent_audit.json'),latest_diagnostic_completed=True,latest_diagnostic_checks_passed=True,latest_diagnostic_exact_prefix_replay=True,latest_diagnostic_accounting=accounting,latest_diagnostic_source_hashes_unchanged=True,latest_diagnostic_internally_exact_raw_checks=2008,latest_diagnostic_private_cache_records=1001,earlier_diagnostic189_exact_failed_preserved=True,effort_telemetry_limitation=audit['effort_semantics'],prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_206.txt'),latest_video=str(D/'all_attempts_video_201/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_202/independent_audit.json'),video_ready_locally=True,video_duration_s=58.4,video_complete_trials=792,video_successes=461,video_failures=331,video_recorded_prefixes=120,video_aborted_case_attempts=96,local_report_original_hashes_verified=18,local_supplemental_hashes_verified=3,local_report_visually_inspected=True,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=24,latest_round_new_aborted_case_attempts=24,latest_round_new_active_ticks=15600,latest_round_initial_settling_ticks=1200,latest_round_optimizer_updates=0,best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,notes=notes)
 status.write_text(json.dumps(st,indent=2))
 closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=15600,new_initial_settling_ticks=1200,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=24,new_aborted_case_attempts=24,new_shared5ms_steps=2600,holdout_started=False,diagnostic='private_diagnostic_198',diagnostic_completed=True,diagnostic_checks_passed=True,exact_prefix_replay=True,physical_admission_passed=False,post_context_completion_marker_present=False,postflight_completion_checked_from_saved_arrays=True,earlier_diagnostic189_global_failure_preserved=True,internally_exact_raw_checks=2008,cache_checks=1001,frozen_source_counts=source_counts,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_203',video='all_attempts_video_201/all_attempts.mp4',video_summary={k:video[k] for k in ['full_trials','successes','failures','recorded_prefixes','aborted_prefixes_disclosed','event_frames_checked','source_runs']},local_review=review,handoff_sha256=sha(D/'handoff_update_206.txt'),next=notes)
 (D/'closure_record_206.json').write_text(json.dumps(closure,indent=2))
 print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,diagnostic_global_exact=True,task_complete=False)),flush=True)
if __name__=='__main__':main()
