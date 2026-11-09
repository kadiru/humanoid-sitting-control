"""Close completed recording/audit/report round after local review, not task."""
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
 assert not (D/'closure_record_266.json').exists();assert read(D/'contact_wrench_finalization_status_264.json')['phase']=='complete'
 folders=['spatial_wrench_diagnosis_v2_255','spatial_wrench_audit_v2_256','spatial_contact_mapping_recovery_255','contact_recorder_preparation_258','contact_wrench_audit_260','contact_wrench_diagnosis_261','all_attempts_video_262','all_attempts_video_audit_263','progress_report_265']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 a=read(D/'contact_wrench_audit_260/independent_audit.json');assert a['diagnostic_checks_passed'] and a['exact_prefix_replay'];assert read(D/'spatial_wrench_audit_v2_256/independent_audit.json')['passed']
 for p,h in read(D/'contact_wrench_revision_259.json')['sources'].items():assert sha(p)==h,p
 review=read(D/'progress_report_265_local_review.json');assert all(review[k] for k in ['local_hashes_verified','plots_visually_inspected','browser_visually_inspected','embedded_video_loaded','video_frames_visually_inspected'])
 proc=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines();workers=[x for x in proc if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|audit|refine|optimize|diagnose|report|render|verify|finalize|launch|prepare|check)_low_seat_\S+\.py',x)];assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_253.json','supplemental_preservation_253.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks);pp=D/'prior_recheck_after_spatial_contact_266.json';pp.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_266.json'),(doc,'handoff_before_round_266.md'),(protocol,'protocol_before_round_266.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 notes=(D/'handoff_update_266.txt').read_text();protected=dict(read(D/'completed_probes_preservation_253.json'));protected.update(read(D/'supplemental_preservation_253.json'))
 excluded={'closure_266.log','completed_probes_preservation_266.json','closure_record_266.json'}
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 255<=int(m[1])<=266 or p.name in excluded:continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 255<=int(m[1])<=266:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_265_local_review.json')]=sha(D/'progress_report_265_local_review.json')
 mf=D/'completed_probes_preservation_266.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound266 closes offline spatial255/256 and contact-recording259 with independent260 audit and complete262/263 video. Exact prefix replay is diagnostic evidence only; no physical repair/PPO/holdout or criterion change. Read closure266/report265.\n')
 st=read(status);st.update(updated_utc=now,utc=now,phase='spatial_contact_diagnostic_audited_repair_unresolved',task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,latest_running_diagnostic=None,latest_diagnostic_running=False,latest_diagnostic_completed=True,latest_diagnostic_checks_passed=a['diagnostic_checks_passed'],latest_diagnostic_exact_prefix_replay=a['exact_prefix_replay'],latest_diagnostic=str(D/'contact_wrench_diagnostic_259'),latest_diagnostic_audit=str(D/'contact_wrench_audit_260'),latest_diagnostic_accounting=a['accounting'],latest_diagnostic_revision=str(D/'contact_wrench_revision_259.json'),latest_diagnostic_source_hashes_unchanged=True,latest_diagnostic_private_cache_records=1001,latest_diagnostic_internally_exact_raw_checks=a['substep_raw_checks'],latest_physical_candidate=str(D/'centered_probe_234'),latest_report=str(D/'progress_report_265'),latest_diagnosis=str(D/'contact_wrench_diagnosis_261'),latest_audit_status=str(D/'contact_wrench_finalization_status_264.json'),latest_video=str(D/'all_attempts_video_262/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_263'),video_ready_locally=True,video_duration_s=58.4,video_complete_trials=840,video_successes=489,video_failures=351,video_recorded_prefixes=168,video_aborted_case_attempts=96,prior_recheck=str(pp),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_266.txt'),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=24,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=15600,latest_round_initial_settling_ticks=600,latest_round_optimizer_updates=0,best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,notes=notes)
 status.write_text(json.dumps(st,indent=2))
 closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=15600,new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=24,new_aborted_case_attempts=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',latest_physical_candidate='centered_probe_234',diagnostic_checks_passed=a['diagnostic_checks_passed'],exact_prefix_replay=a['exact_prefix_replay'],prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_265',video='all_attempts_video_262/all_attempts.mp4',video_unchanged=False,local_review=review,next=notes)
 (D/'closure_record_266.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
