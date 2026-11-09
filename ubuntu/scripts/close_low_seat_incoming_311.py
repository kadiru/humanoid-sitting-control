"""Preserve completed incoming diagnostic301–310; repair, PPO and holdout remain unfinished."""
from pathlib import Path
import datetime,json,hashlib,shutil,re,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'closure_record_311.json').exists()
 for folder in ['incoming_recorder_preparation_302','incoming_recorder_preparation_v2_302','incoming_allocation_recovery_304','incoming_audit_v2_305','incoming_diagnosis_306','incoming_diagnosis_audit_306','all_attempts_video_307','all_attempts_video_audit_308','progress_report_309']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 assert read(D/'incoming_audit_v2_305/independent_audit.json')['diagnostic_checks_passed'] and read(D/'incoming_diagnosis_audit_306/independent_audit.json')['passed'];review=read(D/'progress_report_309_local_review.json');assert all(review[k] for k in ['local_hashes_verified','plots_visually_inspected','browser_visually_inspected','embedded_video_loaded'])
 workers=[s for s in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines() if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|audit|refine|optimize|diagnose|report|render|verify|finalize|launch|prepare|check)_low_seat_\S+\.py',s)];assert not workers,workers
 checks=[]
 for n in ['prior_hashes.json','completed_probes_preservation_300.json','supplemental_preservation_300.json']:
  m=read(D/n);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];assert not bad,bad[:5];checks.append(dict(manifest=n,checked=len(m),passed=True,mismatches=[]))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks);(D/'prior_recheck_after_incoming_311.json').write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_311.json'),(doc,'handoff_before_round_311.md'),(protocol,'protocol_before_round_311.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 notes=(D/'handoff_update_311.txt').read_text();protected=read(D/'completed_probes_preservation_300.json');protected.update(read(D/'supplemental_preservation_300.json'))
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 301<=int(m[1])<=311 or p.name in ['closure_311.log','closure_record_311.json','completed_probes_preservation_311.json']:continue
  for f in (p.rglob('*') if p.is_dir() else [p]):
   if f.is_file() and not f.is_symlink():protected[str(f)]=sha(f)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 301<=int(m[1])<=311:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_309_local_review.json')]=sha(D/'progress_report_309_local_review.json')
 protected[str(D/'handoff_copy_recovery_300.json')]=sha(D/'handoff_copy_recovery_300.json')
 mf=D/'completed_probes_preservation_311.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound311: private incoming-wrench/backend-property diagnostic303v2 exactly replayed23413s. Allocation-only recovery preserves original0active/600settling aborted attempt; combined15600active+1200settling,24recordedprefixes+24abortedcases,zero full20s/PPO/holdout. Audits305v2/306 complete; all-attempt video307 includes every recorded/aborted source. No gains/guards/material/criteria change. Read closure311.\n')
 s=read(status);s.update(updated_utc=now,utc=now,phase='incoming_checks_complete_repair_unresolved',task_complete=False,repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,latest_report=str(D/'progress_report_309'),latest_diagnosis=str(D/'incoming_diagnosis_306'),latest_offline_audit=str(D/'incoming_diagnosis_audit_306/independent_audit.json'),latest_handoff=str(D/'handoff_update_311.txt'),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),prior_recheck=str(D/'prior_recheck_after_incoming_311.json'),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=24,latest_round_new_aborted_case_attempts=24,latest_round_new_active_ticks=15600,latest_round_initial_settling_ticks=1200,latest_round_optimizer_updates=0,notes=notes);status.write_text(json.dumps(s,indent=2))
 # Reconcile stale generic aliases with the already-recorded latest273 physical
 # fields; no historical result or source changes. Backup above retains aliases.
 s.update(latest_candidate='roll_feedback_probe_273',latest_candidate_outcome=s['latest_physical_outcomes'],latest_independent_audit=s['latest_physical_audit'],latest_offline_audit_passed=True,latest_offline_screen_scope='Passive incoming-wrench303v2 exact replay305v2 and independent frame/boundary comparison306. No controller change or physical admission')
 s.update(latest_diagnostic=str(D/'incoming_diagnostic_v2_303'),latest_diagnostic_audit=str(D/'incoming_audit_v2_305'),latest_diagnostic_revision=str(D/'incoming_revision_v2_304.json'),latest_diagnostic_running=False,latest_diagnostic_completed=True,latest_diagnostic_checks_passed=True,latest_diagnostic_exact_prefix_replay=True,latest_running_diagnostic=None,latest_video=str(D/'all_attempts_video_307/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_308'),video_recorded_sources=44,video_recorded_prefixes=192,video_aborted_case_attempts=120,video_panels=49,video_duration_s=58.4)
 status.write_text(json.dumps(s,indent=2))
 c=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=15600,new_initial_settling_ticks=1200,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=24,new_aborted_case_attempts=24,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',best_development_clean=64,latest_physical_candidate='roll_feedback_probe_273',prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_309',video='all_attempts_video_307/all_attempts.mp4',video_unchanged=False,local_review=review,mutable_status_aliases_reconciled_to_existing_physical_273=True,next=notes);(D/'closure_record_311.json').write_text(json.dumps(c,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
