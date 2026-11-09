"""Preserve completed offline290–292; repair, PPO and holdout remain unfinished."""
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
 assert not (D/'closure_record_293.json').exists()
 for folder in ['contact_dynamics_screen_v3_290','contact_dynamics_audit_v3_291','progress_report_292']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 assert read(D/'contact_dynamics_audit_v3_291/independent_audit.json')['passed'];review=read(D/'progress_report_292_local_review.json');assert all(review[k] for k in ['local_hashes_verified','plots_visually_inspected','browser_visually_inspected','embedded_video_loaded'])
 workers=[s for s in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines() if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|audit|refine|optimize|diagnose|report|render|verify|finalize|launch|prepare|check)_low_seat_\S+\.py',s)];assert not workers,workers
 checks=[]
 for n in ['prior_hashes.json','completed_probes_preservation_289.json','supplemental_preservation_289.json']:
  m=read(D/n);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];assert not bad,bad[:5];checks.append(dict(manifest=n,checked=len(m),passed=True,mismatches=[]))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks);(D/'prior_recheck_after_dynamics_293.json').write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_293.json'),(doc,'handoff_before_round_293.md'),(protocol,'protocol_before_round_293.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 notes=(D/'handoff_update_293.txt').read_text();protected=read(D/'completed_probes_preservation_289.json');protected.update(read(D/'supplemental_preservation_289.json'))
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 290<=int(m[1])<=293 or p.name in ['closure_293.log','closure_record_293.json','completed_probes_preservation_293.json']:continue
  for f in (p.rglob('*') if p.is_dir() else [p]):
   if f.is_file() and not f.is_symlink():protected[str(f)]=sha(f)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 290<=int(m[1])<=293:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_292_local_review.json')]=sha(D/'progress_report_292_local_review.json')
 mf=D/'completed_probes_preservation_293.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound293: completed offline authored-inertia/contact-response290v3/independent291v3/report292 only. Zero new physics/optimizer/holdout. Floating pelvis and bilateral foot constraints,43 generalized coordinates,2 existing roll input derivatives. Assumed4x5ms backward-Euler response is not certified PhysX dynamics. Exact authored/frozen mass equality remains false for6finger masses; preserved sources, no adjustment. No runtime or physical admission. Original failed extraction/audit and narrow recoveries preserved. Video276 unchanged. Read closure293.\n')
 s=read(status);s.update(updated_utc=now,utc=now,phase='contact_dynamics_screen_complete_repair_unresolved',task_complete=False,repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,latest_report=str(D/'progress_report_292'),latest_diagnosis=str(D/'contact_dynamics_screen_v3_290'),latest_offline_audit=str(D/'contact_dynamics_audit_v3_291/independent_audit.json'),latest_handoff=str(D/'handoff_update_293.txt'),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),prior_recheck=str(D/'prior_recheck_after_dynamics_293.json'),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_optimizer_updates=0,notes=notes);status.write_text(json.dumps(s,indent=2))
 # Reconcile stale generic aliases with the already-recorded latest273 physical
 # fields; no historical result or source changes. Backup above retains aliases.
 s.update(latest_candidate='roll_feedback_probe_273',latest_candidate_outcome=s['latest_physical_outcomes'],latest_independent_audit=s['latest_physical_audit'],latest_offline_audit_passed=True,latest_offline_screen_scope='290v3 conditional frozen contact/drive sensitivity only; no actual dynamics or trajectory/runtime/physical admission')
 status.write_text(json.dumps(s,indent=2))
 c=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=0,new_initial_settling_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',best_development_clean=64,latest_physical_candidate='roll_feedback_probe_273',prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_292',video='all_attempts_video_276/all_attempts.mp4',video_unchanged=True,local_review=review,mutable_status_aliases_reconciled_to_existing_physical_273=True,next=notes);(D/'closure_record_293.json').write_text(json.dumps(c,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
