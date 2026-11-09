"""Close completed273 experiment/evidence, never declare repair or PPO complete."""
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
 assert not (D/'closure_record_281.json').exists();assert read(D/'roll_audit_recovery_status_278.json')['phase']=='complete'
 folders=['contact_velocity_diagnosis_267','contact_velocity_audit_268','roll_control_audit_v3_271','roll_runtime_preparation_272','roll_preparation_recovery_271','roll_audit_name_recovery_271','roll_audit_source_recovery_274','roll_audit_execution_recovery_278','roll_diagnosis_279','all_attempts_video_276','all_attempts_video_audit_277','progress_report_280']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 a=read(D/'roll_feedback_probe_273/independent_roll_audit_v2_274.json');assert a['software_audit_passed']
 assert read(D/'roll_feedback_probe_273/independent_dual_view_audit_v2_275.json')['software_audit_passed']
 for p,h in read(D/'roll_feedback_probe_273/revision.json')['sources'].items():assert sha(p)==h,p
 review=read(D/'progress_report_280_local_review.json');assert all(review[k] for k in ['local_hashes_verified','plots_visually_inspected','browser_visually_inspected','embedded_video_loaded','video_frames_visually_inspected'])
 procs=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines();workers=[p for p in procs if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|audit|refine|optimize|diagnose|report|render|verify|finalize|launch|prepare|check)_low_seat_\S+\.py',p)];assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_266.json','supplemental_preservation_266.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks);pp=D/'prior_recheck_after_roll_281.json';pp.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_281.json'),(doc,'handoff_before_round_281.md'),(protocol,'protocol_before_round_281.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 notes=(D/'handoff_update_281.txt').read_text();protected=dict(read(D/'completed_probes_preservation_266.json'));protected.update(read(D/'supplemental_preservation_266.json'))
 excluded={'closure_281.log','completed_probes_preservation_281.json','closure_record_281.json'}
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 267<=int(m[1])<=281 or p.name in excluded:continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file() and not f.is_symlink():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 267<=int(m[1])<=281:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_280_local_review.json')]=sha(D/'progress_report_280_local_review.json')
 mf=D/'completed_probes_preservation_281.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound281 closes one bounded raw-roll-feedback273 experiment and independent274v2/275v2 audits, recorded279 diagnosis and complete276/277 video. Original274 post-run syntax failure and finalizer failure remain preserved; corrected audits did not rerun physics/video or change runtime/criteria. No full72retention/PPO/holdout or promotion. Read closure281/report280.\n')
 v=read(D/'all_attempts_video_audit_277/independent_audit.json');st=read(status)
 st.update(updated_utc=now,utc=now,phase='roll_feedback_probe_audited_repair_unresolved',task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,latest_running_diagnostic=None,latest_diagnostic_running=False,latest_physical_candidate=str(D/'roll_feedback_probe_273'),latest_physical_audit=str(D/'roll_feedback_probe_273/independent_roll_audit_v2_274.json'),latest_physical_outcomes=a['candidate'],latest_physical_exact_preservation_gate_passed=a['exact_preservation_gate_passed'],latest_report=str(D/'progress_report_280'),latest_diagnosis=str(D/'roll_diagnosis_279'),latest_audit_status=str(D/'roll_audit_recovery_status_278.json'),latest_video=str(D/'all_attempts_video_276/all_attempts.mp4'),latest_video_audit=str(D/'all_attempts_video_audit_277'),video_ready_locally=True,video_duration_s=58.4,video_complete_trials=v['full_trials'],video_successes=v['successes'],video_failures=v['failures'],video_recorded_prefixes=v['recorded_prefixes'],video_aborted_case_attempts=v['aborted_prefixes_disclosed'],prior_recheck=str(pp),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_281.txt'),latest_round_new_full20s_trials=24,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=24000,latest_round_initial_settling_ticks=600,latest_round_optimizer_updates=0,best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,notes=notes)
 status.write_text(json.dumps(st,indent=2))
 closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=24000,new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=24,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',latest_physical_candidate='roll_feedback_probe_273',software_audit_passed=True,exact_preservation_gate_passed=a['exact_preservation_gate_passed'],candidate=a['candidate'],prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_280',video='all_attempts_video_276/all_attempts.mp4',video_unchanged=False,local_review=review,next=notes)
 (D/'closure_record_281.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
