"""Close one bounded centered-stance experiment after audits/local visual review."""
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
 assert not (D/'closure_record_242.json').exists()
 assert read(D/'centered_finalization_status_239.json')['phase']=='complete'
 folders=['centered_runtime_paths_231','centered_geometry_audit_232','centered_control_audit_232','centered_runtime_preparation_233','centered_diagnosis_240','centered_input_crosscheck_240','all_attempts_video_237','all_attempts_video_audit_238','progress_report_241']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 for folder,file in [('centered_runtime_paths_231','results.json'),('centered_geometry_audit_232','independent_audit.json'),('centered_control_audit_232','independent_audit.json'),('centered_runtime_preparation_233','preparation.json')]:
  z=read(D/folder/file);assert z['passed']
  for p,h in z['source_sha256'].items():assert sha(p)==h,p
 F=D/'centered_probe_234';a=read(F/'independent_centered_audit_235.json');v=read(F/'independent_dual_view_audit_236.json');assert a['software_audit_passed'] and v['software_audit_passed']
 for p,h in read(F/'revision.json')['sources'].items():assert sha(p)==h,p
 review=read(D/'progress_report_241_local_review.json');assert all(review[k] for k in ['local_hashes_verified','plots_visually_inspected','browser_visually_inspected','embedded_video_loaded','video_frames_visually_inspected'])
 procs=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines();workers=[p for p in procs if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|audit|refine|optimize|diagnose|report|render|verify|finalize|launch|prepare|check)_low_seat_\S+\.py',p)];assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_230.json','supplemental_preservation_230.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks);pp=D/'prior_recheck_after_centered_242.json';pp.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_242.json'),(doc,'handoff_before_round_242.md'),(protocol,'protocol_before_round_242.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 report=read(D/'progress_report_241/summary.json');assert not report['repair_complete'] and report['new_active_ticks']==24000
 notes=(D/'handoff_update_242.txt').read_text();protected=dict(read(D/'completed_probes_preservation_230.json'));protected.update(read(D/'supplemental_preservation_230.json'))
 excluded={'closure_242.log','completed_probes_preservation_242.json','closure_record_242.json'}
 for p in D.iterdir():
  match=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not match or not 231<=int(match[1])<=242 or p.name in excluded:continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 231<=int(m[1])<=242:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_241_local_review.json')]=sha(D/'progress_report_241_local_review.json')
 mf=D/'completed_probes_preservation_242.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound242: centered234 physical experiment, independent235/236 audits, recorded240 diagnosis and complete237/238 video are closed. This does not admit a repaired controller or PPO. Actual physical/exactness failures remain reported; no criteria changed. Read closure242/report241.\n')
 st=read(status);st.update(updated_utc=now,utc=now,phase='centered_physical_probe_audited_repair_unresolved',task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,latest_physical_candidate=str(F),latest_report=str(D/'progress_report_241'),latest_diagnosis=str(D/'centered_diagnosis_240'),latest_audit_status=str(D/'centered_finalization_status_239.json'),latest_video=str(D/'all_attempts_video_237/all_attempts.mp4'),prior_recheck=str(pp),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_242.txt'),latest_round_new_full20s_trials=24,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=24000,latest_round_initial_settling_ticks=600,latest_round_optimizer_updates=0,best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,notes=notes)
 status.write_text(json.dumps(st,indent=2))
 closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=24000,new_initial_settling_ticks=600,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=24,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',latest_physical_candidate='centered_probe_234',candidate=a['candidate'],gains=a['gained_best_clean_cases'],losses=a['lost_best_clean_cases'],remaining_target_cases=a['remaining_target_cases'],exact_preservation_gate_passed=a['exact_preservation_gate_passed'],prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_241',video='all_attempts_video_237/all_attempts.mp4',video_unchanged=False,local_review=review,next=notes)
 (D/'closure_record_242.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
