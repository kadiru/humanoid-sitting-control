"""Preserve219–229 offline stance/path evidence; no physical repair admission."""
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
 assert not (D/'closure_record_230.json').exists()
 folders=['stance_diagnosis_219','stance_audit_220','centered_path_check_221','centered_path_audit_222','centered_contact_refinement_223','centered_contact_audit_224','lateral_demand_refinement_225','lateral_demand_audit_226','centered_orientation_refinement_v2_227','centered_orientation_audit_v2_228','orientation_force_recovery_227','progress_report_229']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 for f in ['stance_audit_220','centered_path_audit_222','centered_contact_audit_224','lateral_demand_audit_226','centered_orientation_audit_v2_228']:assert read(D/f/'independent_audit.json')['passed']
 for f in ['stance_diagnosis_219','centered_path_check_221','centered_contact_refinement_223','lateral_demand_refinement_225','centered_orientation_refinement_v2_227']:
  z=read(D/f/'results.json');assert z['physics_ticks']==0 and z['optimizer_updates']==0
  for p,h in z['source_sha256'].items():assert sha(p)==h,p
 review=read(D/'progress_report_229_local_review.json');assert review['local_hashes_verified'] and review['plots_visually_inspected'] and review['browser_visually_inspected'] and review['embedded_video_loaded']
 processes=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines();workers=[s for s in processes if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|bound|screen|render|verify|finalize|report|launch)_low_seat_\S+\.py',s)];assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_218.json','supplemental_preservation_218.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];checks.append(dict(manifest=name,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks);priorpath=D/'prior_recheck_after_stance_230.json';priorpath.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,n in [(status,'status_before_round_230.json'),(doc,'handoff_before_round_230.md'),(protocol,'protocol_before_round_230.md')]:assert not (D/n).exists();shutil.copy2(p,D/n)
 report=read(D/'progress_report_229/summary.json');assert not report['repair_complete'] and report['new_active_physics_ticks']==0
 p227=read(D/'centered_orientation_refinement_v2_227/results.json');best=read(D/'centered_orientation_refinement_v2_227/case_0.json');assert best['summary']['model_passed'] and best['summary']['research_pass'] and best['schedule']['time_gate_passed'];assert all(not z['research_pass'] for z in p227['rows'] if z['case']!=0)
 audits={f:read(D/f/'independent_audit.json').get('records') for f in ['centered_path_audit_222','centered_contact_audit_224','lateral_demand_audit_226','centered_orientation_audit_v2_228']}
 closed=dict(utc=now,phase='offline_stance_paths_audited_physical_repair_unresolved',physical_admission_passed=False,main_training_started=False,fresh_holdout_started=False,workers_running=False,physical_ticks=0,optimizer_updates=0,model_candidate_case0=best['summary'],audits=audits)
 (D/'stance_round_status_230.json').write_text(json.dumps(closed,indent=2))
 protected=dict(read(D/'completed_probes_preservation_218.json'));protected.update(read(D/'supplemental_preservation_218.json'));excluded={'closure_230.log','completed_probes_preservation_230.json','closure_record_230.json'}
 for p in D.iterdir():
  match=re.search(r'_(\d+)(?:\.py)?(?:\.[^.]*)?$',p.name)
  if not match or not 219<=int(match[1])<=230 or p.name in excluded:continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 219<=int(m[1])<=230:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_229_local_review.json')]=sha(D/'progress_report_229_local_review.json');mf=D/'completed_probes_preservation_230.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 notes=(D/'handoff_update_230.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound230:219–228 are completed offline stance diagnostics and bounded whole-path searches only. Zero new physics, settling, optimizer, holdout or promotion. Recorded force asymmetry/raw-velocity association is not causal proof.227v2 case0 passes sampled model and synthetic command schedule, not physical admission; other three remain failed. Original infeasible-allocation crash and five reused candidates preserved; combined case0 coarse-build cap40 unchanged. Read closure230/report229. No original physical, raw-speed quiet, material or exactness criterion changed.\n')
 st=read(status);st.update(utc=now,updated_utc=now,phase=closed['phase'],task_complete=False,repair_complete=False,physical_repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,model_job=None,latest_report=str(D/'progress_report_229'),latest_model_candidate=str(D/'centered_orientation_refinement_v2_227/case_0.json'),latest_model_audit=str(D/'centered_orientation_audit_v2_228/independent_audit.json'),latest_diagnosis=str(D/'stance_diagnosis_219'),latest_audit_status=str(D/'stance_round_status_230.json'),prior_recheck=str(priorpath),initial_prior_hash_check=priorpath.name,latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),latest_handoff=str(D/'handoff_update_230.txt'),local_report_original_hashes_verified=len(read(D/'progress_report_229/SHA256.json')),local_report_visually_inspected=True,latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_optimizer_updates=0,best_development_clean=64,best_development_total=72,remaining_slipping=8,remaining_housing=2,notes=notes)
 status.write_text(json.dumps(st,indent=2));closure=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=0,new_initial_settling_ticks=0,new_indexed_warmup_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',latest_physical_candidate='bias_probe_210',best_model_candidate=str(D/'centered_orientation_refinement_v2_227/case_0.json'),model_summary=best['summary'],all_model_summaries=p227['rows'],audits=audits,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_229',video='all_attempts_video_213/all_attempts.mp4',video_unchanged=True,local_review=review,handoff_sha256=sha(D/'handoff_update_230.txt'),next=notes)
 (D/'closure_record_230.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
