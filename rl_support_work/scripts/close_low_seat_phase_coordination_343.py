"""Preserve completed offline336–342; physical repair/PPO/holdout remain unfinished."""
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
 assert not (D/'closure_record_343.json').exists()
 for folder in ['coordinated_landing_path_336','coordinated_landing_audit_337','decoupled_phase_path_338','decoupled_phase_audit_339','minimum_extension_path_340','minimum_extension_audit_341','progress_report_342']:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 for folder in ['coordinated_landing_audit_337','decoupled_phase_audit_339','minimum_extension_audit_341']:
  audit=read(D/folder/'independent_audit.json');assert audit['passed'];assert not audit['physical_admission']
 for p,h in read(D/'progress_report_342/review_SHA256.json').items():assert sha(D/'progress_report_342'/p)==h
 review=read(D/'progress_report_342_local_review.json');assert all(review[k] for k in ['local_hashes_verified','visual_layout_inspected','browser_visually_inspected','embedded_video_loaded'])
 workers=[s for s in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines() if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|build|bound|audit|refine|optimize|diagnose|report|render|verify|finalize|launch|prepare|check|screen)_low_seat_\S+\.py',s)]
 assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_335.json','supplemental_preservation_335.json']:
  data=read(D/name);bad=[p for p,h in data.items() if not Path(p).exists() or sha(p)!=h];assert not bad,bad[:5]
  checks.append(dict(manifest=name,checked=len(data),passed=True,mismatches=[]))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks)
 (D/'prior_recheck_after_phase_coordination_343.json').write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,name in [(status,'status_before_round_343.json'),(doc,'handoff_before_round_343.md'),(protocol,'protocol_before_round_343.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 notes=(D/'handoff_update_343.txt').read_text();protected=read(D/'completed_probes_preservation_335.json');protected.update(read(D/'supplemental_preservation_335.json'))
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 336<=int(m[1])<=343 or p.name in ['closure_343.log','closure_record_343.json','completed_probes_preservation_343.json']:continue
  for f in (p.rglob('*') if p.is_dir() else [p]):
   if f.is_file() and not f.is_symlink():protected[str(f)]=sha(f)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 336<=int(m[1])<=343:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_342_local_review.json')]=sha(D/'progress_report_342_local_review.json')
 mf=D/'completed_probes_preservation_343.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound343: zero-physics complete nominal landing coordination336/338/340 and independent337/339/341 complete. Three distinct one-placement36cm/+20 paths,115knots each,67original records exact,48recomputed. Allfail landinggeometry, prescribedcommandmargin/time; no forceLP/filter/runtime/physicaladmission. Independent348geometry/345finitework checks pass arithmetic only. Last340solve hits80iterationlimit. Original340docstring global-infeasibility wording corrected explicitly in report342/scope_correction.json; originalsource/data retained. Local sequential failures are not global impossibility. Next examine boundedmultiknot coordination, retaining physical/quiet/authority/filter/rate/material criteria. All8 unresolved/all64 protected;video307unchanged. Read closure343.\n')
 s=read(status);s.update(updated_utc=now,utc=now,phase='offline_nominal_phase_coordination_failed_repair_unresolved',task_complete=False,repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,latest_running_diagnostic=None,latest_report=str(D/'progress_report_342'),latest_diagnosis=str(D/'minimum_extension_path_340'),latest_offline_path=str(D/'minimum_extension_path_340'),latest_offline_paths=[str(D/f) for f in ['coordinated_landing_path_336','decoupled_phase_path_338','minimum_extension_path_340']],latest_offline_bound=str(D/'clearance_lift_command_bound_332'),latest_round_force_lp_started=False,latest_offline_audit=str(D/'minimum_extension_audit_341/independent_audit.json'),latest_offline_audit_passed=True,latest_offline_screen_scope='Three115knotphase-coordination paths: landinggeometry and prescribedcommandmargin/time fail; independent arithmetic passed, no newforceLP/filter/runtime/physicaladmission',latest_handoff=str(D/'handoff_update_343.txt'),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),prior_recheck=str(D/'prior_recheck_after_phase_coordination_343.json'),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_optimizer_updates=0,notes=notes);status.write_text(json.dumps(s,indent=2))
 c=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=0,new_initial_settling_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',best_development_clean=64,latest_physical_candidate='roll_feedback_probe_273',prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_342',video='all_attempts_video_307/all_attempts.mp4',video_unchanged=True,local_review=review,next=notes)
 (D/'closure_record_343.json').write_text(json.dumps(c,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],protected_completed_files=len(protected),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
