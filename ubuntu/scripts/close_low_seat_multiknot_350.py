"""Preserve completed344–349 offline evidence; physical repair and PPO remain open."""
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
 assert not (D/'closure_record_350.json').exists()
 folders=['multiknot_preparation_344','multiknot_path_344','multiknot_audit_345','multiknot_schedule_346','multiknot_schedule_audit_347','multiknot_timing_diagnosis_v2_348','multiknot_metadata_recovery_348','progress_report_349']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,(folder,p)
 for folder in ['multiknot_audit_345','multiknot_schedule_audit_347']:
  a=read(D/folder/'independent_audit.json');assert a['passed'] and not a['physical_admission']
 for name in ['review_SHA256.json','readable_SHA256.json']:
  for p,h in read(D/'progress_report_349'/name).items():assert sha(D/'progress_report_349'/p)==h
 review=read(D/'progress_report_349_local_review.json');assert all(review[k] for k in ['local_hashes_verified','visual_layout_inspected','browser_visually_inspected','embedded_video_loaded'])
 s346=read(D/'multiknot_schedule_346/results.json');assert all(not r['complete'] and not r['time_gate'] for r in s346['rows'])
 a347=read(D/'multiknot_schedule_audit_347/independent_audit.json');assert all(not r['hover_guard_reached'] and not r['quiet_load_fixture_exercised'] for r in a347['rows'])
 workers=[s for s in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines() if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|build|bound|audit|refine|optimize|diagnose|report|review|schedule|render|verify|finalize|launch|prepare|check|screen)_low_seat_\S+\.py',s)]
 assert not workers,workers
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_343.json','supplemental_preservation_343.json']:
  data=read(D/name);bad=[p for p,h in data.items() if not Path(p).exists() or sha(p)!=h];assert not bad,bad[:5]
  checks.append(dict(manifest=name,checked=len(data),passed=True,mismatches=[]))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prior=dict(utc=now,passed=True,checks=checks)
 priorpath=D/'prior_recheck_after_multiknot_350.json';priorpath.write_text(json.dumps(prior,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md'
 for p,name in [(status,'status_before_round_350.json'),(doc,'handoff_before_round_350.md'),(protocol,'protocol_before_round_350.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 notes=(D/'handoff_update_350.txt').read_text();protected=read(D/'completed_probes_preservation_343.json');protected.update(read(D/'supplemental_preservation_343.json'))
 for p in D.iterdir():
  m=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not m or not 344<=int(m[1])<=350 or p.name in ['closure_350.log','closure_record_350.json','completed_probes_preservation_350.json']:continue
  for f in (p.rglob('*') if p.is_dir() else [p]):
   if f.is_file() and not f.is_symlink():protected[str(f)]=sha(f)
 for p in (R/'scripts').glob('*.py'):
  m=re.search(r'_(\d+)\.py$',p.name)
  if m and 344<=int(m[1])<=350:protected[str(p)]=sha(p)
 protected[str(D/'progress_report_349_local_review.json')]=sha(D/'progress_report_349_local_review.json')
 mf=D/'completed_probes_preservation_350.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 with protocol.open('a') as f:f.write('\n\nRound350: one simultaneous48-knot nominal path344, full115geometry/staticcommands and independent345 passed. Optimisticglobal.15commandtime12.821948s does NOT admit execution. Separateposemetric14.262742s fails. Command-only346 three810tickprefixes stop BEFOREhover; independent347CPU/GPUfloat32 exact, but quiet/loadguards NOTexercised. Conditionalcontinuouspolyline actualreservedrates15.067909s +normalextra1.701795s +synthetic.5hover=17.269704s FAIL13.5. Diagnosis348v2 shows inverse-target test excludes a legal oneULPforwardfilter movement; not absolute filterdeadlock or correctedmultijointschedule. Original348metadatafailure preserved, no model/schedule/physics rerun. No newphysical/PPO/holdout; all64protected/all8unresolved; video307unchanged. Next completeformulation must include actualreservedrates/normal/quiettiming andforwardfloat32desiredheadroom, not repeatoptimistic344 orfixonly346inverse. All priorcriteria/authority/materials remain. See handoff350/report349.\n')
 s=read(status);s.update(updated_utc=now,utc=now,phase='offline_multiknot_static_pass_guarded_timing_failed_repair_unresolved',task_complete=False,repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_running_candidate=None,latest_running_diagnostic=None,latest_report=str(D/'progress_report_349'),latest_diagnosis=str(D/'multiknot_timing_diagnosis_v2_348'),latest_offline_path=str(D/'multiknot_path_344'),latest_offline_paths=[str(D/'multiknot_path_344')],latest_offline_bound=str(D/'multiknot_timing_diagnosis_v2_348/results.json'),latest_round_force_lp_started=False,latest_offline_audit=str(D/'multiknot_schedule_audit_347/independent_audit.json'),latest_offline_audit_passed=True,latest_offline_screen_scope='115sampledgeometry/staticcommands pass; reserved/normal/quiet continuouspath timing fails17.269704s; 3x810 command-onlyprefixes stop beforehover, no guardcoverage orphysicaladmission',latest_handoff=str(D/'handoff_update_350.txt'),latest_preservation=str(mf),completed_evidence_manifest=str(mf),completed_evidence_count=len(protected),prior_recheck=str(priorpath),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_optimizer_updates=0,latest_round_offline_nonlinear_iterations=20,latest_round_command_ticks_per_device=2430,latest_round_hover_guard_exercised=False,notes=notes);status.write_text(json.dumps(s,indent=2))
 c=dict(utc=now,task_complete=False,repair_complete=False,new_active_physics_ticks=0,new_initial_settling_ticks=0,new_optimizer_updates=0,new_full20s_trials=0,new_diagnostic_prefixes=0,new_aborted_case_attempts=0,holdout_started=False,physical_admission_passed=False,best_verified_controller='current_retention_09',best_development_clean=64,latest_physical_candidate='roll_feedback_probe_273',offline_static_geometry_commands_passed=True,actual_schedule_complete=False,hover_guard_exercised=False,conditional_continuous_timing_s=17.269704231935616,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),physical_workers_running=False,model_workers_running=False,audit_workers_running=False,report='progress_report_349/readable.html',video='all_attempts_video_307/all_attempts.mp4',video_unchanged=True,local_review=review,next=notes)
 (D/'closure_record_350.json').write_text(json.dumps(c,indent=2));print(json.dumps(dict(utc=now,prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],supplemental=checks[2]['checked'],protected_completed_files=len(protected),manifest_sha256=sha(mf),workers_running=False,task_complete=False)),flush=True)
if __name__=='__main__':main()
