"""Preserve completed planning/video evidence and update mutable status only."""
from pathlib import Path
import json,hashlib,datetime,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def fresh(p,r):
 assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n')
def main():
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_57.json']:
  a=read(D/name);bad=[p for p,h in a.items() if not Path(p).is_file() or sha(p)!=h];assert not bad,(name,bad);checks.append(dict(manifest=name,checked=len(a),passed=True,mismatches=[]));print(name,len(a),'PASS',flush=True)
 fresh(D/'prior_recheck_after_planning_video_64.json',dict(utc=utc,passed=True,checks=checks))
 evidence=read(D/'completed_probes_preservation_57.json')
 dirs=['spatial_contact_screen_58','all_attempts_video_59','offline_recovery_58_59','spatial_contact_screen_v2_58','all_attempts_video_v2_59','all_attempts_video_audit_60','spatial_contact_audit_61','single_step_transition_screen_62','single_step_audit_63','progress_report_64']
 for name in dirs:
  for p in sorted((D/name).rglob('*')):
   if p.is_file() and not p.is_symlink():evidence[str(p.resolve())]=sha(p)
 for number in range(58,65):
  for p in list((R/'scripts').glob(f'*_{number}.py'))+list(D.glob(f'*{number}*.log')):
   if p.is_file():evidence[str(p.resolve())]=sha(p)
 for name in ['progress_report_64_local_review.json','handoff_update_64.txt','single_step_revision_62.json']:
  p=D/name;evidence[str(p)]=sha(p)
 fresh(D/'completed_probes_preservation_64.json',evidence)
 notes=(D/'handoff_update_64.txt').read_text();report=read(D/'progress_report_64/summary.json');sf=D/'status.json';shutil.copy2(sf,D/'status_before_closure_64.json');s=read(sf)
 s.update(utc=utc,phase='planning_prerequisites_failed_all_attempts_video_ready_repair_unresolved',latest_running_candidate=None,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_report='progress_report_64',latest_model_preparation='single_step_transition_screen_62/results.json',latest_diagnostic_audit='single_step_audit_63/independent_audit.json',latest_preservation='prior_recheck_after_planning_video_64.json',initial_prior_hash_check='prior_recheck_after_planning_video_64.json',completed_evidence_manifest='completed_probes_preservation_64.json',completed_evidence_count=len(evidence),latest_video='all_attempts_video_v2_59/all_attempts.mp4',latest_video_audit='all_attempts_video_audit_60/independent_audit.json',video_ready_locally=True,video_duration_s=58.4,video_complete_trials=696,video_successes=405,video_failures=291,local_report_original_hashes_verified=9,local_supplemental_hashes_verified=2,local_report_visually_inspected=True,scope=report['next_requirement'],task_complete=False,repair_complete=False,main_training_started=False,fresh_holdout_started=False,policy_promoted=False,new_physical_ticks_since43=0,new_optimizer_updates_since43=0)
 sf.write_text(json.dumps(s,indent=2)+'\n');hd=R/'docs/LOW_SEAT_CONTACT_STATUS.md';shutil.copy2(hd,D/'handoff_before_64.md');hd.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+hd.read_text());pr=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(pr,D/'protocol_before_64.md');pr.write_text(pr.read_text()+'\n\nOffline planning and video58–64: no runtime/physics/criterion change. Corrected cooked fixed-foot spatial endpoints58 retain deepest clearance failure; full6joint static replay61 passed arithmetic only. Single lifted right rearward60/100mm steps62 with closest inset stance COM and coupled crouch both fail corrected clearance;100mm also foot pose. Independent63 preserves failed gates and clarifies pose schedule is not actual PD command timing. No path executed. User requested a complete sub-minute all-attempt video:59/60 exports58.4sH264 all696recorded full trials and72prefixes, with72aborted cases without saved motion explicitly disclosed. No hidden failures or fabricated motion. Existing64-retention/8-removal prerequisite remains before PPO; freshholdout closed.\n')
 fresh(D/'closure_record_64.json',dict(utc=utc,task_complete=False,repair_complete=False,notes=notes,report=report,local_review=read(D/'progress_report_64_local_review.json'),preserved_prior_count=checks[0]['checked'],preserved_previous_evidence_count=checks[1]['checked'],new_completed_evidence_count=len(evidence),preservation_sha256=sha(D/'completed_probes_preservation_64.json'),handoff_sha256=sha(hd),protocol_sha256=sha(pr),new_physical_ticks=0,new_optimizer_updates=0,workers_running=False,next=report['next_requirement']));print(json.dumps(dict(utc=utc,evidence=len(evidence),task_complete=False,video_ready=True)))
if __name__=='__main__':main()
