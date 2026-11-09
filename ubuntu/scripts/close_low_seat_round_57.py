"""Preserve offline stance studies and update mutable handoffs; task unfinished."""
from pathlib import Path
import json,hashlib,datetime,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fresh(p,x):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_51.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).is_file() or sha(p)!=h];assert not bad,(name,bad);checks.append(dict(manifest=name,checked=len(m),passed=True,mismatches=[]));print(name,len(m),'PASS',flush=True)
 fresh(D/'prior_recheck_after_stance_57.json',dict(utc=utc,passed=True,checks=checks))
 evidence=read(D/'completed_probes_preservation_51.json')
 dirs=['stance_endpoint_screen_52','supported_step_screen_53','supported_step_screen_v2_53','stance_independent_audit_54','progress_report_55','supported_step_commands_v3_56','supported_step_commands_v4_56','stance_independent_audit_v3_56','progress_report_57']
 for name in dirs:
  for p in sorted((D/name).rglob('*')):
   if p.is_file() and not p.is_symlink():evidence[str(p.resolve())]=sha(p)
 for number in range(52,58):
  for p in list((R/'scripts').glob(f'*_{number}.py'))+list(D.glob(f'*{number}*.log')):
   if p.is_file():evidence[str(p.resolve())]=sha(p)
 for name in ['progress_report_55_local_review.json','progress_report_57_local_review.json','handoff_update_55.txt','handoff_update_57.txt']:
  p=D/name;evidence[str(p)]=sha(p)
 fresh(D/'completed_probes_preservation_57.json',evidence)
 notes=(D/'handoff_update_57.txt').read_text();report=read(D/'progress_report_57/summary.json');statusfile=D/'status.json';shutil.copy2(statusfile,D/'status_before_closure_57.json');status=read(statusfile)
 status.update(utc=utc,phase='stance_paths_fail_model_prerequisites_repair_unresolved',latest_running_candidate=None,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_report='progress_report_57',latest_diagnostic='progress_report_57/step_diagnosis.json',latest_diagnostic_audit='stance_independent_audit_v3_56/independent_audit.json',latest_model_preparation='supported_step_commands_v4_56/results.json',latest_preservation='prior_recheck_after_stance_57.json',initial_prior_hash_check='prior_recheck_after_stance_57.json',completed_evidence_manifest='completed_probes_preservation_57.json',completed_evidence_count=len(evidence),local_report_original_hashes_verified=11,local_supplemental_hashes_verified=2,local_report_visually_inspected=True,scope='Offline stance endpoints improve clearance but deepest loaded assumption fails3mm. Both tested explicit step paths fail timing/full-foot orientation. Command-only correction preserves leg/path results and restores actual constant nonleg/finger commands; originals retained. No physical repair, new effort, PPO or holdout.',task_complete=False,repair_complete=False,main_training_started=False,fresh_holdout_started=False,policy_promoted=False,new_physical_ticks_since43=0,new_optimizer_updates_since43=0)
 statusfile.write_text(json.dumps(status,indent=2)+'\n');handoff=R/'docs/LOW_SEAT_CONTACT_STATUS.md';shutil.copy2(handoff,D/'handoff_before_57.md');handoff.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+handoff.read_text())
 protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,D/'protocol_before_57.md');protocol.write_text(protocol.read_text()+'\n\nOffline stance prerequisite52–57 (no physics/runtime/criteria change): corrected cooked model, hypothetical common60mm foot shifts and under38degree endpoint research limit; deepest assumed seat-supported endpoint still fails3mm. Explicit60/100mm foot-lift/airborne-translation/land paths with full6joint IK and static support both fail foot orientation and the1–15s command-time screen. No step or endpoint is executed.56 corrects offline non-leg command construction to actual recorded constant targets, including all14 fixed fingers; all leg commands, kinematics and command arc unchanged. Original53 failed non-leg contract and serialization/recovery assumptions remain preserved. Independent audit passes arithmetic and fixed-command checks, not physical admission. Existing64-case retention and8-failure removal prerequisite remains; no PPO/freshholdout/promotion.\n')
 closure=dict(utc=utc,task_complete=False,repair_complete=False,notes=notes,report=report,local_review=read(D/'progress_report_57_local_review.json'),preserved_prior_count=checks[0]['checked'],preserved_previous_evidence_count=checks[1]['checked'],new_completed_evidence_count=len(evidence),preservation_sha256=sha(D/'completed_probes_preservation_57.json'),handoff_sha256=sha(handoff),protocol_sha256=sha(protocol),new_physical_ticks=0,new_optimizer_updates=0,workers_running=False,next=report['next_requirement']);fresh(D/'closure_record_57.json',closure);print(json.dumps(dict(utc=utc,evidence=len(evidence),task_complete=False,new_physics=0),indent=2))
if __name__=='__main__':main()
