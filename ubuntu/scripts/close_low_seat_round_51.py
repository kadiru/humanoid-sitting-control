"""Preserve diagnosis51 and update mutable handoffs; task remains unfinished."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fresh(p,v):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_43.json']:
  m=read(D/name);bad=[p for p,h in m.items() if not Path(p).is_file() or sha(p)!=h];assert not bad,(name,bad)
  checks.append(dict(manifest=name,checked=len(m),passed=True,mismatches=[]));print(name,len(m),'PASS',flush=True)
 fresh(D/'prior_recheck_after_collision_model_51.json',dict(utc=utc,passed=True,checks=checks))
 evidence=read(D/'completed_probes_preservation_43.json')
 dirs=['rolling_contact_screen_44','rolling_contact_screen_v2_44','support_authority_45','model_audit_46','model_audit_v2_46','asset_contact_47','cooked_contact_v2_47','cooked_geometry_48','remaining_cooked_meshes_49','remaining_cooked_meshes_v2_49','cooked_model_50','terminal_cooked_screen_50','progress_report_51']
 for name in dirs:
  for p in sorted((D/name).rglob('*')):
   if p.is_file() and not p.is_symlink():evidence[str(p.resolve())]=sha(p)
 scripts=['screen_low_seat_rolling_contact_44.py','screen_low_seat_rolling_contact_v2_44.py','diagnose_low_seat_support_authority_45.py','audit_low_seat_model_46.py','audit_low_seat_model_v2_46.py','inspect_low_seat_collision_cooking_47.py','inspect_low_seat_collision_cooking_v2_47.py','diagnose_low_seat_cooked_geometry_48.py','inspect_low_seat_remaining_cooking_49.py','inspect_low_seat_remaining_cooking_v2_49.py','screen_low_seat_cooked_terminal_50.py','report_low_seat_collision_51.py','close_low_seat_round_51.py']
 for name in scripts:
  p=R/'scripts'/name;evidence[str(p)]=sha(p)
 for number in range(44,52):
  for p in D.glob(f'*{number}*.log'):
   if p.is_file():evidence[str(p.resolve())]=sha(p)
 for name in ['collision_query_revision_47.json','progress_report_51_local_review.json']:
  p=D/name;evidence[str(p)]=sha(p)
 fresh(D/'completed_probes_preservation_51.json',evidence)
 report=read(D/'progress_report_51/summary.json');notes=(D/'handoff_update_51.txt').read_text()
 statusfile=D/'status.json';shutil.copy2(statusfile,D/'status_before_closure_51.json');status=read(statusfile)
 status.update(utc=utc,phase='collision_model_discrepancy_diagnosed_repair_unresolved',latest_running_candidate=None,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_report='progress_report_51',latest_diagnostic='cooked_geometry_48/recorded_comparison.json',latest_diagnostic_audit='progress_report_51/independent_model_audit_51.json',latest_model_preparation='terminal_cooked_screen_50/results.json',latest_preservation='prior_recheck_after_collision_model_51.json',initial_prior_hash_check='prior_recheck_after_collision_model_51.json',completed_evidence_manifest='completed_probes_preservation_51.json',completed_evidence_count=len(evidence),local_report_original_hashes_verified=14,scope=report['finding'],task_complete=False,repair_complete=False,main_training_started=False,fresh_holdout_started=False,policy_promoted=False,new_physical_ticks_since43=0,new_optimizer_updates_since43=0)
 statusfile.write_text(json.dumps(status,indent=2)+'\n')
 handoff=R/'docs/LOW_SEAT_CONTACT_STATUS.md';shutil.copy2(handoff,D/'handoff_before_51.md');handoff.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+handoff.read_text())
 protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,D/'protocol_before_51.md')
 protocol.write_text(protocol.read_text()+'\n\nOffline planning-model correction51 (no runtime/physics/criteria change): configured USD raw vertices match the frozen model, but the installed PhysX convex cooking query changes the pelvis support surface by about5–6mm in recorded leaning contact directions. Existing raw hull paths therefore misestimate legal pelvis contact; original failed trials/screens remain preserved.32 mesh queries completed with no environment/reset/physics step. cooked_model_50/model.json is a SEPARATE offline planning-collision model; joint/inertial/default metadata and6primitives unchanged. Do not replace visual/perception self-mask geometry or measured inputs with it. Independent source/FK/clearance/zero-tick checks passed as audits, not physical admission. Corrected bounded endpoint50 uses the same fixedfeet/arms/bounds and38degree research reserve: deepest36/+20 remains below3mm even with hypotheticalseat support. No executable path, contact-transition certificate or impossibility claim. Future repair must include physically feasible load transfer and preserve64 clean cases while removing8 failures before PPO; freshholdout staysclosed.\n')
 closure=dict(utc=utc,task_complete=False,repair_complete=False,notes=notes,report=report,local_review=read(D/'progress_report_51_local_review.json'),preserved_prior_count=checks[0]['checked'],preserved_previous_evidence_count=checks[1]['checked'],new_completed_evidence_count=len(evidence),preservation_sha256=sha(D/'completed_probes_preservation_51.json'),handoff_sha256=sha(handoff),protocol_sha256=sha(protocol),new_physical_ticks=0,new_optimizer_updates=0,workers_running=False,next=report['next'])
 fresh(D/'closure_record_51.json',closure);print(json.dumps(dict(utc=utc,evidence=len(evidence),task_complete=False,new_physics=0),indent=2))
if __name__=='__main__':main()
