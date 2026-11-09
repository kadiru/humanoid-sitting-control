"""Preserve completed evidence and update mutable handoff; not task completion."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,v):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_39.json']:
  entries=read(D/name);bad=[str(p) for p,h in entries.items() if not Path(p).is_file() or sha(p)!=h]
  checks.append(dict(manifest=name,checked=len(entries),mismatches=bad,passed=not bad));assert not bad,(name,bad)
  print(name,len(entries),'PASS',flush=True)
 write(D/'prior_recheck_after_dual_view_43.json',dict(utc=utc,passed=True,checks=checks))
 # The completed trial source list is independently frozen in its revision.
 revision=read(D/'dual_view_approach_40/revision.json')
 for path,digest in revision['sources'].items():assert sha(path)==digest,('Frozen40 source changed',path)
 assert sha(D/'dual_view_preparation_40/preparation.json')==revision['preparation_sha256']
 evidence=read(D/'completed_probes_preservation_39.json')
 for name in ['dual_view_approach_40','measurement_report_39','support_diagnosis_41','terminal_screen_42','progress_report_43']:
  for p in sorted((D/name).rglob('*')):
   if p.is_file() and not p.is_symlink() and 'baseline' not in p.relative_to(D/name).parts:evidence[str(p.resolve())]=sha(p)
 for pattern in ['*40*.log','*41*.log','*42*.log','*43*.log','status_dual_view_approach_40.json','progress_report_43_local_review.json']:
  for p in D.glob(pattern):
   if p.is_file():evidence[str(p.resolve())]=sha(p)
 for name in ['diagnose_low_seat_support_41.py','low_seat_terminal_contract_42.py','screen_low_seat_terminal_42.py','report_low_seat_terminal_43.py','close_low_seat_round_43.py']:
  p=R/'scripts'/name;evidence[str(p)]=sha(p)
 write(D/'completed_probes_preservation_43.json',evidence)
 report=read(D/'progress_report_43/summary.json');notes=(D/'handoff_update_43.txt').read_text()
 old=D/'status.json';shutil.copy2(old,D/'status_before_closure_43.json');status=read(old)
 status.update(utc=utc,phase='dual_view40_rejected_planning_requirement_corrected',latest_candidate='dual_view_approach_40',latest_candidate_outcome=read(D/'status_dual_view_approach_40.json')['summary'],latest_independent_audit_passed=False,latest_exact_preservation_gate_passed=False,latest_software_audit_passed=True,latest_running_candidate=None,physical_jobs_running=False,model_jobs_running=False,audit_jobs_running=False,latest_report='progress_report_43',latest_diagnostic='support_diagnosis_41',latest_diagnostic_audit='dual_view_approach_40/independent_dual_view_audit_40.json',latest_model_preparation='terminal_screen_42/results.json',latest_preservation='prior_recheck_after_dual_view_43.json',initial_prior_hash_check='prior_recheck_after_dual_view_43.json',completed_evidence_manifest='completed_probes_preservation_43.json',completed_evidence_count=len(evidence),local_report_original_hashes_verified=16,local_supplemental_hashes_verified=2,scope='Repair unresolved. Current-view coverage passed; physical40 rejected. Planned.9rad terminal exceeds existing40degree support criterion. Corrected future model screen, no physical criterion change. Preserve64 successes; eight failures remain before PPO.',task_complete=False,repair_complete=False,main_training_started=False,fresh_holdout_started=False,policy_promoted=False)
 old.write_text(json.dumps(status,indent=2)+'\n')
 handoff=R/'docs/LOW_SEAT_CONTACT_STATUS.md';shutil.copy2(handoff,D/'handoff_before_43.md');handoff.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+handoff.read_text())
 protocol=R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md';shutil.copy2(protocol,D/'protocol_before_43.md')
 protocol.write_text(protocol.read_text()+'\n\nPlanning correction43 (no physical/evaluation change): the unchanged resting objective requires trunk tilt<=40degrees and legal seat normal load>=0.05BW, in addition to all other criteria. The.9rad endpoint in prior17/31/36/40 is51.6degrees and cannot qualify even under perfect tracking. Original sources/results/model-screen assertions remain preserved; those screens were incomplete, not certificates. New low_seat_terminal_contract_42.py derives the40degree limit from the existing objective and uses a separate2degree research reserve for future model endpoints. Offline42 fixes the previously tested arm offsets and measured1s foot anchors, compares foot-only versus hypothetical seat-supported endpoints, and does not define an executable contact-transfer motion. Do not execute endpoint-only solutions or unload before sensed support, relax physical criteria, or resume rejected physical jobs. Physical admission and64-case retention remain mandatory before PPO.\n')
 closure=dict(utc=utc,task_complete=False,repair_complete=False,notes=notes,report=report,local_review=read(D/'progress_report_43_local_review.json'),preserved_prior_count=24242,preserved_previous_evidence_count=2392,new_completed_evidence_count=len(evidence),preservation_sha256=sha(D/'completed_probes_preservation_43.json'),handoff_sha256=sha(handoff),protocol_sha256=sha(protocol),minor_execution_note='First offline diagnosis41 shell invocation used a nonexistent absolute Python path. It executed no Python or physics; original log retained, second invocation used existing project IsaacLab/.venv/bin/python. No result overwrite or threshold change.',next='Develop a measured, support-feasible contact transition under the unchanged40degree limit. Corrected42 endpoint availability is not a path; deepest foot-only clearance fails3mm. No blind rate/lean/arm sweep, foot dragging/teleportation, truth substitution, or PPO before full physical repair.')
 write(D/'closure_record_43.json',closure);print(json.dumps(dict(utc=utc,evidence=len(evidence),task_complete=False),indent=2))
if __name__=='__main__':main()
