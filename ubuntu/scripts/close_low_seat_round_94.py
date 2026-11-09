"""Preserve completed offline evidence; task remains active and95 may run."""
from pathlib import Path
import json,hashlib,datetime,re,subprocess,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert read(D/'prior_recheck_during_contact_path_91.json')['passed'];assert read(D/'contact_path_audit_v2_92/independent_audit.json')['passed'];assert (D/'progress_report_93_local_review.json').exists()
 for f in ['completed_probes_preservation_94.json','closure_record_94.json','status_before_round_94.json','handoff_before_round_94.md']:assert not (D/f).exists(),f
 checks=[]
 for mf in ['prior_hashes.json','completed_probes_preservation_v2_64.json']:
  m=read(D/mf);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];checks.append(dict(manifest=mf,checked=len(m),mismatches=bad,passed=not bad));assert not bad,bad[:5]
 prior=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks);(D/'prior_recheck_after_contact_planning_94.json').write_text(json.dumps(prior,indent=2))
 old=read(D/'completed_probes_preservation_v2_64.json');protected=dict(old)
 for p in D.iterdir():
  match=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not match or not 65<=int(match[1])<=94 or p.name in ['closure_94.log','completed_probes_preservation_94.json','closure_record_94.json']:continue
  if p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():protected[str(f)]=sha(f)
  elif p.is_file():protected[str(p)]=sha(p)
 for p in (R/'scripts').glob('*.py'):
  match=re.search(r'_(\d+)\.py$',p.name)
  if match and 65<=int(match[1])<=94:protected[str(p)]=sha(p)
 for name in ['progress_report_74_local_review.json','progress_report_88_local_review.json','progress_report_93_local_review.json']:
  p=D/name
  if p.exists():protected[str(p)]=sha(p)
 mf=D/'completed_probes_preservation_94.json';mf.write_text(json.dumps(protected,indent=2));assert all(sha(p)==h for p,h in protected.items())
 pending=[R/'scripts/refine_low_seat_landing_ankle_95.py',R/'scripts/audit_low_seat_landing_ankle_96.py',D/'landing_ankle_revision_95.json'];pending_manifest={str(p):sha(p) for p in pending};(D/'pending_sources_95.json').write_text(json.dumps(pending_manifest,indent=2))
 status=D/'status.json';doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';shutil.copy2(status,D/'status_before_round_94.json');shutil.copy2(doc,D/'handoff_before_round_94.md');notes=(D/'handoff_update_94.txt').read_text();doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text());running=subprocess.run(['tmux','has-session','-t','low_seat_landing_ankle_95'],capture_output=True).returncode==0
 st=read(status);st.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='bounded_offline_landing_ankle_path_running' if running else 'offline_contact_repair_unresolved',task_complete=False,repair_complete=False,main_training_started=False,fresh_holdout_started=False,policy_promoted=False,physical_jobs_running=False,model_jobs_running=running,latest_running_candidate='low_seat_landing_ankle_95' if running else None,latest_report=str(D/'progress_report_93'),latest_completed_model=str(D/'contact_path_screen_v2_91/results.json'),latest_diagnostic_audit=str(D/'contact_path_audit_v2_92/independent_audit.json'),prior_recheck=str(D/'prior_recheck_after_contact_planning_94.json'),completed_evidence_manifest=str(mf),latest_handoff=str(D/'handoff_update_94.txt'),best_verified_clean=64,best_verified_total=72,remaining_slipping=8,remaining_housing_contacts=2,notes='Model95 and independent96 remain bounded offline checks; no physics/PPO/holdout admission.');status.write_text(json.dumps(st,indent=2))
 closure=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_complete=False,repair_complete=False,new_physical_ticks=0,new_optimizer_updates=0,holdout_started=False,prior_recheck=prior,completed_evidence_manifest=str(mf),protected_completed_files=len(protected),manifest_sha256=sha(mf),pending95_sources=pending_manifest,model95_running=running,physical_workers_running=False,report='progress_report_93',local_review=read(D/'progress_report_93_local_review.json'),handoff_sha256=sha(D/'handoff_update_94.txt'),next='Inspect existing95 result and independent96, without restarting. No model/endpoint audit authorizes physical execution or PPO. Complete filtered-command/sensed-load/dynamic transition and all72 matched development physical gates remain required. Video59 remains current; no motion added.');(D/'closure_record_94.json').write_text(json.dumps(closure,indent=2));print(json.dumps(dict(prior_files=checks[0]['checked'],previous_evidence=checks[1]['checked'],completed_evidence=len(protected),model95_running=running,task_complete=False)),flush=True)
if __name__=='__main__':main()
