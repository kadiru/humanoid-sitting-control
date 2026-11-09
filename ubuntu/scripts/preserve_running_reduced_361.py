"""Preserve completed coordinate evidence and snapshot the still-active single361 solve; no closure."""
from pathlib import Path
import json,hashlib,datetime,subprocess,re,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'coordinate_evidence_preservation_361.json').exists()
 folders=['constraint_coordinates_357','constraint_coordinates_audit_358','projected_complete_path_v2_359','projected_helper_scope_recovery_359','projected_path_audit_360','reduced_preflight_recovery_361','reduced_path_preparation_v2_361']
 counts=[]
 for folder in folders:
  v=read(D/folder/'SHA256.json')
  for p,h in v.items():assert sha(D/folder/p)==h,(folder,p)
  counts.append(len(v))
 for folder in ['constraint_coordinates_audit_358','projected_path_audit_360']:assert read(D/folder/'independent_audit.json')['passed']
 checks=[]
 for name in ['completed_probes_preservation_356.json','supplemental_preservation_356.json']:
  v=read(D/name);bad=[p for p,h in v.items() if sha(p)!=h];assert not bad,bad[:5];checks.append(dict(manifest=name,files=len(v),passed=True))
 ps=subprocess.run(['ps','-eo','pid,etime,args'],capture_output=True,text=True,check=True).stdout;workers=[s for s in ps.splitlines() if re.search(r'^\s*\d+\s+\S+\s+\S*python\S* scripts/build_low_seat_reduced_path_v2_361.py$',s)];assert len(workers)==1,workers;assert not (D/'reduced_path_v2_361/results.json').exists()
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();history=read(D/'reduced_path_v2_361/iterations.json') if (D/'reduced_path_v2_361/iterations.json').exists() else []
 active=dict(utc=now,task_complete=False,repair_complete=False,model_running=True,unified_exec_session=73038,worker=workers[0],latest_callback=history[-1] if history else None,callback_count=len(history),intermediate_not_admitted=True,main_exit_status=None,maximum_optimizer_runs=1,maximum_iterations=30,physical_ticks=0,policy_optimizer_updates=0,audit362_executed=False,report363_executed=False,completed_manifests=folders,completed_original_entries=sum(counts),prior_checks=checks)
 (D/'active_run_361.json').write_text(json.dumps(active,indent=2));notes=(D/'handoff_running_361.txt').read_text()
 for path,target in [(D/'status.json','status_before_running_361.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_running_361.md'),(R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md','protocol_before_running_361.md')]:
  assert not (D/target).exists();shutil.copy2(path,D/target)
 s=read(D/'status.json');s.update(utc=now,updated_utc=now,phase='reduced_complete_path_v2_361_running_offline',task_complete=False,repair_complete=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,physical_jobs_running=False,model_jobs_running=True,audit_jobs_running=False,latest_running_candidate='reduced_path_v2_361_OFFLINE_ONLY',latest_running_diagnostic=None,latest_report=str(D/'progress_report_355'),latest_offline_path=str(D/'projected_complete_path_v2_359'),latest_offline_paths=[str(D/'projected_complete_path_v2_359')],latest_offline_audit=str(D/'projected_path_audit_360/independent_audit.json'),latest_offline_audit_passed=True,latest_offline_geometry_passed=True,latest_offline_static_commands_passed=False,latest_offline_nlp_passed=False,latest_offline_time_gate_passed=False,latest_offline_gate_scope='Completed359 restored all115 sampledgeometry but command/rate/headroom/hover/time remainfailed;361 single reduced-coordinate solve RUNNING, nofinalresult.',latest_offline_screen_scope='359 geometrypass; .029961109rad commandmargin and13.843043s reserved/normal/quiet metric fail, noactualfilter.361pending.',latest_diagnosis=str(D/'constraint_coordinates_357'),latest_handoff=str(D/'handoff_running_361.txt'),latest_round_new_full20s_trials=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_new_active_ticks=0,latest_round_initial_settling_ticks=0,latest_round_optimizer_updates=0,latest_round_command_ticks_per_device=0,latest_round_offline_nonlinear_iterations=None,latest_round_hover_guard_exercised=False,notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier sealed entries follow.\n\n'+doc.read_text())
 with (R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md').open('a') as f:f.write('\n\nActive361: completed309-coordinate assessment357/analytic358 and completeprojected115path359v2/360. Geometryrestored but commands/timing fail. One NEW reduced-coordinate694free-variable completepath361v2 runs with original physical/controller/command/rate/quiet criteria; dependentkinematic equations solved internally and planningbounds retained asouterconstraints. Max30iterations, no restart/extension. Original359helper-scope and361preparationfixture failures preserved; no completed scientific/physicalrun repeated. Corrected361preflight passed2e-6 derivativebound; numerical root tolerance tightened1e-13, no physical waiver. Audit362/report363 unexecutedpendingmainexit. No physics/PPO/holdout/video motion. This active snapshot is not closure or repair.\n')
 protected=read(D/'completed_probes_preservation_356.json');protected.update(read(D/'supplemental_preservation_356.json'))
 for p in D.iterdir():
  mat=re.search(r'_(\d+)(?:\.[^.]*)?$',p.name)
  if not mat or not 357<=int(mat[1])<=363 or p.name in ['reduced_path_v2_361','reduced_path_v2_361.log','coordinate_evidence_preservation_361.json','preserve_running_reduced_361.log']:continue
  for q in p.rglob('*') if p.is_dir() else [p]:
   if q.is_file() and not q.is_symlink():protected[str(q)]=sha(q)
 for p in (R/'scripts').glob('*.py'):
  mat=re.search(r'_(\d+)\.py$',p.name)
  if mat and 357<=int(mat[1])<=363:protected[str(p)]=sha(p)
 mf=D/'coordinate_evidence_preservation_361.json';mf.write_text(json.dumps(protected,indent=2));print(json.dumps(dict(utc=now,completed_original_entries=sum(counts),protected_files=len(protected),manifest_sha256=sha(mf),worker=workers[0],callbacks=len(history),task_complete=False,closure_created=False)),flush=True)
if __name__=='__main__':main()
